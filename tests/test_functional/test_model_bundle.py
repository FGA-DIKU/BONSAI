"""Explicit-file bundles preserve predictions and frozen token transformations."""

import json
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import polars as pl
import torch

from bonsai.functional.model_bundle import export_transform, pack_bundle, unpack_bundle
from bonsai.functional.model_loading import load_bundle
from bonsai.modules.networks.bonsai_nets import (
    BonsaiFinetune,
    BonsaiPretrain,
    BonsaiValueFinetune,
    BonsaiValuePretrain,
)


class TestModelBundle(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "bundle.zip"
        self.checkpoint = self.root / "best.ckpt"
        self.vocabulary = self.root / "vocabulary.pt"
        self.config = self.root / "data.yaml"
        self.config.write_text("tokenizer: {}\n")
        self.numeric = self.root / "numeric_metadata.json"
        pl.DataFrame(
            {
                "code": ["C2"],
                "numeric/bin_1": [0.0],
                "numeric/bin_99": [10.0],
                "numeric/bin_edges": [[0.0, 0.5, 1.0]],
                "numeric/bin_representatives": [[0.0, 0.25, 0.75, 1.0]],
                "subject_ids": [[123]],
                "raw_values": [[1.0, 2.0]],
                "count": [2],
            }
        ).write_json(self.numeric)
        self.mapping = self.root / "adaptive_code_mapping.parquet"
        pl.DataFrame(
            {"code": ["C2"], "adaptive/mapped_code": ["C2"], "count": [2]}
        ).write_parquet(self.mapping)
        self.original = self.network()

    def network(self, cls=BonsaiPretrain):
        parameters = {
            "vocab_size": 8,
            "max_seqlen": 16,
            "hidden_size": 64,
            "num_layers": 1,
            "bias": True,
            "dropout": 0.1,
            "attention_dropout": 0.2,
            "causal": True,
            "attn_type": "sdpa",
        }
        if cls in (BonsaiValuePretrain, BonsaiValueFinetune):
            parameters["value_embedding_mode"] = "film"
        if cls in (BonsaiFinetune, BonsaiValueFinetune):
            parameters["predict_token_id"] = 3
        model = cls(**parameters).eval()
        torch.save(
            {
                "state_dict": {
                    **{f"model.{k}": v for k, v in model.state_dict().items()},
                    "training_metric": torch.tensor([123]),
                },
                "hyper_parameters": {**model.hparams, "patient_ids": [123]},
                "callbacks": {"patient_ids": [123]},
                "optimizer_states": [{"unused": 1}],
            },
            self.checkpoint,
        )
        torch.save(
            {"[PAD]": 0, "[SEP]": 1, **{f"C{i}": i for i in range(2, 8)}},
            self.vocabulary,
        )
        return model

    def pack(self, configs=(), include_numeric=True):
        return pack_bundle(
            self.archive,
            checkpoint=self.checkpoint,
            vocabulary=self.vocabulary,
            data_config=self.config,
            numeric_metadata=self.numeric if include_numeric else None,
            code_mapping=self.mapping,
            configs=configs,
        )

    @staticmethod
    def batch():
        return {
            "code": torch.tensor([[2, 3, 4]]),
            "age": torch.ones(1, 3),
            "abspos": torch.ones(1, 3),
            "segment": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "target": torch.tensor([[3, 4, 5]]),
            "numeric_value": torch.tensor([[0.2, float("nan"), 0.4]]),
            "numeric_target": torch.tensor([[0.3, float("nan"), 0.5]]),
        }

    def test_predictions_survive_transfer_for_all_four_networks(self):
        for cls in (
            BonsaiPretrain,
            BonsaiValuePretrain,
            BonsaiFinetune,
            BonsaiValueFinetune,
        ):
            with self.subTest(model=cls.__name__):
                original = self.network(cls)
                self.archive = self.root / f"{cls.__name__}.zip"
                self.pack()
                destination = unpack_bundle(self.archive, self.root / cls.__name__)
                loaded = load_bundle(destination, attn_type="sdpa")
                self.assertIsInstance(loaded.model, cls)
                self.assertFalse(loaded.model.training)
                self.assertEqual(next(loaded.model.parameters()).device.type, "cpu")
                self.assertEqual(loaded.vocabulary["[SEP]"], 1)
                self.assertEqual(
                    pl.read_json(loaded.numeric_metadata)[
                        "numeric/bin_edges"
                    ].to_list(),
                    [[0.0, 0.5, 1.0]],
                )
                self.assertEqual(
                    pl.read_parquet(loaded.code_mapping)[
                        "adaptive/mapped_code"
                    ].to_list(),
                    ["C2"],
                )
                with torch.inference_mode():
                    torch.testing.assert_close(
                        original(self.batch()),
                        loaded.model(self.batch()),
                        rtol=0,
                        atol=0,
                    )

    def test_only_selected_clean_assets_are_exported(self):
        (self.root / "subject_data_train.pt").write_bytes(b"never included")
        before = self.checkpoint.read_bytes()
        extra = self.root / "event.yaml"
        extra.write_text("events: {}\n")
        self.pack(configs=(extra,))
        destination = unpack_bundle(self.archive, self.root / "restored")
        manifest = json.loads((destination / "manifest.json").read_text())
        self.assertEqual(
            set(manifest["files"]),
            {
                "model/model.ckpt",
                "model/vocabulary.pt",
                "settings/000_data.yaml",
                "settings/001_event.yaml",
                "preprocessing/numeric_metadata.json",
                "preprocessing/code_mapping.parquet",
            },
        )
        checkpoint = torch.load(destination / "model/model.ckpt", weights_only=True)
        self.assertEqual(set(checkpoint), {"state_dict", "hyper_parameters"})
        self.assertNotIn("patient_ids", checkpoint["hyper_parameters"])
        self.assertTrue(all(k.startswith("model.") for k in checkpoint["state_dict"]))
        numeric = pl.read_json(destination / manifest["artifacts"]["numeric_metadata"])
        self.assertFalse({"subject_ids", "raw_values", "count"} & set(numeric.columns))
        self.assertEqual(
            pl.read_parquet(
                destination / manifest["artifacts"]["code_mapping"]
            ).columns,
            ["code", "adaptive/mapped_code"],
        )
        self.assertEqual(self.checkpoint.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            self.pack()
        with self.assertRaises(FileExistsError):
            unpack_bundle(self.archive, destination)

    def test_parameter_views_do_not_export_unrelated_storage(self):
        checkpoint = torch.load(self.checkpoint, weights_only=True)
        checkpoint["state_dict"]["model.layernorm.bias"] = torch.arange(
            65, dtype=torch.float32
        )[:64]
        torch.save(checkpoint, self.checkpoint)
        self.pack()
        destination = unpack_bundle(self.archive, self.root / "restored")
        bias = torch.load(destination / "model/model.ckpt", weights_only=True)[
            "state_dict"
        ]["model.layernorm.bias"]
        self.assertEqual(
            bias.untyped_storage().nbytes(), bias.numel() * bias.element_size()
        )

    def test_cli_pack_unpack_and_load_with_explicit_unused_transforms(self):
        commands = [
            [
                "pack",
                "--checkpoint",
                str(self.checkpoint),
                "--vocabulary",
                str(self.vocabulary),
                "--data-config",
                str(self.config),
                "--no-numeric-metadata",
                "--no-code-mapping",
                "--output",
                str(self.archive),
            ],
            ["unpack", str(self.archive), str(self.root / "restored")],
            ["load", str(self.root / "restored"), "--attn-type", "sdpa"],
        ]
        for command in commands:
            result = subprocess.run(
                [sys.executable, "-m", "bonsai.run.model_bundle", *command],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Loaded BonsaiPretrain", result.stdout)
        loaded = load_bundle(self.root / "restored")
        self.assertIsNone(loaded.numeric_metadata)
        self.assertIsNone(loaded.code_mapping)

    def test_invalid_vocabulary_transform_and_weights_are_rejected(self):
        torch.save({"[PAD]": 0, "[SEP]": 99}, self.vocabulary)
        with self.assertRaisesRegex(ValueError, "embedding table"):
            self.pack()
        self.assertFalse(self.archive.exists())
        self.network()
        pl.DataFrame({"code": ["C2"]}).write_json(self.numeric)
        with self.assertRaisesRegex(ValueError, "missing standard"):
            self.pack()
        self.assertFalse(self.archive.exists())
        self.pack(include_numeric=False)
        destination = unpack_bundle(self.archive, self.root / "restored")
        checkpoint = torch.load(destination / "model/model.ckpt", weights_only=True)
        checkpoint["state_dict"].pop("model.layernorm.weight")
        torch.save(checkpoint, destination / "model/model.ckpt")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_bundle(destination)

    def test_incomplete_vocabulary_cannot_leave_output_ids_undefined(self):
        vocabulary = torch.load(self.vocabulary, weights_only=True)
        vocabulary.pop("C5")  # Remaining IDs still fit inside the embedding table.
        torch.save(vocabulary, self.vocabulary)
        with self.assertRaisesRegex(ValueError, "cover.*exactly"):
            self.pack()
        self.assertFalse(self.archive.exists())

    def test_missing_transform_declaration_has_a_clear_error(self):
        self.pack()
        destination = unpack_bundle(self.archive, self.root / "restored")
        path = destination / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["artifacts"].pop("numeric_metadata")
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "use null for unused"):
            load_bundle(destination)

    def test_misspelled_transform_role_is_not_treated_as_code_mapping(self):
        with self.assertRaisesRegex(ValueError, "Unknown transformation role"):
            export_transform(self.mapping, self.root / "wrong.parquet", "code_maping")
        self.assertFalse((self.root / "wrong.parquet").exists())

    def test_unsafe_and_unlisted_archive_members_leave_no_destination(self):
        self.pack()
        with zipfile.ZipFile(self.archive) as bundle:
            contents = {name: bundle.read(name) for name in bundle.namelist()}
        for name, symlink in [
            ("../escape", False),
            ("unlisted.pt", False),
            ("link", True),
        ]:
            archive = self.root / ("symlink.zip" if symlink else "invalid.zip")
            with zipfile.ZipFile(archive, "w") as bundle:
                for member, data in contents.items():
                    bundle.writestr(member, data)
                info = zipfile.ZipInfo(name)
                if symlink:
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(info, b"outside")
            destination = self.root / "invalid"
            with self.assertRaises(ValueError):
                unpack_bundle(archive, destination)
            self.assertFalse(destination.exists())
        self.assertFalse((self.root / "escape").exists())
