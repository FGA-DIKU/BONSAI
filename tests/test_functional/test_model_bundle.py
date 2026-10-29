"""Model bundle round trips and rejection of damaged or unsafe archives."""

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import polars as pl
import torch

from bonsai.functional.model_bundle import discover_artifact, pack_bundle, unpack_bundle


class TestModelBundle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for package in ("bonsai", "ehr2meds"):
            repo = self.root / package
            (repo / package).mkdir(parents=True)
            (repo / "pyproject.toml").write_text("[project]\n")
            (repo / package / "code.py").write_text("# source\n")
            (repo / "configs").mkdir()
            (repo / "configs" / "pipeline.yaml").write_text("stages: []\n")
            (repo / ".env").write_text("SECRET=excluded")
        self.model = self.root / "run"
        self.model.mkdir()
        torch.save(
            {
                "state_dict": {"model.weight": torch.tensor([1.0])},
                "hyper_parameters": {"hidden_size": 32, "patient_ids": [123]},
                "callbacks": {"patient_ids": [123]},
                "optimizer_states": [{"unused": 1}],
            },
            self.model / "best.ckpt",
        )
        torch.save({"[PAD]": 0, "[SEP]": 1}, self.model / "vocabulary.pt")
        (self.model / "subject_data_train.pt").write_bytes(b"excluded")
        (self.model / "hparams.yaml").write_text("hidden_size: 32\n")
        self.archive = self.root / "bundle.zip"

    def pack(self, **kwargs):
        return pack_bundle(
            self.archive,
            bonsai_dir=self.root / "bonsai",
            ehr2meds_dir=self.root / "ehr2meds",
            model_dir=self.model,
            data_config=self.root / "bonsai/configs/pipeline.yaml",
            **({"numeric_metadata": None, "code_mapping": None} | kwargs),
        )

    def test_round_trip_and_selection(self):
        config = self.root / "pipeline.yaml"
        config.write_text("stages: []\n")
        self.pack(configs=(config,))
        destination = unpack_bundle(self.archive, self.root / "restored")
        checkpoint = torch.load(destination / "model/run/best.ckpt", weights_only=True)
        self.assertEqual(set(checkpoint), {"state_dict", "hyper_parameters"})
        self.assertEqual(checkpoint["hyper_parameters"], {"hidden_size": 32})
        self.assertTrue(
            torch.equal(checkpoint["state_dict"]["model.weight"], torch.tensor([1.0]))
        )
        manifest = json.loads((destination / "manifest.json").read_text())
        self.assertIn("settings/preprocessing/000/pipeline.yaml", manifest["files"])
        self.assertFalse(
            any("subject_data" in p or ".env" in p for p in manifest["files"])
        )
        with self.assertRaises(FileExistsError):
            unpack_bundle(self.archive, destination)
        with self.assertRaises(FileExistsError):
            self.pack()

    def test_missing_or_ambiguous_vocabulary(self):
        (self.model / "vocabulary.pt").unlink()
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.pack()
        self.assertFalse(self.archive.exists())

    def test_discovery_refuses_ambiguous_files(self):
        nested = self.model / "another_run"
        nested.mkdir()
        (nested / "vocabulary.pt").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "found 2"):
            discover_artifact(self.model, "vocabulary.pt")

    def test_parent_hydra_config_is_included(self):
        hydra = self.root / "hydra"
        hydra.mkdir()
        (hydra / "config.yaml").write_text("model:\n  attention_dropout: 0.1\n")
        (hydra / "overrides.yaml").write_text("- model.attention_dropout=0.1\n")
        self.pack()
        restored = unpack_bundle(self.archive, self.root / "restored")
        self.assertEqual(
            (restored / "settings/model_hydra/config.yaml").read_text(),
            (hydra / "config.yaml").read_text(),
        )
        self.assertTrue((restored / "settings/model_hydra/overrides.yaml").is_file())

    def test_folder_cli_discovers_fitted_outputs(self):
        data = self.root / "data"
        (data / "logs/hydra").mkdir(parents=True)
        (data / "logs/hydra/config.yaml").write_text("tokenizer: {}\n")
        torch.save({"[PAD]": 0, "[SEP]": 1}, data / "vocabulary.pt")
        meds = self.root / "MEDS"
        (meds / "fit_stage").mkdir(parents=True)
        self.numeric_frame().write_json(meds / "numeric_metadata.json")
        pl.DataFrame(
            {"code": ["DIAGNOSIS//D123"], "adaptive/mapped_code": ["DIAGNOSIS//D12"]}
        ).write_parquet(meds / "fit_stage/adaptive_code_mapping.parquet")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "bonsai.run.model_bundle",
                "pack",
                "--bonsai-dir",
                str(self.root / "bonsai"),
                "--ehr2meds-dir",
                str(self.root / "ehr2meds"),
                "--ehr2meds-output-dir",
                str(meds),
                "--data-dir",
                str(data),
                "--model-dir",
                str(self.model),
                "--output",
                str(self.archive),
            ],
            check=True,
            capture_output=True,
        )
        restored = unpack_bundle(self.archive, self.root / "restored")
        self.assertEqual(
            torch.load(restored / "model/vocabulary.pt", weights_only=True),
            {"[PAD]": 0, "[SEP]": 1},
        )
        manifest = json.loads((restored / "manifest.json").read_text())
        self.assertEqual(
            manifest["artifacts"]["code_mapping"], "preprocessing/code_mapping.parquet"
        )
        self.assertEqual(
            pl.read_json(restored / "preprocessing/numeric_metadata.json")[
                "code"
            ].to_list(),
            ["LABTEST//X"],
        )

    @staticmethod
    def numeric_frame():
        return pl.DataFrame(
            {
                "code": ["LABTEST//X"],
                "numeric/bin_1": [0.0],
                "numeric/bin_99": [1.0],
                "numeric/bin_edges": [[0.5]],
                "numeric/bin_representatives": [[0.25, 0.75]],
                "numeric_values/training_values": [[0.1, 0.7]],
                "subject_id": [123],
            }
        )

    def test_transform_exports_drop_raw_values_identifiers_and_counts(self):
        numeric = self.root / "numeric_metadata.json"
        self.numeric_frame().write_json(numeric)
        mapping = self.root / "mapping.parquet"
        pl.DataFrame(
            {
                "code": ["DIAGNOSIS//D123"],
                "adaptive/mapped_code": ["DIAGNOSIS//D12"],
                "adaptive/count": [20],
                "subject_id": [123],
            }
        ).write_parquet(mapping)
        self.pack(numeric_metadata=numeric, code_mapping=mapping)
        restored = unpack_bundle(self.archive, self.root / "restored")
        numeric_export = pl.read_json(restored / "preprocessing/numeric_metadata.json")
        self.assertEqual(
            set(numeric_export.columns),
            set(self.numeric_frame().columns)
            - {"subject_id", "numeric_values/training_values"},
        )
        self.assertEqual(numeric_export["numeric/bin_edges"].to_list(), [[0.5]])
        self.assertEqual(
            pl.read_parquet(restored / "preprocessing/code_mapping.parquet").columns,
            ["code", "adaptive/mapped_code"],
        )
        self.assertIn(
            "subject_id", pl.read_json(numeric).columns
        )  # Source is untouched.

    def test_patient_data_files_in_source_are_not_collected(self):
        for folder in ("bonsai/bonsai", "ehr2meds/configs", "ehr2meds/resources"):
            directory = self.root / folder
            directory.mkdir(exist_ok=True)
            (directory / "subject_id_mapping.csv").write_text("subject_id\n123\n")
            (directory / "patient_records.parquet").write_bytes(b"not a resource")
        self.pack()
        with zipfile.ZipFile(self.archive) as archive:
            self.assertFalse(
                any(
                    "subject_id_mapping" in name or "patient_records" in name
                    for name in archive.namelist()
                )
            )

    def test_arbitrary_attachments_are_rejected(self):
        patient_file = self.root / "subject_id_mapping.csv"
        patient_file.write_text("subject_id\n123\n")
        with self.assertRaisesRegex(ValueError, "YAML"):
            self.pack(configs=(patient_file,))

    def rewrite(self, change):
        self.pack()
        with zipfile.ZipFile(self.archive) as src:
            contents = {name: src.read(name) for name in src.namelist()}
        change(contents)
        with zipfile.ZipFile(self.archive, "w") as dst:
            for name, data in contents.items():
                dst.writestr(name, data)

    def test_corrupt_payload_leaves_no_destination(self):
        self.rewrite(lambda files: files.update({"model/run/best.ckpt": b"damaged"}))
        with self.assertRaisesRegex(ValueError, "Checksum"):
            unpack_bundle(self.archive, self.root / "restored")
        self.assertFalse((self.root / "restored").exists())

    def test_traversal_rejected(self):
        self.rewrite(lambda files: files.update({"../escape": b"bad"}))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            unpack_bundle(self.archive, self.root / "restored")
        self.assertFalse((self.root / "escape").exists())

    def test_unlisted_file_rejected(self):
        self.rewrite(lambda files: files.update({"unexpected": b"bad"}))
        with self.assertRaisesRegex(ValueError, "manifest"):
            unpack_bundle(self.archive, self.root / "restored")

    def test_symlink_source_rejected(self):
        (self.root / "ehr2meds/ehr2meds/link.py").symlink_to(self.model / "best.ckpt")
        with self.assertRaisesRegex(ValueError, "Symbolic"):
            self.pack()


if __name__ == "__main__":
    unittest.main()
