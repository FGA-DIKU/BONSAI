from os.path import exists, join

import pandas as pd
import torch

from corebehrt.constants.data import CONCEPT_COL, PID_COL
from corebehrt.constants.paths import DATA_CFG
from corebehrt.functional.preparation.convert import dataframe_to_patient_list
from corebehrt.main.create_data import main_data
from tests.test_main.base import TestMainScript


class TestCreateData(TestMainScript):
    def setUp(self):
        super().setUp()

        # Paths
        self.features_dir = join(self.tmp_dir, "features")
        self.tokenized_dir = join(self.tmp_dir, "tokenized")
        self.data_path = "./tests/data/raw_with_values"
        self.set_config(
            {
                "paths": {
                    "data": self.data_path,
                    "features": self.features_dir,
                    "tokenized": self.tokenized_dir,
                },
                "features": {
                    "values": {
                        "value_creator_kwargs": {"num_bins": 100},
                    },
                },
                "tokenizer": {"sep_tokens": True, "cls_token": True},
                "excluder": {"min_age": -1, "max_age": 120},
            }
        )

    def test_create_data(self):
        ### Call create data script

        main_data(self.config_path)

        ### Validate generated files.

        # 1: Copy of configuration file should be created in the features and tokenized dirs.
        self.check_config(join(self.features_dir, DATA_CFG))
        self.check_config(join(self.tokenized_dir, DATA_CFG))

        # 2: Check that the features file is created as expected
        self.assertTrue(exists(self.features_dir))
        features_train = pd.read_parquet(join(self.features_dir, "train"))
        features_tuning = pd.read_parquet(join(self.features_dir, "tuning"))
        features_held_out = pd.read_parquet(join(self.features_dir, "held_out"))
        features = pd.concat([features_train, features_tuning, features_held_out])
        self.assertEqual(
            features.columns.to_list(),
            [PID_COL, "age", "abspos", "segment", CONCEPT_COL],
        )

        # 3: Check patients
        patient_info = pd.read_parquet(f"{self.features_dir}/patient_info.parquet")
        self.assertEqual(
            sorted(features[PID_COL].unique().tolist()),
            sorted(patient_info[PID_COL].tolist()),
        )

        # 4: Check vocabulary
        vocab_path = join(self.tokenized_dir, "vocabulary.pt")
        self.assertTrue(exists(vocab_path))
        vocab = torch.load(vocab_path, weights_only=True)
        bg_tokens = [v for k, v in vocab.items() if k.startswith("BG")]
        self.assertEqual(len(bg_tokens), 9)
        val_tokens = [v for k, v in vocab.items() if k.startswith("VAL")]

        # 5. Check tokenisation
        for mode in ["train", "tuning", "held_out"]:
            # Load the parquet file and immediately convert to pandas DataFrame
            tokenised_features = pd.read_parquet(
                join(
                    self.tokenized_dir,
                    f"features_{mode}",
                )
            )

            # # Debug print to verify DataFrame structure
            # print(f"Columns in DataFrame: {tokenised_features.columns}")
            # print(f"First few rows:\n{tokenised_features.head()}")

            # Ensure required columns exist
            required_columns = [PID_COL, CONCEPT_COL, "abspos", "segment", "age"]
            for col in required_columns:
                self.assertIn(
                    col, tokenised_features.columns, f"Missing required column: {col}"
                )

            patient_list = dataframe_to_patient_list(tokenised_features)
            for patient in patient_list:
                concepts = patient.concepts
                self.assertTrue(concepts[0] == vocab["[CLS]"])
                self.assertTrue(concepts[1] == vocab["DOB"])
                self.assertTrue(concepts[2] in bg_tokens)

                index_vals = [i for i, x in enumerate(concepts) if x in val_tokens]
                for i in range(len(index_vals) - 1):
                    self.assertNotEqual(
                        index_vals[i] + 1,
                        index_vals[i + 1],
                        f"Two value tokens are next to each other with indices {index_vals[i]} and {index_vals[i + 1]}. Values {concepts[index_vals[i]]} and {concepts[index_vals[i + 1]]}",
                    )
