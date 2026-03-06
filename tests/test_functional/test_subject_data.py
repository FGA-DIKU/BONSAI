import unittest
from unittest.mock import patch, MagicMock
from bonsai.functional.subject_data import filter_subject_data, prepare_subject_data
from pathlib import Path
import torch
import pandas as pd


class TestSubjectData(unittest.TestCase):
    def test_filter_subject_data(self):
        subject_data = [
            {"subject_id": 1, "code": [1]},
            {"subject_id": 2, "code": [2]},
            {"subject_id": 3, "code": [3]},
        ]
        cohort = [1, 3]
        filtered = filter_subject_data(subject_data, cohort)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["subject_id"], 1)
        self.assertEqual(filtered[1]["subject_id"], 3)

    @patch("bonsai.functional.subject_data.pd.read_parquet")
    def test_prepare_subject_data(self, mock_read_parquet):
        # Mock the dataframe returned by read_parquet
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "code": [10, 11, 20],
                "abspos": [0, 1, 0],
                "segment": [0, 0, 1],
                "age": [30, 31, 40],
            }
        )
        mock_read_parquet.return_value = df

        # Mock Path.glob to return a list with a single 'file'
        mock_path = MagicMock(spec=Path)
        mock_path.glob.return_value = ["fake.parquet"]

        result = prepare_subject_data(mock_path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["subject_id"], 1)
        self.assertTrue(torch.equal(result[0]["code"], torch.tensor([10, 11])))
        self.assertEqual(result[1]["subject_id"], 2)
        self.assertTrue(torch.equal(result[1]["code"], torch.tensor([20])))
