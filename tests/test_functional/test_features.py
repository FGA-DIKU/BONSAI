import unittest
import pandas as pd
from bonsai.functional.features import (
    create_features,
    create_background,
    compute_age,
    compute_abspos,
    compute_segments,
    drop_invalids,
    exclude_incorrect_event_ages,
)
import numpy as np


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 2],
                "code": ["DOB", "GENDER", "A", "DOB", "GENDER", "B"],
                "time": [
                    pd.Timestamp("2000-01-01"),
                    None,
                    pd.Timestamp("2000-01-02"),
                    pd.Timestamp("2010-01-01"),
                    None,
                    pd.Timestamp("2010-01-03"),
                ],
            }
        )

    def test_create_features_basic(self):
        features = create_features(self.df.copy())
        self.assertIn("subject_id", features.columns)
        self.assertIn("code", features.columns)
        self.assertIn("age", features.columns)
        self.assertIn("abspos", features.columns)
        self.assertIn("segment", features.columns)
        self.assertEqual(len(features), 6)

    def test_create_background(self):
        df, dob_info = create_background(self.df.copy())
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(isinstance(dob_info, pd.Series))
        self.assertEqual(dob_info.loc[1], pd.Timestamp("2000-01-01"))
        self.assertEqual(dob_info.loc[2], pd.Timestamp("2010-01-01"))
        # Check that background rows are filled in
        bg_rows = df[df["code"].str.startswith("BACKGROUND//", na=False)]
        self.assertEqual(len(bg_rows), 2)
        self.assertEqual(len(bg_rows[bg_rows["time"].isna()]), 0)

    def test_compute_age(self):
        df, dob_info = create_background(self.df.copy())
        ages = compute_age(df, dob_info)
        # The DOB row for each subject should have age 0
        dob_ages = ages[df["code"] == "DOB"]
        self.assertTrue(np.allclose(dob_ages.values, 0.0))
        # The non-DOB rows should have positive age
        non_dob_ages = ages[df["code"] != "DOB"]
        self.assertTrue((non_dob_ages > 0).any())

        bg_ages = ages[df["code"].str.startswith("BACKGROUND//")]
        self.assertTrue((bg_ages == 0).all())

    def test_compute_abspos(self):
        times = pd.Series(
            [
                pd.Timestamp("2000-01-01T00:00:00"),
                pd.Timestamp("2000-01-01T01:00:00"),
                None,
            ]
        )
        abspos = compute_abspos(times)
        self.assertTrue(np.isclose(abspos.iloc[1] - abspos.iloc[0], 1.0, atol=0.01))
        self.assertTrue(pd.isna(abspos.iloc[2]))
        # Test with datetime input
        dt = pd.Timestamp("2000-01-01T00:00:00")
        abspos_single = compute_abspos(dt)
        self.assertTrue(isinstance(abspos_single, float))

    def test_compute_segments(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 2],
                "time": [1, 2, 1, 1, 2],
            }
        )
        segs = compute_segments(df)
        self.assertEqual(list(segs), [1, 2, 1, 1, 2])

    def test_drop_invalids(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, None, 2],
                "code": ["A", "B", None],
                "time": [1, 2, 3],
            }
        )
        cleaned = drop_invalids(df)
        self.assertEqual(len(cleaned), 1)

    def test_exclude_incorrect_event_ages(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "code": ["A", "B", "C"],
                "age": [-2, 10, 130],
                "time": [1, 2, 3],
            }
        )
        filtered = exclude_incorrect_event_ages(df)
        self.assertEqual(len(filtered), 1)
