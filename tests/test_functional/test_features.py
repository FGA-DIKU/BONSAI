import unittest
import polars as pl
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
from datetime import datetime


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 2],
                "code": ["DOB", "GENDER", "A", "DOB", "GENDER", "B"],
                "time": [
                    datetime(2000, 1, 1),
                    None,
                    datetime(2000, 1, 2),
                    datetime(2010, 1, 1),
                    None,
                    datetime(2010, 1, 3),
                ],
            }
        )

    def test_create_features_basic(self):
        features = create_features(self.df)
        self.assertIn("subject_id", features.columns)
        self.assertIn("code", features.columns)
        self.assertIn("age", features.columns)
        self.assertIn("abspos", features.columns)
        self.assertIn("segment", features.columns)
        self.assertEqual(len(features), 6)

    def test_create_background(self):
        df, dob_info = create_background(self.df)
        self.assertTrue(isinstance(df, pl.DataFrame))
        self.assertTrue(isinstance(dob_info, pl.DataFrame))
        dob_dict = dict(zip(dob_info["subject_id"], dob_info["time"]))
        self.assertEqual(dob_dict[1], datetime(2000, 1, 1))
        self.assertEqual(dob_dict[2], datetime(2010, 1, 1))
        # Check that background rows are filled in
        bg_rows = df.filter(pl.col("code").str.starts_with("BACKGROUND//"))
        self.assertEqual(len(bg_rows), 2)
        self.assertEqual(bg_rows.filter(pl.col("time").is_null()).height, 0)

    def test_compute_age(self):
        df, dob_info = create_background(self.df)
        features = df.join(
            dob_info.rename({"time": "dob_time"}), on="subject_id", how="left"
        ).with_columns(
            age=compute_age(
                time=pl.col("time"),
                dob_time=pl.col("dob_time"),
            )
        )

        dob_ages = features.filter(pl.col("code") == "DOB")["age"]
        self.assertTrue(np.allclose(dob_ages.to_numpy(), 0.0))

        non_dob_ages = features.filter(pl.col("code") != "DOB")["age"]
        self.assertTrue((non_dob_ages.to_numpy() > 0).any())

        bg_ages = features.filter(pl.col("code").str.starts_with("BACKGROUND//"))["age"]
        self.assertTrue((bg_ages.to_numpy() == 0).all())

    def test_compute_abspos(self):
        times = pl.Series(
            [
                datetime(2000, 1, 1, 0, 0, 0),
                datetime(2000, 1, 1, 1, 0, 0),
                None,
            ]
        )
        abspos = compute_abspos(times)
        self.assertTrue(np.isclose(abspos[1] - abspos[0], 1e-3, atol=1e-5))
        self.assertTrue(abspos[2] is None)

        dt = datetime(2000, 1, 1, 0, 0, 0)
        abspos_single = compute_abspos(dt)
        self.assertTrue(isinstance(abspos_single, float))

    def test_compute_segments(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 2],
                "time": [1, 2, 1, 1, 2],
            }
        )
        segs = df.with_columns(
            segment=compute_segments(
                time=pl.col("time"),
                subject_id=pl.col("subject_id"),
            )
        )["segment"]
        self.assertEqual(segs.to_list(), [1, 2, 1, 1, 2])

    def test_drop_invalids(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, None, 2],
                "code": ["A", "B", None],
                "time": [1, 2, 3],
            }
        )
        cleaned = drop_invalids(df)
        self.assertEqual(len(cleaned), 1)

    def test_exclude_incorrect_event_ages(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "code": ["A", "B", "C"],
                "age": [-2, 10, 130],
                "time": [1, 2, 3],
            }
        )
        filtered = exclude_incorrect_event_ages(df)
        self.assertEqual(len(filtered), 1)
