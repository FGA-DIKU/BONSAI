import unittest
from datetime import datetime

import polars as pl
from omegaconf import OmegaConf

from bonsai.functional.outcomes import (
    binarize_outcomes,
    fill_nans_with_sampled,
    get_index_dates,
    split_and_binarize_outcomes,
)


class TestIndexDates(unittest.TestCase):
    def setUp(self):
        # Subject 1 has an index event (A), subject 2 does not
        self.df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "code": ["A", "B", "B"],
                "time": [
                    datetime(2020, 1, 1),
                    datetime(2021, 1, 1),
                    datetime(2020, 6, 1),
                ],
            }
        )
        self.criteria = [
            {
                "action": "overwrite",
                "mode": "match",
                "args": {
                    "dependence": "independent",
                    "conditions": [
                        {"column": "code", "value": ["A"], "operator": "in"}
                    ],
                },
            },
            {"action": "overwrite", "mode": "first", "args": {"col": "time"}},
            {"action": "overwrite", "mode": "rename", "args": {"time": "index_date"}},
        ]

    def cfg(self, missing):
        return OmegaConf.create(
            {"missing": missing, "conditional_criteria": self.criteria}
        )

    def test_drop(self):
        result = get_index_dates(self.df, self.cfg("drop"), seed=0)
        self.assertEqual(result["subject_id"].to_list(), [1])

    def test_sample(self):
        result = get_index_dates(self.df, self.cfg("sample"), seed=0).sort("subject_id")
        self.assertEqual(result["subject_id"].to_list(), [1, 2])
        self.assertEqual(result["index_date"].to_list(), [datetime(2020, 1, 1)] * 2)


class TestFillNansWithSampled(unittest.TestCase):
    def test_fills_only_nulls_from_observed(self):
        observed = [datetime(2020, 1, 1), datetime(2021, 1, 1)]
        res = fill_nans_with_sampled(pl.Series(observed + [None, None]), seed=0)
        self.assertFalse(res.is_null().any())
        self.assertEqual(res[:2].to_list(), observed)
        self.assertTrue(set(res[2:]) <= set(observed))

    def test_reproducible_with_seed(self):
        dates = pl.Series([datetime(2020 + i, 1, 1) for i in range(10)] + [None] * 20)
        self.assertTrue(
            fill_nans_with_sampled(dates, seed=1).equals(
                fill_nans_with_sampled(dates, seed=1)
            )
        )

    def test_all_null_raises(self):
        with self.assertRaises(ValueError):
            fill_nans_with_sampled(pl.Series([None, None, None]))


class TestBinizationOutcomes(unittest.TestCase):
    def test_binarize_outcomes_basic(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "index_date": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "outcome_date": [datetime(2020, 1, 3), None],
                "censor_date": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
            }
        )
        result = binarize_outcomes(df, start_include={"hours": 24})
        self.assertEqual(result["label"][0], 1)
        self.assertEqual(result["label"][1], 0)

    def test_binarize_outcomes_with_end(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "index_date": [datetime(2020, 1, 1)] * 3,
                "outcome_date": [
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 5),
                ],
                "censor_date": [datetime(2020, 1, 1)] * 3,
            }
        )
        result = binarize_outcomes(
            df, start_include={"hours": 24}, end_include={"hours": 72}
        )
        self.assertEqual(result["label"][0], 1)
        self.assertEqual(result["label"][1], 1)
        self.assertEqual(result["label"][2], 0)

    def test_binarize_outcomes_empty(self):
        df = pl.DataFrame(
            schema={
                "subject_id": pl.Int64,
                "index_date": pl.Datetime,
                "outcome_date": pl.Datetime,
                "censor_date": pl.Datetime,
            }
        )
        result = binarize_outcomes(df, start_include={"hours": 24})
        self.assertTrue(result.is_empty())

    def test_split_and_binarize_outcomes(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 2, 3, 4, 5, 6],
                "index_date": [datetime(2020, 1, 1)] * 6,
                "outcome_date": [
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 4),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 5),
                    None,
                ],
                "censor_date": [datetime(2020, 1, 1)] * 6,
                "split": ["train", "train", "val", "val", "test", "test"],
            }
        )
        train, val, test = split_and_binarize_outcomes(
            df, "train", "val", "test", start_include={"hours": 24}
        )
        self.assertEqual(set(train.keys()), {1})
        self.assertEqual(set(val.keys()), {3})
        self.assertEqual(set(test.keys()), {5, 6})
        self.assertEqual(train[1]["label"], 1)
        self.assertEqual(val[3]["label"], 1)
        self.assertEqual(test[5]["label"], 1)
        self.assertEqual(test[6]["label"], 0)


if __name__ == "__main__":
    unittest.main()
