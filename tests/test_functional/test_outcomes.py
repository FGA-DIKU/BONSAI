import unittest
import pandas as pd
from bonsai.functional.outcomes import (
    get_subject_first_row_for_conditions,
    get_index_date_from_absolute_date,
    get_index_date_from_relative_date,
    get_index_date_from_exposure_date,
    binarize_outcomes,
    split_and_binarize_outcomes,
)
from datetime import datetime


class TestCreateOutcomesUtils(unittest.TestCase):
    def test_find_independent(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2],
                "code": ["A", "B", "A", "C"],
            }
        )
        conditions = [
            {"col": "code", "vals": ["A"]},
            {"col": "code", "vals": ["C"]},
        ]
        result = get_subject_first_row_for_conditions(
            df, conditions, dependence="independent"
        )
        self.assertEqual(set(result["subject_id"]), {1, 2})
        # Should return first row for each subject that matches any condition
        self.assertIn("A", result["code"].tolist())

    def test_find_dependent(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3],
                "code": ["A", "B", "A", "C", "A"],
            }
        )
        conditions = [
            {"col": "code", "vals": ["A"]},
            {"col": "code", "vals": ["C"]},
        ]
        # Only subject 2 has both A and C
        result = get_subject_first_row_for_conditions(
            df, conditions, dependence="dependent"
        )
        self.assertEqual(set(result["subject_id"]), {2})
        self.assertIn("A", result["code"].tolist())

    def test_find_dependent2(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3],
                "code": ["A", "B", "A", "C", "A"],
            }
        )
        conditions = [
            {"col": "code", "vals": ["C"]},
            {"col": "code", "vals": ["A"]},
        ]
        # Only subject 2 has both A and C
        result = get_subject_first_row_for_conditions(
            df, conditions, dependence="dependent"
        )
        self.assertEqual(set(result["subject_id"]), {2})
        self.assertIn("C", result["code"].tolist())  # NEW PRIORITY!

    def test_find_invalid_dependence(self):
        df = pd.DataFrame({"subject_id": [1], "code": ["A"]})
        conditions = [{"col": "code", "vals": ["A"]}]
        with self.assertRaises(ValueError):
            get_subject_first_row_for_conditions(df, conditions, dependence="invalid")

    def test_get_index_date_from_absolute_date(self):
        date_dict = {"year": 2020, "month": 1, "day": 2}
        result = get_index_date_from_absolute_date(absolute_date=date_dict)
        self.assertEqual(result, datetime(2020, 1, 2))

    def test_get_index_date_from_relative_date(self):
        base_dates = pd.Series([datetime(2020, 1, 1), datetime(2020, 1, 2), None])
        result = get_index_date_from_relative_date(
            relative_dates=base_dates, relative_hour_shift=24
        )
        self.assertTrue(
            (
                result.iloc[:2]
                == pd.Series([datetime(2020, 1, 2), datetime(2020, 1, 3)])
            ).all()
        )
        self.assertTrue(pd.isna(result.iloc[2]))

    def test_get_index_date_from_exposure_date(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3],
                "code": ["A", "B", "A", "C", "D"],
                "time": [datetime(2020 + i, 1, 1) for i in range(5)],
            }
        )
        conditions = [
            {"col": "code", "vals": ["A"]},
            {"col": "code", "vals": ["C"]},
        ]
        # Only subject 1 and 2 has A or C
        outcomes = get_subject_first_row_for_conditions(
            df, conditions, dependence="independent"
        )
        outcomes = (
            df[["subject_id"]]
            .drop_duplicates()
            .merge(outcomes, on="subject_id", how="left")
        )
        outcomes = outcomes.drop(columns="code").rename(
            columns={"time": "outcome_date"}
        )

        result = get_index_date_from_exposure_date(
            subjects=outcomes[["subject_id"]],
            df=df,
            exposure_conditions=conditions,
            exposure_dependence="independent",
        )
        self.assertEqual(result.iloc[0], datetime(2020, 1, 1))
        self.assertEqual(result.iloc[1], datetime(2022, 1, 1))
        self.assertTrue(pd.isna(result.iloc[2]))

    def test_get_index_date_from_absolute_date_invalid(self):
        with self.assertRaises(ValueError):
            get_index_date_from_absolute_date("foo")


class TestBinizationOutcomes(unittest.TestCase):
    def test_binarize_outcomes_basic(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 2],
                "index_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "outcome_date": pd.to_datetime(["2020-01-03", "2020-01-01"]),
                "censor_abspos": [10, 20],
            }
        )
        result = binarize_outcomes(df, n_hours_start_include=24)
        self.assertEqual(result[1]["label"], 1)
        self.assertEqual(result[2]["label"], 0)

    def test_binarize_outcomes_with_end(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "index_date": pd.to_datetime(["2020-01-01"] * 3),
                "outcome_date": pd.to_datetime(
                    ["2020-01-03", "2020-01-02", "2020-01-05"]
                ),
                "censor_abspos": [10, 20, 30],
            }
        )
        # Only subject 2 is between 24 and 72 hours
        result = binarize_outcomes(df, n_hours_start_include=24, n_hours_end_include=72)
        self.assertEqual(result[1]["label"], 1)
        self.assertEqual(result[2]["label"], 1)
        self.assertEqual(result[3]["label"], 0)

    def test_binarize_outcomes_empty(self):
        df = pd.DataFrame(
            columns=["subject_id", "index_date", "outcome_date", "censor_abspos"]
        )
        df["index_date"] = pd.to_datetime(df["index_date"])
        df["outcome_date"] = pd.to_datetime(df["outcome_date"])
        result = binarize_outcomes(df, n_hours_start_include=24)
        self.assertEqual(result, {})

    def test_split_and_binarize_outcomes(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 2, 3, 4, 5, 6],
                "index_date": pd.to_datetime(["2020-01-01"] * 6),
                "outcome_date": pd.to_datetime(
                    [
                        "2020-01-03",
                        "2020-01-01",
                        "2020-01-04",
                        "2020-01-01",
                        "2020-01-05",
                        "2020-01-01",
                    ]
                ),
                "censor_abspos": [10, 20, 30, 40, 50, 60],
                "split": ["train", "train", "val", "val", "test", "test"],
            }
        )
        train, val, test = split_and_binarize_outcomes(
            df, "train", "val", "test", n_hours_start_include=24
        )
        self.assertEqual(set(train.keys()), {1, 2})
        self.assertEqual(set(val.keys()), {3, 4})
        self.assertEqual(set(test.keys()), {5, 6})
        self.assertEqual(train[1]["label"], 1)
        self.assertEqual(train[2]["label"], 0)
        self.assertEqual(val[3]["label"], 1)
        self.assertEqual(val[4]["label"], 0)
        self.assertEqual(test[5]["label"], 1)
        self.assertEqual(test[6]["label"], 0)
