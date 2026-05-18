import unittest
import polars as pl
from bonsai.functional.outcomes import (
    get_subject_first_row_for_conditions,
    get_date_from_absolute_date,
    get_date_from_relative_date,
    get_date_from_exposure_date,
    fill_nans_with_sampled,
    binarize_outcomes,
    split_and_binarize_outcomes,
    expand_subjects_for_outcomes,
    outcomes_to_frame,
)
from datetime import datetime


class TestCreateOutcomesUtils(unittest.TestCase):
    def test_find_independent(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2, 2],
                "code": ["A", "B", "A", "C"],
                "time": [datetime(2020, 1, i + 1) for i in range(4)],
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
        self.assertIn("A", result["code"])

    def test_find_dependent(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3],
                "code": ["A", "B", "A", "C", "A"],
                "time": [datetime(2020, 1, i + 1) for i in range(5)],
            }
        )
        conditions = [
            {"col": "code", "vals": ["A"]},
            {"col": "code", "vals": ["C"]},
        ]
        result = get_subject_first_row_for_conditions(
            df, conditions, dependence="dependent"
        )
        self.assertEqual(set(result["subject_id"]), {2})
        self.assertIn("A", result["code"])

    def test_find_dependent2(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3],
                "code": ["A", "B", "A", "C", "A"],
                "time": [datetime(2020, 1, i + 1) for i in range(5)],
            }
        )
        conditions = [
            {"col": "code", "vals": ["C"]},
            {"col": "code", "vals": ["A"]},
        ]
        result = get_subject_first_row_for_conditions(
            df, conditions, dependence="dependent"
        )
        self.assertEqual(set(result["subject_id"]), {2})
        self.assertIn("C", result["code"])

    def test_find_invalid_dependence(self):
        df = pl.DataFrame(
            {
                "subject_id": [1],
                "code": ["A"],
                "time": [datetime(2020, 1, 1)],
            }
        )
        conditions = [{"col": "code", "vals": ["A"]}]
        with self.assertRaises(ValueError):
            get_subject_first_row_for_conditions(df, conditions, dependence="invalid")

    def test_get_date_from_absolute_date(self):
        date_dict = {"year": 2020, "month": 1, "day": 2}
        result = get_date_from_absolute_date(absolute_date=date_dict)
        self.assertEqual(result, datetime(2020, 1, 2))

    def test_get_date_from_relative_date(self):
        base_dates = pl.Series([datetime(2020, 1, 1), datetime(2020, 1, 2), None])
        result = get_date_from_relative_date(
            relative_dates=base_dates, relative_hour_shift=24
        )
        self.assertEqual(result[0], datetime(2020, 1, 2))
        self.assertEqual(result[1], datetime(2020, 1, 3))
        self.assertIsNone(result[2])

    def test_get_date_from_exposure_date(self):
        df = pl.DataFrame(
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
        outcomes = get_subject_first_row_for_conditions(
            df, conditions, dependence="independent"
        )
        outcomes = (
            df.select("subject_id")
            .unique()
            .join(outcomes, on="subject_id", how="left")
            .drop("code")
            .rename({"time": "outcome_date"})
        )

        result = get_date_from_exposure_date(
            subjects=outcomes.select("subject_id"),
            df=df,
            conditions=conditions,
            dependence="independent",
        )
        result_by_subject = dict(zip(outcomes["subject_id"], result))

        self.assertEqual(result_by_subject[1], datetime(2020, 1, 1))
        self.assertEqual(result_by_subject[2], datetime(2022, 1, 1))
        self.assertIsNone(result_by_subject[3])

    def test_fill_nans_with_sampled(self):
        dates = pl.Series([datetime(2020, 1, 1), datetime(2021, 1, 1), None])
        res = fill_nans_with_sampled(dates)
        self.assertFalse(res.is_null().any())

    def test_fill_nans_with_sampled_false(self):
        dates = pl.Series([None, None, None])
        with self.assertRaises(ValueError):
            fill_nans_with_sampled(dates)


class TestBinizationOutcomes(unittest.TestCase):
    def test_binarize_outcomes_basic(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "index_date": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "outcome_date": [datetime(2020, 1, 3), datetime(2020, 1, 1)],
                "censor_abspos": [10, 20],
            }
        )
        result = binarize_outcomes(df, n_hours_start_include=24)
        by_subject = {row["subject_id"]: row for row in result}
        self.assertEqual(by_subject[1]["label"], 1)
        self.assertEqual(by_subject[2]["label"], 0)

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
                "censor_abspos": [10, 20, 30],
            }
        )
        result = binarize_outcomes(df, n_hours_start_include=24, n_hours_end_include=72)
        by_subject = {row["subject_id"]: row for row in result}
        self.assertEqual(by_subject[1]["label"], 1)
        self.assertEqual(by_subject[2]["label"], 1)
        self.assertEqual(by_subject[3]["label"], 0)

    def test_binarize_outcomes_duplicate_subjects(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "index_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "outcome_date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                "censor_abspos": [10, 20],
            }
        )
        result = binarize_outcomes(df, n_hours_start_include=24)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["label"], 0)
        self.assertEqual(result[1]["label"], 1)
        frame = outcomes_to_frame(result)
        self.assertListEqual(frame["subject_id"].tolist(), [1, 1])

    def test_binarize_outcomes_empty(self):
        df = pl.DataFrame(
            schema={
                "subject_id": pl.Int64,
                "index_date": pl.Datetime,
                "outcome_date": pl.Datetime,
                "censor_abspos": pl.Int64,
            }
        )
        result = binarize_outcomes(df, n_hours_start_include=24)
        self.assertEqual(result, [])

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
                    datetime(2020, 1, 1),
                ],
                "censor_abspos": [10, 20, 30, 40, 50, 60],
                "split": ["train", "train", "val", "val", "test", "test"],
            }
        )
        train, val, test = split_and_binarize_outcomes(
            df, "train", "val", "test", n_hours_start_include=24
        )
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(test), 2)
        train_by_subject = {row["subject_id"]: row for row in train}
        val_by_subject = {row["subject_id"]: row for row in val}
        test_by_subject = {row["subject_id"]: row for row in test}
        self.assertEqual(train_by_subject[1]["label"], 1)
        self.assertEqual(train_by_subject[2]["label"], 0)
        self.assertEqual(val_by_subject[3]["label"], 1)
        self.assertEqual(val_by_subject[4]["label"], 0)
        self.assertEqual(test_by_subject[5]["label"], 1)
        self.assertEqual(test_by_subject[6]["label"], 0)

    def test_expand_subjects_for_outcomes_duplicates(self):
        subjects = [{"subject_id": 1, "code": [1]}, {"subject_id": 2, "code": [2]}]
        outcomes = [
            {"subject_id": 1, "label": 0, "censor_abspos": 10},
            {"subject_id": 1, "label": 1, "censor_abspos": 20},
        ]
        expanded_subjects, expanded_outcomes = expand_subjects_for_outcomes(
            subjects, outcomes
        )
        self.assertEqual(len(expanded_subjects), 2)
        self.assertEqual(expanded_outcomes[0]["censor_abspos"], 10)
        self.assertEqual(expanded_outcomes[1]["censor_abspos"], 20)
