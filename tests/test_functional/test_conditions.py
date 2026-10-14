import unittest
from datetime import datetime

import polars as pl

from bonsai.functional.conditions import add_column, get_condition_expression, match

CODE_A_OR_C = [
    {"column": "code", "value": ["A"], "operator": "in"},
    {"column": "code", "value": ["C"], "operator": "in"},
]


def events() -> pl.DataFrame:
    """Subject 1: A, B. Subject 2: A, C. Subject 3: A. One day apart."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3],
            "code": ["A", "B", "A", "C", "A"],
            "time": [datetime(2020, 1, i + 1) for i in range(5)],
        }
    )


class TestConditionExpression(unittest.TestCase):
    def test_in_and_startswith_any(self):
        df = pl.DataFrame(
            {"code": ["DIAGNOSIS//DE11", "DIAGNOSIS//DE10", "MEDICATION//A10"]}
        )
        is_in = {"column": "code", "value": ["DIAGNOSIS//DE10"], "operator": "in"}
        starts = {
            "column": "code",
            "value": ["DIAGNOSIS//DE1", "MEDICATION"],
            "operator": "startswith_any",
        }
        self.assertEqual(df.filter(get_condition_expression(is_in)).height, 1)
        self.assertEqual(df.filter(get_condition_expression(starts)).height, 3)

    def test_compare_to_column_with_offset(self):
        df = pl.DataFrame(
            {
                "time": [datetime(2019, 1, 1), datetime(2021, 1, 1)],
                "index_date": [datetime(2020, 1, 1)] * 2,
            }
        )
        before = {"column": "time", "operator": "lt", "value_col": "index_date"}
        lookback = {
            "column": "time",
            "operator": "ge",
            "value_col": "index_date",
            "offset": "-6mo",
        }
        self.assertEqual(
            df.filter(get_condition_expression(before))["time"].to_list(),
            [datetime(2019, 1, 1)],
        )
        self.assertEqual(
            df.filter(get_condition_expression(lookback))["time"].to_list(),
            [datetime(2021, 1, 1)],
        )

    def test_unsupported_operator(self):
        with self.assertRaises(ValueError):
            get_condition_expression(
                {"column": "code", "value": ["A"], "operator": "contains"}
            )


class TestMatch(unittest.TestCase):
    def test_independent_is_any_condition(self):
        result = match(events(), CODE_A_OR_C, dependence="independent")
        times = dict(result.select("subject_id", "time").iter_rows())
        self.assertEqual(set(times), {1, 2, 3})
        self.assertEqual(times[2], datetime(2020, 1, 3))  # earliest condition met

    def test_dependent_is_all_conditions_any_rows(self):
        result = match(events(), CODE_A_OR_C, dependence="dependent")
        self.assertEqual(result["subject_id"].to_list(), [2])
        self.assertEqual(
            result["time"][0], datetime(2020, 1, 4)
        )  # when the last condition is met

    def test_same_row_differs_from_dependent(self):
        # Subject 1: X before index, another event after. Subject 2: X after index.
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "code": ["X", "Y", "X"],
                "time": [
                    datetime(2019, 1, 1),
                    datetime(2021, 1, 1),
                    datetime(2021, 1, 1),
                ],
                "index_date": [datetime(2020, 1, 1)] * 3,
            }
        )
        conditions = [
            {"column": "code", "value": ["X"], "operator": "in"},
            {"column": "time", "operator": "ge", "value_col": "index_date"},
        ]
        same_row = match(df, conditions, dependence="same_row")
        dependent = match(df, conditions, dependence="dependent")
        self.assertEqual(same_row["subject_id"].to_list(), [2])
        self.assertEqual(set(dependent["subject_id"]), {1, 2})

    def test_invalid_dependence(self):
        with self.assertRaises(ValueError):
            match(events(), CODE_A_OR_C, dependence="invalid")


class TestAddColumn(unittest.TestCase):
    def test_datetime_column(self):
        result = add_column(
            events(), name="index_date", value="2020-01-02", dtype="datetime"
        )
        self.assertEqual(
            result["index_date"].unique().to_list(), [datetime(2020, 1, 2)]
        )


if __name__ == "__main__":
    unittest.main()
