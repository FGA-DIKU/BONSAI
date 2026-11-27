import unittest
from datetime import datetime

import polars as pl

from bonsai.functional.matching import get_matches


class TestMatching(unittest.TestCase):
    def test_nested_criteria(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 3, 3],
                "code": [
                    "SEX//MAND",
                    "LABTEST//A",
                    "DIAGNOSIS//SYN_OTHER",
                    "DIAGNOSIS//SYN_OTHER",
                    "SEX//MAND",
                    "LABTEST//A",
                ],
                "time": [
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 4),
                    datetime(2020, 1, 2),
                    datetime(2019, 12, 31),
                    datetime(2020, 1, 3),
                ],
                "index_date": [datetime(2020, 1, 1)] * 6,
            }
        )
        args = {
            "occurrence": "first",
            "criteria": {
                "any": [
                    {
                        "all": [
                            {
                                "all_row": [
                                    {
                                        "column": "code",
                                        "operator": "startswith_any",
                                        "value": ["SEX//MAND"],
                                    },
                                    {
                                        "column": "time",
                                        "operator": "gt",
                                        "value_col": "index_date",
                                    },
                                ]
                            },
                            {
                                "column": "code",
                                "operator": "startswith_any",
                                "value": ["LABTEST"],
                            },
                        ]
                    },
                    {
                        "all_row": [
                            {
                                "column": "code",
                                "operator": "startswith_any",
                                "value": ["DIAGNOSIS//SYN_OTHER"],
                            },
                            {
                                "column": "time",
                                "operator": "gt",
                                "value_col": "index_date",
                            },
                        ]
                    },
                ]
            },
        }

        matches = get_matches(df, args).sort("subject_id")

        self.assertEqual(matches["subject_id"].to_list(), [1, 2])
        self.assertEqual(
            matches["time"].to_list(),
            [datetime(2020, 1, 3), datetime(2020, 1, 2)],
        )

    def test_any_and_first_occurrence(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "code": ["A", "B", "C"],
                "time": [1, 2, 1],
            }
        )

        matches = get_matches(
            df,
            {
                "occurrence": "first",
                "criteria": {
                    "any": [{"column": "code", "operator": "in", "value": ["B", "C"]}]
                },
            },
        ).sort("subject_id")

        self.assertEqual(
            matches.to_dict(as_series=False),
            {
                "subject_id": [1, 2],
                "time": [2, 1],
            },
        )

    def test_invalid_group_raises(self):
        df = pl.DataFrame({"subject_id": [1], "code": ["A"], "time": [1]})
        leaf = {"column": "code", "operator": "in", "value": ["A"]}
        for criteria in ({"any": [leaf], "all": [leaf]}, {"none": [leaf]}, {"any": []}):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                get_matches(df, {"occurrence": "all", "criteria": criteria})

    def test_all_with_static_null_time_condition(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 3, 4, 4],
                "code": [
                    "SEX//MAND",
                    "LABTEST//A",
                    "LABTEST//A",
                    "SEX//KVINDE",
                    "LABTEST//A",
                    "SEX//MAND",
                    "SEX//MAND",
                    "ETH//X",
                ],
                "time": [
                    None,
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 10),
                    None,
                    datetime(2020, 1, 5),
                    None,
                    None,
                    None,
                ],
            }
        )
        criteria = {
            "all": [
                {"column": "code", "operator": "in", "value": ["SEX//MAND"]},
                {"column": "code", "operator": "startswith_any", "value": ["LABTEST"]},
            ]
        }

        first_matches = get_matches(df, {"occurrence": "first", "criteria": criteria})
        self.assertEqual(
            first_matches.to_dict(as_series=False),
            {"subject_id": [1], "time": [datetime(2020, 1, 5)]},
        )

        last_matches = get_matches(df, {"occurrence": "last", "criteria": criteria})
        self.assertEqual(
            last_matches.to_dict(as_series=False),
            {"subject_id": [1], "time": [datetime(2020, 1, 10)]},
        )

        both_static = get_matches(
            df,
            {
                "occurrence": "first",
                "criteria": {
                    "all": [
                        {"column": "code", "operator": "in", "value": ["SEX//MAND"]},
                        {"column": "code", "operator": "in", "value": ["ETH//X"]},
                    ]
                },
            },
        )
        self.assertEqual(
            both_static.to_dict(as_series=False),
            {"subject_id": [4], "time": [None]},
        )


if __name__ == "__main__":
    unittest.main()
