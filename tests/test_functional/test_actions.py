import unittest
from datetime import datetime

import polars as pl
from omegaconf import OmegaConf

from bonsai.functional.actions import action_router

MATCH_A = {
    "mode": "match",
    "args": {
        "dependence": "independent",
        "conditions": [{"column": "code", "value": ["A"], "operator": "in"}],
    },
}


def criteria(*steps) -> list:
    return OmegaConf.create(list(steps))


def events() -> pl.DataFrame:
    """Subject 1: A, B. Subject 2: A, C. Subject 3: D. One day apart."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3],
            "code": ["A", "B", "A", "C", "D"],
            "time": [datetime(2020, 1, d) for d in range(1, 6)],
        }
    )


class TestActions(unittest.TestCase):
    def test_include_and_exclude(self):
        included = action_router(events(), criteria({"action": "include", **MATCH_A}))
        excluded = action_router(events(), criteria({"action": "exclude", **MATCH_A}))
        self.assertEqual(set(included["subject_id"]), {1, 2})
        self.assertEqual(included.height, 4)  # keeps all rows of included subjects
        self.assertEqual(set(excluded["subject_id"]), {3})

    def test_first_and_rename(self):
        result = action_router(
            events(),
            criteria(
                {"action": "overwrite", "mode": "first", "args": {"col": "time"}},
                {
                    "action": "overwrite",
                    "mode": "rename",
                    "args": {"time": "index_date"},
                },
            ),
        )
        self.assertEqual(result.columns, ["subject_id", "index_date"])
        dates = dict(result.sort("subject_id").iter_rows())
        self.assertEqual(
            dates,
            {1: datetime(2020, 1, 1), 2: datetime(2020, 1, 3), 3: datetime(2020, 1, 5)},
        )

    def test_append_reference(self):
        refs = {
            "index_dates": pl.DataFrame(
                {"subject_id": [1, 2], "index_date": [datetime(2020, 1, 1)] * 2}
            )
        }
        result = action_router(
            events(),
            criteria(
                {
                    "action": "append",
                    "mode": "reference",
                    "args": {"name": "index_dates"},
                }
            ),
            refs,
        )
        self.assertEqual(result.height, 5)
        self.assertTrue(
            result.filter(pl.col("subject_id") == 3)["index_date"].is_null().all()
        )

    def test_append_rejects_several_rows_per_subject(self):
        step = {
            "action": "append",
            "mode": "shift_date",
            "args": {"col": "time", "shift": "1d"},
        }
        with self.assertRaises(pl.exceptions.PolarsError):
            action_router(events(), criteria(step))

    def test_shift_date(self):
        refs = {
            "index_dates": pl.DataFrame(
                {"subject_id": [1], "index_date": [datetime(2020, 1, 1)]}
            )
        }
        result = action_router(
            events(),
            criteria(
                {
                    "action": "overwrite",
                    "mode": "reference",
                    "args": {"name": "index_dates"},
                },
                {
                    "action": "overwrite",
                    "mode": "shift_date",
                    "args": {"col": "index_date", "shift": "5y"},
                },
                {
                    "action": "overwrite",
                    "mode": "rename",
                    "args": {"index_date": "censor_date"},
                },
            ),
            refs,
        )
        self.assertEqual(result["censor_date"][0], datetime(2025, 1, 1))

    def test_unsupported_mode_and_action(self):
        with self.assertRaises(ValueError):
            action_router(
                events(), criteria({"action": "overwrite", "mode": "nope", "args": {}})
            )
        with self.assertRaises(ValueError):
            action_router(events(), criteria({"action": "nope", **MATCH_A}))


# Mini version of the index -> outcome -> cohort steps in create_outcome_cohort
PIPELINE = OmegaConf.create(
    """
index:
  - {action: overwrite, mode: match, args: {dependence: independent, conditions: [{column: code, value: [A], operator: in}]}}
  - {action: overwrite, mode: first, args: {col: time}}
  - {action: overwrite, mode: rename, args: {time: index_date}}
outcome:
  - {action: append, mode: reference, args: {name: index_dates}}
  - action: overwrite
    mode: match
    args:
      dependence: same_row
      conditions:
        - {column: code, value: &outcome [O], operator: in}
        - {column: time, operator: ge, value_col: index_date}
  - {action: overwrite, mode: first, args: {col: time}}
  - {action: overwrite, mode: rename, args: {time: outcome_date}}
cohort:
  - {action: include, mode: reference, args: {name: index_dates}}
  - {action: append, mode: reference, args: {name: index_dates}}
  - action: exclude
    mode: match
    args:
      dependence: same_row
      conditions:
        - {column: code, value: *outcome, operator: in}
        - {column: time, operator: lt, value_col: index_date}
"""
)


class TestPipeline(unittest.TestCase):
    def test_index_outcome_cohort(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3, 4, 4],
                "code": ["A", "O", "A", "O", "O", "A", "O"],
                "time": [
                    datetime(2020, 1, 1),
                    datetime(2021, 1, 1),  # 1: outcome after index
                    datetime(2020, 1, 1),
                    datetime(2019, 1, 1),  # 2: outcome before index
                    datetime(2021, 1, 1),  # 3: no index
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 1),  # 4: outcome on index day
                ],
            }
        )
        refs = {"index_dates": action_router(df, PIPELINE.index)}
        refs["outcome_dates"] = action_router(df, PIPELINE.outcome, refs)
        cohort = action_router(df, PIPELINE.cohort, refs)

        outcomes = dict(refs["outcome_dates"].iter_rows())
        self.assertEqual(outcomes, {1: datetime(2021, 1, 1), 4: datetime(2020, 1, 1)})
        self.assertEqual(set(cohort["subject_id"]), {1, 4})


if __name__ == "__main__":
    unittest.main()
