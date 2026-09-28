from typing import List, Literal
import polars as pl
from bonsai.functional.conditions import get_subject_first_row_for_conditions


def get_cohort_subjects(
    df: pl.DataFrame,
    include_conditions: List,
    include_dependence: Literal["independent", "dependent"],
    exclude_conditions: List | None = None,
    exclude_dependence: Literal["independent", "dependent"] | None = None,
) -> pl.DataFrame:
    """Return subject IDs satisfying the cohort inclusion/exclusion criteria."""

    included = get_subject_first_row_for_conditions(
        df,
        conditions=include_conditions,
        dependence=include_dependence,
    ).select("subject_id")

    if exclude_conditions is not None:
        if exclude_dependence is None:
            raise ValueError(
                "exclude_dependence must be provided when exclude_conditions are provided"
            )

        excluded = get_subject_first_row_for_conditions(
            df,
            conditions=exclude_conditions,
            dependence=exclude_dependence,
        ).select("subject_id")

        included = included.join(
            excluded,
            on="subject_id",
            how="anti",
        )

    return included.unique()