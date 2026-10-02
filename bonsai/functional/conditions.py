from typing import Literal

import polars as pl


def get_condition_expression(cond) -> pl.Expr:
    """Build a Polars expression for a condition."""
    match = cond.get("match", "exact")

    if match == "exact":
        return pl.col(cond["col"]).is_in(cond["vals"])

    if match == "prefix":
        return pl.any_horizontal(
            [pl.col(cond["col"]).str.starts_with(val) for val in cond["vals"]]
        )

    raise ValueError(f"Match can only be [exact, prefix], not {match}")


def get_subject_first_row_for_conditions(
    df: pl.DataFrame, conditions: list, dependence: Literal["independent", "dependent"]
) -> pl.DataFrame:
    """Earliest time each subject meets the definition: any condition (independent) or all (dependent)."""
    if dependence not in ("independent", "dependent"):
        raise ValueError(
            f"Dependence can only be [independent, dependent], not {dependence}"
        )

    # Build conditions
    per_cond = [
        df.filter(get_condition_expression(cond))
        .group_by("subject_id")
        .agg(pl.col("time").min().alias(f"_time{i}"))
        for i, cond in enumerate(conditions)
    ]

    # Joins conditions
    how = "full" if dependence == "independent" else "inner"
    res = per_cond[0]
    for other in per_cond[1:]:
        res = res.join(other, on="subject_id", how=how, coalesce=True)

    # Find dependence time
    cols = [f"_time{i}" for i in range(len(conditions))]
    combine = pl.min_horizontal if dependence == "independent" else pl.max_horizontal

    if len(res) == 0:
        raise ValueError(
            f"No subjects meet the conditions {conditions} with dependence={dependence}"
        )

    return res.select("subject_id", combine(cols).alias("time"))
