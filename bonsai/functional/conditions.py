import operator as op_
from typing import Literal

import polars as pl

COMPARE = {"lt": op_.lt, "le": op_.le, "gt": op_.gt, "ge": op_.ge, "eq": op_.eq}


def get_condition_expression(cond) -> pl.Expr:
    op = cond["operator"]
    col = pl.col(cond["column"])
    if op == "in":
        return col.is_in(cond["value"])
    if op == "startswith_any":
        return pl.any_horizontal([col.str.starts_with(v) for v in cond["value"]])
    if op in COMPARE:
        rhs = (
            pl.col(cond["value_col"]) if "value_col" in cond else pl.lit(cond["value"])
        )
        if cond.get("offset"):
            rhs = rhs.dt.offset_by(cond["offset"])
        return COMPARE[op](col, rhs)
    raise ValueError(f"Unsupported operator: {op}")


def match(
    df: pl.DataFrame,
    conditions: list,
    dependence: Literal["independent", "dependent", "same_row"],
) -> pl.DataFrame:
    """Earliest time each subject meets the definition.

    independent: any condition (earliest time any is met)
    dependent:   all conditions, possibly on different rows (time the last one is met)
    same_row:    all conditions on the same row (earliest such row)
    """
    if dependence not in ("independent", "dependent", "same_row"):
        raise ValueError(
            f"Dependence can only be [independent, dependent, same_row], not {dependence}"
        )

    if dependence == "same_row":
        mask = pl.all_horizontal([get_condition_expression(c) for c in conditions])
        return df.filter(mask).group_by("subject_id").agg(pl.col("time").min())

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

    return res.select("subject_id", combine(cols).alias("time"))


def add_column(
    df: pl.DataFrame, name: str, value, dtype: str | None = None
) -> pl.DataFrame:
    """Add a constant column. dtype='datetime' parses an ISO date string."""
    expr = pl.lit(value)
    if dtype == "datetime":
        expr = expr.str.to_datetime()
    elif dtype:
        expr = expr.cast(getattr(pl, dtype))
    return df.with_columns(expr.alias(name))
