import operator as op_

import polars as pl

COMPARE = {"lt": op_.lt, "le": op_.le, "gt": op_.gt, "ge": op_.ge, "eq": op_.eq}


def get_condition_expression(condition) -> pl.Expr:
    if "column" in condition:
        return get_match_expression(condition)

    # Otherwise a group, e.g. {"any": [condition, ...]}
    if len(condition) != 1:
        raise ValueError(
            f"A group must have exactly one dependence, got {list(condition)}"
        )
    dependence, conditions = list(condition.items())[0]
    if not conditions:
        raise ValueError(f"Dependence={dependence} requires at least one condition")
    exprs = [get_condition_expression(c) for c in conditions]

    if dependence == "any":
        return pl.any_horizontal(exprs)
    if dependence == "all_row":
        return pl.all_horizontal(exprs)
    if dependence == "all":
        # Every condition has been met at or before this row's time, or as a static (null-time) event
        time = pl.col("time")
        return pl.any_horizontal(exprs) & pl.all_horizontal(
            (time >= time.filter(expr).min().over("subject_id"))
            | (expr & time.is_null()).any().over("subject_id")
            for expr in exprs
        )
    raise ValueError(f"Dependence={dependence} not supported")


def get_match_expression(condition) -> pl.Expr:
    op = condition["operator"]
    col = pl.col(condition["column"])
    if op == "in":
        return col.is_in(condition["value"])
    if op == "startswith_any":
        return pl.any_horizontal([col.str.starts_with(v) for v in condition["value"]])
    if op in COMPARE:
        if "value_col" in condition:
            rhs = pl.col(condition["value_col"])
        elif "date_format" in condition:
            rhs = pl.lit(condition["value"]).str.to_date(condition["date_format"])
        else:
            rhs = pl.lit(condition["value"])
        if condition.get("offset_by"):
            rhs = rhs.dt.offset_by(condition["offset_by"])
        return COMPARE[op](col, rhs)
    raise ValueError(f"Operator={op} not supported")


def get_matches(df: pl.DataFrame, args) -> pl.DataFrame:
    """Return all times each subject meets the criteria."""
    occurrence = args["occurrence"]
    expression = get_condition_expression(args["criteria"])
    matches = df.filter(expression).select("subject_id", "time").unique()

    if occurrence == "first":
        matches = matches.group_by("subject_id").agg(pl.col("time").min())
    elif occurrence == "last":
        matches = matches.group_by("subject_id").agg(pl.col("time").max())
    elif occurrence == "all":
        pass
    else:
        raise ValueError(f"occurrence={occurrence} not supported")

    return matches
