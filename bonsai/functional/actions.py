import polars as pl

from bonsai.functional.conditions import get_subject_first_row_for_conditions


def first_match(df, conditional_criteria):
    outputs = []
    for criterion in conditional_criteria:
        outcomes = get_subject_first_row_for_conditions(
            df, criterion.conditions, criterion.dependence
        )
        outputs.append(outcomes)

    return pl.concat(outputs).group_by("subject_id").agg(pl.col("time").min())


def absolute_date(absolute_date: dict):
    return pl.datetime(**absolute_date)


def relative_date(dates: pl.Expr, relative_shift: dict):
    return dates.dt.offset_by(**relative_shift)
