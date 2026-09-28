from typing import List, Literal
import polars as pl


def get_condition_expression(cond) -> pl.Expr:
    """Build a Polars expression for a condition."""
    match = cond.get("match", "exact")

    if match == "exact":
        return pl.col(cond["col"]).is_in(cond["vals"])

    if match == "prefix":
        return pl.any_horizontal(
            [
                pl.col(cond["col"]).str.starts_with(val)
                for val in cond["vals"]
            ]
        )

    raise ValueError(
        f"Match can only be [exact, prefix], not {match}"
    )


def get_subject_first_row_for_conditions(
    df: pl.DataFrame,
    conditions: List,
    dependence: Literal["independent", "dependent"],
) -> pl.DataFrame:
    """Returns the first row (priority based on condition order) for each subject that matches the conditions."""
    # Initialization
    df = df.with_columns(_prio=pl.lit(None).cast(pl.Int32))
    row_mask = pl.lit(False)
    subject_sets = []

    # Find matches (dataframe rows AND subject_ids) of conditions
    for i, cond in enumerate(conditions):
        cond_expr = get_condition_expression(cond)

        # OR operation
        row_mask = row_mask | cond_expr

        # Set priority (to take first row later)
        df = df.with_columns(
            _prio=pl.when(
                cond_expr & pl.col("_prio").is_null()
            )
            .then(pl.lit(i))
            .otherwise(pl.col("_prio"))
        )

        # Get subjects that match condition
        subject_sets.append(
            set(
                df.filter(cond_expr)
                .get_column("subject_id")
                .to_list()
            )
        )

    # Toggle between any or all conditions met
    if dependence == "independent":
        matched_subjects = set.union(*subject_sets)  # Any condition met

    elif dependence == "dependent":  # TODO: Implement time_window
        matched_subjects = set.intersection(*subject_sets)  # All conditions met

    else:
        raise ValueError(
            f"Dependence can only be [independent, dependent], not {dependence}"
        )

    # Get matched subjects AND rows
    res = df.filter(
        pl.col("subject_id").is_in(list(matched_subjects))
        & row_mask
    )

    # Take first row based on `conditions` ordering
    res = (
        res.sort(["_prio", "time"])
        .group_by("subject_id", maintain_order=True)
        .first()
    )

    res = res.drop("_prio")

    return res