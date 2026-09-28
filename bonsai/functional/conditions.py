from typing import List, Literal
import polars as pl

def get_subject_first_row_for_conditions(
    df: pl.DataFrame, conditions: List, dependence: Literal["independent", "dependent"]
) -> pl.DataFrame:
    """Returns the first row (priority based on condition order) for each subject that matches the conditions"""
    # Initialization
    df = df.with_columns(_prio=pl.lit(None).cast(pl.Int32))
    row_mask = pl.lit(False)
    subject_sets = []

    # Find matches (dataframe rows AND subject_ids) of conditions
    for i, cond in enumerate(conditions):
        cond_expr = pl.col(cond["col"]).is_in(cond["vals"])  # Rows that meet condition
        row_mask = row_mask | cond_expr  # OR operation
        df = df.with_columns(
            _prio=pl.when(cond_expr & pl.col("_prio").is_null())
            .then(pl.lit(i))
            .otherwise(pl.col("_prio"))
        )  # Set priority (to take first row later)
        subject_sets.append(
            set(df.filter(cond_expr).get_column("subject_id").to_list())
        )  # Get subjects that match condition

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
    res = df.filter(pl.col("subject_id").is_in(list(matched_subjects)) & row_mask)

    # Take first row based on `conditions` ordering
    res = (
        res.sort(["_prio", "time"]).group_by("subject_id", maintain_order=True).first()
    )
    res = res.drop("_prio")
    return res