import re
import warnings
import pandas as pd
from typing import Set


def is_valid_regex(pattern: str) -> bool:
    """
    Checks whether a string is a valid regular expression pattern.

    Args:
        pattern: The regex pattern string to validate.

    Returns:
        True if the pattern can be compiled as a regular expression, False otherwise.
    """
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def filter_rows_by_regex(df, col, regex):
    """
    Filter rows in a DataFrame based on a regex pattern applied to a specific column.
    All rows containing a match to the regex pattern will be excluded.

    Args:
        df: DataFrame to filter.
        col: Column name to apply the regex filter.
        regex: Regex pattern to filter rows by.

    Returns:
        Filtered DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mask = df[col].astype(str).str.contains(regex, case=False, na=False, regex=True)
    return df.loc[~mask]


def check_required_columns(
    df: pd.DataFrame, required_columns: Set[str], type_: str
) -> None:
    if not required_columns.issubset(set(df.columns)):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(f"Missing columns in {type_}: {missing_columns}")


def check_features_columns(df: pd.DataFrame) -> None:
    """Check if required columns are present in features."""
    check_required_columns(
        df, required_columns={"subject_id", "time", "code"}, type_="features"
    )
