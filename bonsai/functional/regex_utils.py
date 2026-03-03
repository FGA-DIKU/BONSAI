import re
import warnings
import pandas as pd


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


def filter_rows_by_regex(df: pd.DataFrame, col: str, regex: str):
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


def exclude_codes(df: pd.DataFrame, exclude_regex: str):
    if not is_valid_regex(exclude_regex):
        raise ValueError(f"Invalid regex: {exclude_regex}")
    df = filter_rows_by_regex(df, col="code", regex=exclude_regex)
    return df
