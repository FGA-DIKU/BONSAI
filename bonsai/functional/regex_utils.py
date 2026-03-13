import re
import pandas as pd


def is_valid_regex(pattern: str) -> bool:
    """
    Checks whether a string is a valid regular expression pattern.
    """
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def filter_rows_by_regex(df: pd.DataFrame, col: str, regex: str) -> pd.DataFrame:
    """
    Filter rows in a DataFrame based on a regex pattern applied to a specific column.
    All rows containing a match to the regex pattern will be excluded.
    """

    mask = df[col].astype(str).str.contains(regex, case=False, na=False, regex=True)
    return df.loc[~mask]


def exclude_codes(df: pd.DataFrame, exclude_regex: str):
    if not is_valid_regex(exclude_regex):
        raise ValueError(f"Invalid regex: {exclude_regex}")
    df = filter_rows_by_regex(df, col="code", regex=exclude_regex)
    return df
