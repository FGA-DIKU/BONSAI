import pandas as pd


def load_concept(path) -> pd.DataFrame:
    """
    Load concept data from formatted_data_dir.
    Expects time column to be present.
    Returns a pandas dataframe.
    """

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unknown file type: {path}")

    df["time"] = df["time"].dt.tz_localize(None)  # to prevent tz-naive/tz-aware issues
    return df
