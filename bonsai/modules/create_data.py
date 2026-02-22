import pandas as pd
from pathlib import Path
from typing import Optional
import pyarrow as pa
import torch

from bonsai.functional.io_ops import load_concept
from bonsai.functional.create_data import filter_rows_by_regex, is_valid_regex
from bonsai.modules.features.values import ValueCreator
from bonsai.modules.features.features import FeatureCreator
from bonsai.modules.features.tokenizer import EHRTokenizer


def create_and_save_features(
    path_data: Path,
    path_features: Path,
    splits,
    logger,
    agg_kwargs: Optional[dict] = None,
    exclude_regex: Optional[str] = None,
    values_kwargs: Optional[dict] = None,
) -> None:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    combined_patient_info = pd.DataFrame()
    for split_name in splits:
        logger.info(f"Creating features for {split_name}")
        path_name = path_data / split_name
        if not path_name.exists():
            raise ValueError(f"{path_name} does not exist")

        split_save_path = path_features / split_name
        split_save_path.mkdir(parents=True, exist_ok=True)
        shards = [shard for shard in path_name.iterdir()]

        logger.info(f"Found {len(shards)} shards to process in {split_name}")

        # Initialize counters for this split
        total_concept_counts = {
            "loaded": 0,
            "after_agg": 0,
            "after_exclusion": 0,
            "after_value_handling": 0,
            "after_incorrect_age_removal": 0,
        }

        for shard_idx, shard in enumerate(shards, 1):
            logger.info(f"Processing shard {shard_idx}/{len(shards)}: {shard}")
            shard_n = shard.stem

            concepts = load_concept(shard)
            total_concept_counts["loaded"] += len(concepts)

            if agg_kwargs and (agg_kwargs.get("agg_type") is not None):
                concepts = handle_aggregations(
                    concepts,
                    **agg_kwargs,
                )
            total_concept_counts["after_agg"] += len(concepts)

            if exclude_regex is not None:
                concepts = exclude_concepts(
                    concepts,
                    exclude_regex,
                )
            total_concept_counts["after_exclusion"] += len(concepts)

            if values_kwargs and "numeric_value" in concepts.columns:
                concepts = handle_numeric_values(concepts, **values_kwargs)
            total_concept_counts["after_value_handling"] += len(concepts)
            
            feature_creator = FeatureCreator()
            features, patient_info = feature_creator(concepts)
            combined_patient_info = pd.concat([combined_patient_info, patient_info])

            features = exclude_incorrect_event_ages(features)
            total_concept_counts["after_incorrect_age_removal"] += len(features)

            features.to_parquet(
                f"{split_save_path}/{shard_n}.parquet",
                index=False,
                schema=pa.schema(
                    {
                        "subject_id": "int64",
                        "age": "float32",
                        "abspos": "float64",
                        "segment": "int32",
                        "code": "str",
                    }
                ),
            )

        # Log final statistics for this split
        logger.info(f"Finished processing {split_name}")
        logger.info(f"Total concepts loaded: {total_concept_counts['loaded']}")
        logger.info(f"Total concepts after aggregation: {total_concept_counts['after_agg']}")
        logger.info(f"Total concepts after exclusion: {total_concept_counts['after_exclusion']}")
        logger.info(f"Total concepts after value handling: {total_concept_counts['after_value_handling']}")
        logger.info(
            f"Total concepts after incorrect age removal: {total_concept_counts['after_incorrect_age_removal']}"
        )

    patient_info_path = path_features / "patient_info.parquet"
    combined_patient_info.to_parquet(patient_info_path, index=False)
    logger.info(
        f"Total number of patients across all splits: {len(combined_patient_info)}"
    )


def handle_aggregations(
    concepts: pd.DataFrame,
    agg_type: Optional[str] = None,
    agg_window: Optional[int] = None,
    regex: str = ".*",
) -> pd.DataFrame:
    """
    Aggregates rows in the DataFrame based on PID, TIMESTAMP, and CONCEPT columns.
    Filters rows based on the provided regex before aggregation and concatenates excluded rows back after aggregation.
    Keeps NaN values in TIMESTAMP column to preserve background codes.
    Optionally aggregates values within a specified time window.

    Args:
        concepts: DataFrame to aggregate.
        agg_type: Aggregation type (e.g., 'first', 'sum', 'mean', etc.). If None, no aggregation is performed.
        agg_window: Time window in hours for aggregation. If None, no time window aggregation is performed.
        regex: Regular expression to filter rows based on the CONCEPT_COL before aggregation.

    Returns:
        Aggregated DataFrame with specified rows.
    """
    matching_rows = concepts[concepts["subject_id"].astype(str).str.match(regex)]
    non_matching_rows = concepts[~concepts["subject_id"].astype(str).str.match(regex)]
    nan_rows = matching_rows[matching_rows[["time"]].isna().any(axis=1)]
    non_nan_rows = matching_rows.dropna(subset=["time"])

    if agg_window:
        min_time = non_nan_rows["time"].min()
        normalized_timestamps = (non_nan_rows["time"] - min_time).dt.total_seconds()
        normalized_timestamps = normalized_timestamps.fillna(-1)
        non_nan_rows["TIME_GROUP"] = (
            normalized_timestamps // (agg_window * 3600)
        ).astype(int)

        aggregated_df = (
            non_nan_rows.groupby(["subject_id", "TIME_GROUP", "code"])
            .agg(agg_type)
            .reset_index()
        )
        aggregated_df = aggregated_df.drop(columns="TIME_GROUP")
    else:
        aggregated_df = (
            non_nan_rows.groupby(["subject_id", "time", "code"])
            .agg(agg_type)
            .reset_index()
        )

    # Concatenate aggregated rows with NaN rows and non-matching rows
    concatted_df = pd.concat(
        [aggregated_df, nan_rows, non_matching_rows], ignore_index=True
    )
    return concatted_df


def exclude_concepts(concepts, exclude_regex):
    if not is_valid_regex(exclude_regex):
        raise ValueError(f"Invalid regex: {exclude_regex}")
    concepts = filter_rows_by_regex(concepts, col="subject_id", regex=exclude_regex)
    concepts = concepts.copy()  # to avoid SettingWithCopyWarning
    return concepts


def exclude_incorrect_event_ages(
    df: pd.DataFrame, min_age: int = -1, max_age: int = 120
) -> pd.DataFrame:
    """Exclude patients with incorrect ages (outside defined range)"""
    return df[(df["age"] >= min_age) & (df["age"] <= max_age)]


def handle_numeric_values(
    concepts: pd.DataFrame,
    num_bins: int = 100,
    add_prefix: bool = False,
    separator_regex: Optional[str] = None,
    drop: bool = False,
) -> pd.DataFrame:
    """
    Process numeric values in concepts DataFrame based on configuration.
    Either bins the values or drops the numeric_value column.

    Parameters:
        concepts: DataFrame containing concepts data
        num_bins: Number of bins to use if binning values.
        add_prefix: Whether to add a prefix to the binned value codes.
        separator_regex: Optional regex to extract prefix from subject_id for value codes. Only used if add_prefix is True.
        drop: Whether to drop the numeric_value column instead of binning.
    """
    if drop:
        return concepts.drop(columns=["numeric_value"])

    if separator_regex is not None and not is_valid_regex(separator_regex):
        raise ValueError(f"Invalid regex: {separator_regex}")
    return ValueCreator.bin_results(
        concepts,
        num_bins=num_bins,
        add_prefix=add_prefix,
        separator_regex=separator_regex,
    )

def load_tokenize_and_save(
    path_features: Path,
    tokenizer: EHRTokenizer,
    path_tokenized: Path,
    split: str,
):
    """
    Load df for split, tokenize and write to tokenized_path.
    """
    pids = set()
    (path_tokenized / split).mkdir(parents=True, exist_ok=True)
    for shard in (path_tokenized / split).iterdir():
        shard_path = path_features / split / shard
        shard_n = shard.stem
        df = pd.read_parquet(shard_path).set_index("subject_id")

        df = tokenizer(df).reset_index()
        df.to_parquet(
            path_tokenized / split / f"{shard_n}.parquet",
            index=False,
            schema=pa.schema({
                "subject_id": "int64",
                "age": "float32",
                "abspos": "float64",
                "segment": "int32",
                "code": "int32",
            }),
        )
        pids.update(df["subject_id"].unique().tolist())
    torch.save(pids, path_tokenized / f"pids_{split}.pt")  # save pids as ints