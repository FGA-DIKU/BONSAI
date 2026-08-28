import logging
from pathlib import Path
import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
import operator

from hydra.core.plugins import Plugins
from bonsai.paths import get_config_path
from bonsai.functional.outcomes import (
    get_date_from_relative_date,
)
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin

load_dotenv()
Plugins.instance().register(DataCreationSearchpathPlugin)

def get_python_operator(operator_str):
    ops = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
    }
    try:
        return ops[operator_str]
    except KeyError as e:
        raise NotImplementedError(f"Unknown operator: {operator_str}") from e

@hydra.main(
    config_path=get_config_path(),
    config_name="example_outcome1",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.paths.input_dir)
    patient_table = Path(cfg.paths.patient_table)
    mapping_file = Path(cfg.paths.mapping_file)
    mapping_id_column = cfg.paths.mapping_id_column
    mapping_df = pd.read_csv(mapping_file, usecols=[mapping_id_column, "mapping"])
    mapping_dict = mapping_df.set_index(mapping_id_column)["mapping"].to_dict()
    save_path = Path(cfg.paths.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    patient_table = pd.read_csv(patient_table)
    patient_table["subject_id"] = patient_table["m_cpr"].map(mapping_dict)
    py_op = get_python_operator(cfg.outcome.outcome.conditions[0].operator)
    patient_table["label"] = py_op(patient_table[cfg.outcome.outcome.conditions[0].col], cfg.outcome.outcome.conditions[0].value)
    n_before_map = len(patient_table)
    patient_table = patient_table.dropna(subset=["subject_id"])
    if len(patient_table) != n_before_map:
        logging.warning(
            f"Dropped {n_before_map - len(patient_table):_} patient rows with unmapped m_cpr"
        )
    outcome = cfg.outcome.outcome
    index = cfg.outcome.index
    censor = cfg.outcome.censor
    outcome_cols = ["subject_id", "m_cpr", outcome.date, index, "label"]
    print(patient_table[outcome_cols].head())

    logging.info(f"Starting create_outcome for `{save_path.stem}`")
    logging.info(f"Outcome date assigned with {outcome}")
    logging.info(f"Index date assigned with {index}")
    logging.info(f"Censor date assigned with {censor}")

    all_outcomes = pd.DataFrame()
    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        split_subject_ids = set()
        for shard in shards:
            df = pd.read_parquet(shard, columns=["subject_id", "time", "code"])
            df = df.dropna(subset=["subject_id", "time", "code"])
            split_subject_ids.update(df["subject_id"].unique())

        outcomes = patient_table.loc[
            patient_table["subject_id"].isin(split_subject_ids),
            ["subject_id", outcome.date, index, "label"],
        ].rename(columns={outcome.date: "outcome_date", index: "index_date"})
        outcomes["outcome_date"] = pd.to_datetime(outcomes["outcome_date"])
        outcomes["index_date"] = pd.to_datetime(outcomes["index_date"])
        outcomes["outcome_date"] = outcomes["outcome_date"].where(
            outcomes["label"], pd.NaT
        )
        outcomes["split"] = split
        all_outcomes = pd.concat((all_outcomes, outcomes))
        logging.info(
            f"Processed {outcomes['subject_id'].nunique():_} unique subjects "
            f"({len(outcomes):_} outcome rows) in {split}"
        )
    all_outcomes = all_outcomes.reset_index(drop=True)

    n_before = len(all_outcomes)
    missing_index = all_outcomes["index_date"].isna()
    if missing_index.any():
        missing_subject_ids = (
            all_outcomes.loc[missing_index, "subject_id"].dropna().drop_duplicates()
        )
        logging.warning(
            f"Dropping {missing_index.sum():_} rows with missing index_date "
            f"({len(missing_subject_ids):_} subjects; showing up to 20 below)"
        )
        print(
            patient_table.loc[
                patient_table["subject_id"].isin(missing_subject_ids), outcome_cols
            ].head(20)
        )
        all_outcomes = all_outcomes.loc[~missing_index].reset_index(drop=True)
        logging.info(
            f"Kept {len(all_outcomes):_} of {n_before:_} outcome rows after dropping NaN index_date"
        )

    all_outcomes["censor_date"] = get_date_from_relative_date(
        relative_dates=all_outcomes["index_date"],  # Censoring is based on index_date
        relative_hour_shift=censor[
            "relative_hour_shift"
        ],  # 0 sets index_date=censor_date
    )

    logging.info(
        f"Total outcome rows: {len(all_outcomes):_} "
        f"({all_outcomes['subject_id'].nunique():_} unique subjects, "
        f"{(~all_outcomes['outcome_date'].isna()).sum():_} positives)"
    )
    logging.info(f"Saving to {save_path}")
    all_outcomes.to_parquet(save_path)


if __name__ == "__main__":
    main()
