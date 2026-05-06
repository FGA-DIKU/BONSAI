import logging
from pathlib import Path
import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
import pickle 

from hydra.core.plugins import Plugins
from bonsai.paths import get_config_path
from bonsai.functional.outcomes import (
    get_subject_first_row_for_conditions,
    get_date_from_absolute_date,
    get_date_from_relative_date,
    get_date_from_exposure_date,
    fill_nans_with_sampled,
)
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin

load_dotenv()
Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="example_outcome1",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.paths.input_dir)
    patient_table = Path(cfg.paths.patient_table)
    mapping_file = Path(cfg.paths.mapping_file)
    with open(mapping_file, "rb") as f:
        mapping_dict = pickle.load(f)
    save_path = Path(cfg.paths.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    patient_table = pd.read_csv(patient_table)
    patient_table["subject_id"] = patient_table["m_cpr"].map(mapping_dict)
    outcome = cfg.outcome.outcome
    index = cfg.outcome.index
    censor = cfg.outcome.censor
    print(patient_table[["m_cpr", "subject_id", outcome, index]].head())

    logging.info(f"Starting create_outcome for `{save_path.stem}`")
    logging.info(f"Outcome date assigned with {outcome}")
    logging.info(f"Index date assigned with {index}")
    logging.info(f"Censor date assigned with {censor}")

    all_outcomes = pd.DataFrame()
    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        for shard in shards:
            df = pd.read_parquet(shard, columns=["subject_id", "time", "code"])

            df = df.dropna(subset=["subject_id", "time", "code"])

            subject_ids = df["subject_id"].astype(patient_table.index.dtype, copy=False)
            outcomes = (
                patient_table.reindex(subject_ids)[[outcome, index]]
                .rename(columns={outcome: "outcome_date", index: "index_date"})
                .reset_index(names="subject_id")
            )
            outcomes["split"] = split
            all_outcomes = pd.concat((all_outcomes, outcomes))
    all_outcomes = all_outcomes.reset_index(drop=True)

    if (dates := all_outcomes["index_date"]).isna().any():
        logging.warning(
            f"Found {dates.isna().sum()} NaN index dates -- Replacing them with randomly sampled {(~dates.isna()).sum()} non-NaNs"
        )
        all_outcomes["index_date"] = fill_nans_with_sampled(dates)

    all_outcomes["censor_date"] = get_date_from_relative_date(
        relative_dates=all_outcomes["index_date"],  # Censoring is based on index_date
        relative_hour_shift=censor[
            "relative_hour_shift"
        ],  # 0 sets index_date=censor_date
    )

    logging.info(
        f"Total number of subjects: {len(all_outcomes):_} ({(~all_outcomes['outcome_date'].isna()).sum():_} positives)"
    )
    logging.info(f"Saving to {save_path}")
    all_outcomes.to_parquet(save_path)


if __name__ == "__main__":
    main()
