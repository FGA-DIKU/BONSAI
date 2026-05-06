import logging
from pathlib import Path
import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

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
    save_path = Path(cfg.paths.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    patient_table = pd.read_parquet(patient_table)
    outcome = cfg.outcome.outcome
    index = cfg.outcome.index
    censor = cfg.outcome.censor

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

            outcomes = patient_table.loc[df["subject_id"], outcome]
            logging.info(f"Matched {len(outcomes)} subjects")
            outcomes = (
                (
                    df[["subject_id"]]
                    .drop_duplicates()
                    .merge(outcomes, on="subject_id", how="left")
                )
                .drop(columns="code")
                .rename(columns={outcome: "outcome_date"})
            )
            # TODO: handle multiple outcomes pr subject assert len(outcomes) == df["subject_id"].nunique()

            index_dates = patient_table.loc[df["subject_id"], index].rename(columns={index: "index_date"})["subject_id", "index_date"]
            outcomes = outcomes.merge(index_dates, on="subject_id", how="left")
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
