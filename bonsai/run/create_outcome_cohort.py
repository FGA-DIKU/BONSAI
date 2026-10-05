import logging
from pathlib import Path

import hydra
import polars as pl
from dotenv import load_dotenv
from hydra.core.plugins import Plugins
from omegaconf import DictConfig

from bonsai.functional.outcomes import (
    fill_nans_with_sampled,
    get_censor_dates,
    get_cohort,
    get_index_dates,
    get_outcome_dates,
)
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin
from bonsai.paths import get_config_path

load_dotenv()
Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="example_outcome1",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.paths.input_dir)
    save_path = Path(cfg.paths.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting create_outcome for `{save_path.stem}`")

    all_subjects = set()
    all_outcomes = []
    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        for shard in shards:
            df = pl.read_parquet(shard, columns=["subject_id", "time", "code"])
            df = df.drop_nulls(["subject_id", "code"])

            df_cohort = df.select("subject_id").unique()

            # Define outcome dates (returns subject_id, outcome_date)
            outcome_dates = get_outcome_dates(
                df, cfg.outcome.outcome.action, cfg.outcome.outcome.conditional_criteria
            )

            # Define index dates (returns subject_id, index_date)
            index_dates = get_index_dates(
                df, cfg.outcome.index.action, cfg.outcome.index.conditional_criteria
            )

            # Join to cohort and add split column
            shard_outcome = (
                df_cohort.join(outcome_dates, on="subject_id", how="left")
                .join(index_dates, on="subject_id", how="left")
                .with_columns(split=pl.lit(split))
            )

            # Filter outcome_date >= index_date #TODO: Long term this should be optional, but requires first_match revamp
            shard_outcome = shard_outcome.filter(
                (pl.col("outcome_date").is_null())
                | (pl.col("outcome_date") >= pl.col("index_date"))
            )
            all_outcomes.append(shard_outcome)

            # Get cohort (done inside loop to optimize file reading)
            shard_cohort = get_cohort(df, cfg.cohort.get("conditional_criteria", []))
            all_subjects.update(shard_cohort)

    all_outcomes = (
        pl.concat(all_outcomes).sort("subject_id", maintain_order=True)
        if all_outcomes
        else pl.DataFrame()
    )
    logging.info(f"Total number of subjects: {len(all_outcomes):_}")

    # Apply cohort filtering (calculated inside loop above)
    all_outcomes = all_outcomes.filter(pl.col("subject_id").is_in(all_subjects))
    assert len(all_outcomes) == len(all_subjects), (
        "Mismatch in cohort size after filtering"
    )
    logging.info(f"Total number of subjects in cohort: {len(all_subjects):_}")

    if (
        index_dates := all_outcomes["index_date"]
    ).is_null().any() and cfg.index.action != "exposure":
        logging.warning(
            f"Found {index_dates.is_null().sum()} NaN index dates -- Replacing them..."
        )
        all_outcomes = all_outcomes.with_columns(
            index_date=fill_nans_with_sampled(all_outcomes["index_date"], seed=42)
        )
    elif cfg.index.action == "exposure":
        all_outcomes = all_outcomes.filter(pl.col("index_date").is_not_null())

    # Apply censoring based on index date and censoring criteria
    censor_dates = get_censor_dates(
        all_outcomes, cfg.outcome.censor.action, cfg.outcome.censor.conditional_criteria
    )
    all_outcomes = all_outcomes.join(censor_dates, on="subject_id", how="left")

    logging.info(
        f"Total number of subjects: {len(all_outcomes):_} ({all_outcomes['outcome_date'].is_not_null().sum():_} positives)"
    )
    logging.info(f"Saving to {save_path}")
    all_outcomes.write_parquet(save_path)


if __name__ == "__main__":
    main()
