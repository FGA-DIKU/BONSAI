import logging
from pathlib import Path

import hydra
import polars as pl
from dotenv import load_dotenv
from hydra.core.plugins import Plugins
from omegaconf import DictConfig

from bonsai.functional.outcomes import (
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

    logging.info(f"Starting create_outcome_cohort for `{save_path.stem}`")

    all_outcomes = []

    stats = {
        "initial": 0,
        "after_nulls": 0,
        "index_matches": 0,
        "outcome_matches": 0,
        "after_cohort": 0,
        "after_index_join": 0,
        "after_censor_join": 0,
    }

    for split in cfg.splits:
        shards = sorted((input_dir / split).glob("*.parquet"))

        if not shards:
            raise FileNotFoundError(f"No Parquet files found in {input_dir / split}")

        for shard in shards:
            refs = {}

            df = pl.read_parquet(shard, columns=["subject_id", "time", "code"])

            stats["initial"] += df["subject_id"].drop_nulls().n_unique()

            df = df.drop_nulls(["subject_id", "code"])
            stats["after_nulls"] += df["subject_id"].n_unique()

            refs["index_date"] = get_index_dates(df, cfg.index)
            refs["outcome_date"] = get_outcome_dates(df, cfg.outcome, refs)
            refs["censor_date"] = get_censor_dates(df, cfg.censor, refs)

            stats["index_matches"] += refs["index_date"]["subject_id"].n_unique()
            stats["outcome_matches"] += refs["outcome_date"]["subject_id"].n_unique()

            shard_cohort = get_cohort(df, cfg.cohort, refs)

            cohort = shard_cohort.select("subject_id").unique()
            stats["after_cohort"] += cohort.height

            cohort = cohort.join(refs["outcome_date"], on="subject_id", how="left")

            cohort = cohort.join(refs["index_date"], on="subject_id", how="inner")
            stats["after_index_join"] += cohort["subject_id"].n_unique()

            cohort = cohort.join(refs["censor_date"], on="subject_id", how="inner")
            stats["after_censor_join"] += cohort["subject_id"].n_unique()

            all_outcomes.append(cohort.with_columns(split=pl.lit(split)))

    all_outcomes = pl.concat(all_outcomes).sort("subject_id", maintain_order=True)

    final_patients = all_outcomes["subject_id"].n_unique()
    with_outcome = all_outcomes.filter(pl.col("outcome_date").is_not_null())[
        "subject_id"
    ].n_unique()

    logging.info(
        "\n========== OVERALL COHORT SUMMARY ==========\n"
        f"Initial patients:          {stats['initial']:,}\n"
        f"After null filtering:      {stats['after_nulls']:,}\n"
        f"With index date:           {stats['index_matches']:,}\n"
        f"With outcome match:        {stats['outcome_matches']:,}\n"
        f"After cohort filtering:    {stats['after_cohort']:,}\n"
        f"After index join:          {stats['after_index_join']:,}\n"
        f"After censor join:         {stats['after_censor_join']:,}\n"
        f"Final unique patients:     {final_patients:,}\n"
        f"With outcome date:         {with_outcome:,}\n"
        f"Without outcome date:      {final_patients - with_outcome:,}\n"
        "============================================"
    )

    logging.info(f"Saving to {save_path}")
    all_outcomes.write_parquet(save_path)


if __name__ == "__main__":
    main()
