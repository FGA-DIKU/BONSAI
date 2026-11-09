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


def add_count(stats, key, df):
    stats[key][0] += len(df)
    stats[key][1] += df["subject_id"].n_unique()


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
        key: [0, 0]
        for key in (
            "initial",
            "after_nulls",
            "index_matches",
            "outcome_matches",
            "after_cohort",
            "after_outcome_join",
            "after_index_join",
            "after_censor_join",
        )
    }

    for split in cfg.splits:
        shards = sorted((input_dir / split).glob("*.parquet"))

        if not shards:
            raise FileNotFoundError(f"No Parquet files found in {input_dir / split}")

        for shard in shards:
            refs = {}

            df = pl.read_parquet(shard, columns=["subject_id", "time", "code"])
            add_count(stats, "initial", df)

            df = df.drop_nulls(["subject_id", "code"])
            add_count(stats, "after_nulls", df)

            refs["index_date"] = get_index_dates(df, cfg.index)
            refs["outcome_date"] = get_outcome_dates(df, cfg.outcome, refs)
            refs["censor_date"] = get_censor_dates(df, cfg.censor, refs)

            add_count(
                stats,
                "index_matches",
                refs["index_date"].filter(pl.col("index_date").is_not_null()),
            )
            add_count(
                stats,
                "outcome_matches",
                refs["outcome_date"].filter(pl.col("outcome_date").is_not_null()),
            )

            shard_cohort = get_cohort(df, cfg.cohort, refs)
            cohort = shard_cohort.select("subject_id").unique()
            add_count(stats, "after_cohort", cohort)

            cohort = cohort.join(
                refs["outcome_date"],
                on="subject_id",
                how="left",
            )
            add_count(stats, "after_outcome_join", cohort)

            cohort = cohort.join(
                refs["index_date"],
                on="subject_id",
                how="inner",
            )
            add_count(stats, "after_index_join", cohort)

            cohort = cohort.join(
                refs["censor_date"],
                on="subject_id",
                how="inner",
            )
            add_count(stats, "after_censor_join", cohort)

            all_outcomes.append(cohort.with_columns(split=pl.lit(split)))

    all_outcomes = pl.concat(all_outcomes).sort("subject_id", maintain_order=True)

    with_outcome = all_outcomes.filter(pl.col("outcome_date").is_not_null())
    without_outcome = all_outcomes.filter(pl.col("outcome_date").is_null())

    logging.info(
        "\n========== OVERALL COHORT SUMMARY ==========\n"
        f"Initial:                  {stats['initial'][0]:,} rows | {stats['initial'][1]:,} patients\n"
        f"After null filtering:     {stats['after_nulls'][0]:,} rows | {stats['after_nulls'][1]:,} patients\n"
        f"With index date:          {stats['index_matches'][0]:,} rows | {stats['index_matches'][1]:,} patients\n"
        f"With outcome match:       {stats['outcome_matches'][0]:,} rows | {stats['outcome_matches'][1]:,} patients\n"
        f"After cohort filtering:   {stats['after_cohort'][0]:,} rows | {stats['after_cohort'][1]:,} patients\n"
        f"After outcome join:       {stats['after_outcome_join'][0]:,} rows | {stats['after_outcome_join'][1]:,} patients\n"
        f"After index join:         {stats['after_index_join'][0]:,} rows | {stats['after_index_join'][1]:,} patients\n"
        f"After censor join:        {stats['after_censor_join'][0]:,} rows | {stats['after_censor_join'][1]:,} patients\n"
        f"Final cohort:             {len(all_outcomes):,} rows | {all_outcomes['subject_id'].n_unique():,} patients\n"
        f"With outcome date:        {len(with_outcome):,} rows | {with_outcome['subject_id'].n_unique():,} patients\n"
        f"Without outcome date:     {len(without_outcome):,} rows | {without_outcome['subject_id'].n_unique():,} patients\n"
        "============================================"
    )

    logging.info(f"Saving to {save_path}")
    all_outcomes.write_parquet(save_path)


if __name__ == "__main__":
    main()
