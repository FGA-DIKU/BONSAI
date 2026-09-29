import logging
from pathlib import Path

import hydra
import polars as pl
from dotenv import load_dotenv
from hydra.core.plugins import Plugins
from omegaconf import DictConfig

from bonsai.functional.conditions import get_subject_first_row_for_conditions
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin
from bonsai.paths import get_config_path

load_dotenv()
Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="example_cohort",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.paths.input_dir)
    save_path = Path(cfg.paths.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    include = cfg.cohort.include
    exclude = cfg.cohort.exclude

    logging.info(f"Starting create_cohort for `{save_path.stem}`")
    logging.info(f"Including subjects with {include}")
    logging.info(f"Excluding subjects with {exclude}")

    all_subjects = []
    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        for shard in shards:
            df = pl.read_parquet(shard, columns=["subject_id", "time", "code"])

            df = df.drop_nulls(["subject_id", "time", "code"])

            # Exclude subjects matching exclude.conditions
            if exclude is not None:
                exclude_df = get_subject_first_row_for_conditions(
                    df, exclude.conditions, exclude.dependence
                )
                logging.info(f"Excluding {len(exclude_df)} subjects")
                df = df.join(
                    exclude_df.select("subject_id"),
                    on="subject_id",
                    how="anti",
                )

            cohort = get_subject_first_row_for_conditions(
                df, include.conditions, include.depedence
            )

            all_subjects.append(cohort)

    cohort = (
        pl.concat(all_subjects).unique("subject_id")
        if all_subjects
        else pl.DataFrame(schema={"subject_id": pl.Int64})
    )

    logging.info(f"Total number of subjects: {len(cohort):_}")
    logging.info(f"Saving to {save_path}")

    cohort.write_csv(save_path)


if __name__ == "__main__":
    main()
