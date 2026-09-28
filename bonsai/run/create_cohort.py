import logging
from pathlib import Path

import hydra
import polars as pl
from dotenv import load_dotenv
from omegaconf import DictConfig

from hydra.core.plugins import Plugins
from bonsai.paths import get_config_path
from bonsai.functional.cohorts import get_cohort_subjects
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin


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
            df = pl.read_parquet(
                shard,
                columns=["subject_id", "time", "code"],
            )

            df = df.drop_nulls(["subject_id", "time", "code"])

            cohort = get_cohort_subjects(
                df=df,
                include_conditions=include.conditions,
                include_dependence=include.dependence,
                exclude_conditions=(
                    exclude.conditions if exclude is not None else None
                ),
                exclude_dependence=(
                    exclude.dependence if exclude is not None else None
                ),
            )

            all_subjects.append(cohort)

    cohort = (
        pl.concat(all_subjects).unique("subject_id")
        if all_subjects
        else pl.DataFrame(
            schema={"subject_id": pl.Int64}
        )
    )

    logging.info(f"Total number of subjects: {len(cohort):_}")
    logging.info(f"Saving to {save_path}")

    cohort.write_parquet(save_path)


if __name__ == "__main__":
    main()