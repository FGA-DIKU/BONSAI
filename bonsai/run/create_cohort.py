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

    logging.info(f"Starting create_cohort for `{save_path.stem}`")

    all_subjects = set()
    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        for shard in shards:
            df = pl.read_parquet(shard, columns=["subject_id", "time", "code"])

            df = df.drop_nulls(["subject_id", "code"])

            criterion_population = set(df["subject_id"].to_list())
            for criterion in cfg.cohort.conditional_criteria:
                if criterion.action == "exclude":
                    exclude_df = get_subject_first_row_for_conditions(
                        df, criterion.conditions, criterion.dependence
                    )
                    logging.info(f"Excluding {len(exclude_df)} subjects")
                    criterion_population -= set(exclude_df["subject_id"].to_list())
                elif criterion.action == "include":
                    include_df = get_subject_first_row_for_conditions(
                        df, criterion.conditions, criterion.dependence
                    )
                    logging.info(f"Including {len(include_df)} subjects")
                    criterion_population &= set(include_df["subject_id"].to_list())
                else:
                    raise ValueError(
                        f"Criterion action can only be [include, exclude], not {criterion.action}"
                    )

            all_subjects.update(criterion_population)

    cohort = pl.DataFrame(list(all_subjects), schema={"subject_id": pl.Int64})

    logging.info(f"Total number of subjects: {len(cohort):_}")
    logging.info(f"Saving to {save_path}")

    cohort.write_csv(save_path)


if __name__ == "__main__":
    main()
