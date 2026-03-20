import logging
from pathlib import Path
import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from hydra.core.plugins import Plugins
from bonsai.paths import get_config_path
from bonsai.functional.outcomes import find, set_dates
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin

load_dotenv()
Plugins.instance().register(DataCreationSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),  # TODO: make this more flexible to allow for different config paths
    config_name="example_outcome1",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.paths.input_dir)
    save_path = Path(cfg.paths.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    exclude = cfg.outcome.exclude
    match = cfg.outcome.match
    index = cfg.outcome.index
    censor = cfg.outcome.censor

    logging.info(f"Starting create_outcome for `{save_path.stem}`")
    logging.info(f"Excluding with {exclude}")
    logging.info(f"Matching with {match}")
    logging.info(f"Index date assigned with {index}")
    logging.info(f"Censor date assigned with {censor}")

    all_outcomes = pd.DataFrame()
    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        for shard in shards:
            df = pd.read_parquet(shard, columns=["subject_id", "time", "code"])

            df = df.dropna(subset=["subject_id", "time", "code"])

            # Find rows matching exclude.conditions and exclude them
            if exclude is not None:
                exclude_df = find(df, exclude.conditions, exclude.dependence)
                logging.info(f"Excluding {len(exclude_df)} subjects")
                df = df[~df["subject_id"].isin(exclude_df["subject_id"])]

            # Find the outcomes matching match.conditions
            outcomes = find(df, match.conditions, match.dependence)
            logging.info(f"Matched {len(outcomes)} subjects")
            outcomes = (
                df[["subject_id"]]
                .drop_duplicates()
                .merge(outcomes, on="subject_id", how="left")
            )
            assert len(outcomes) == df["subject_id"].nunique()

            outcomes = outcomes.drop(columns="code").rename(
                columns={"time": "outcome_date"}
            )

            logging.info(f"Assigning {index.type} index_dates")
            outcomes["index_date"] = set_dates(
                date_type=index.type,  # Absolute/relative/exposure
                dates=outcomes["outcome_date"],  # Required for relative
                hour_shift=index.get("hour_shift"),  # Required for relative
                date=index.get("date"),  # Required for absolute
                subjects=outcomes[["subject_id"]],  # Required for exposure
                df=df,  # Required for exposure
                conditions=index.conditions,  # Required for exposure
                dependence=index.dependence,  # Required for exposure
            )

            outcomes["censor_date"] = set_dates(
                date_type="relative",  # Only relative censoring
                dates=outcomes["index_date"],  # Censoring is based on index_date
                hour_shift=censor["hour_shift"],  # 0 sets index_date=censor_date
            )

            outcomes["split"] = split
            all_outcomes = pd.concat((all_outcomes, outcomes))
    logging.info(
        f"Total number of subjects: {len(all_outcomes):_} ({(~all_outcomes['outcome_date'].isna()).sum():_} positives)"
    )
    logging.info(f"Saving to {save_path}")
    all_outcomes.to_parquet(save_path)


if __name__ == "__main__":
    main()
