import pandas as pd
import hydra
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
from bonsai.paths import get_config_path
from bonsai.functional.outcomes import find, set_dates
from bonsai.modules.hydra.plugins import DataCreationSearchpathPlugin
from hydra.core.plugins import Plugins
import os

load_dotenv()

Plugins.instance().register(DataCreationSearchpathPlugin)


load_dotenv()


@hydra.main(
    config_path=get_config_path(),  # TODO: make this more flexible to allow for different config paths
    config_name="default_create_outcomes",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.data.input_dir)
    save_path = Path(cfg.data.save_path)
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    outcome = cfg.outcome
    match = outcome.match
    exclude = outcome.exclude
    index = outcome.index
    censor = outcome.censor
    print(outcome, outcome.censor, outcome.censor.get("hour_shift"))
    # TODO: Single file for each outcome or one for each split_outcome combination?
    # TODO: Implement some logging?
    all_outcomes = pd.DataFrame()

    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        for shard in shards:
            df = pd.read_parquet(shard, columns=["subject_id", "time", "code"])
            print(len(df))

            df = df.dropna(subset=["subject_id", "time", "code"])

            # Find rows matching exclude.conditions and exclude them
            if exclude is not None:
                exclude_df = find(df, exclude.conditions, exclude.dependence)
                df = df[~df["subject_id"].isin(exclude_df["subject_id"])]

            # Find the outcomes matching match.conditions
            outcomes = find(df, match.conditions, match.dependence)
            outcomes = (
                df[["subject_id"]]
                .drop_duplicates()
                .merge(outcomes, on="subject_id", how="left")
            )

            assert len(outcomes) == df["subject_id"].nunique()

            outcomes = outcomes.drop(columns="code").rename(
                columns={"time": "outcome_date"}
            )
            outcomes["index_date"] = set_dates(
                date_type=index.type,  # Absolute/relative
                outcome_dates=outcomes["outcome_date"],  # Required for relative
                hour_shift=index.get("hour_shift"),  # Required for relative
                date=index.get("date"),  # Required for absolute
            )
            outcomes["censor_date"] = set_dates(
                date_type=censor.type,  # Absolute/relative
                outcome_dates=outcomes["outcome_date"],  # Required for relative
                hour_shift=censor.get("hour_shift"),  # Required for relative
                date=censor.get("date"),  # Required for absolute
            )
            all_outcomes = pd.concat((all_outcomes, outcomes))

    all_outcomes.to_parquet(save_path)


if __name__ == "__main__":
    main()
