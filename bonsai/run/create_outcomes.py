import pandas as pd
import hydra
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig
from bonsai.paths import get_config_path
from bonsai.functional.outcomes import find, set_dates


load_dotenv()


@hydra.main(
    config_path=get_config_path(),  # TODO: make this more flexible to allow for different config paths
    config_name="create_outcomes",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    path_data = Path(cfg.paths.data)
    path_outcomes = Path(cfg.paths.outcomes)

    # TODO: Implement some logging?
    all_outcomes = {out: pd.DataFrame() for out in cfg.outcomes}
    for split in cfg.splits:
        shards = [shard for shard in (path_data / split).glob("*.parquet")]
        for shard in shards:
            df = pd.read_parquet(shard, columns=["subject_id", "time", "code"])
            df = df.dropna(subset=["subject_id", "time", "code"])

            for outcome in cfg.outcomes:
                match = cfg.outcomes[outcome].match
                exclude = cfg.outcomes[outcome].exclude
                index = cfg.outcomes[outcome].index
                censor = cfg.outcomes[outcome].censor

                # Find rows matching exclude.conditions and exclude them
                if exclude is not None:
                    exclude_df = find(df, exclude.conditions, exclude.dependence)
                    df = df[~df["subject_id"].isin(exclude_df["subject_id"])]

                # Find the outcomes matching match.conditions
                outcomes = find(df, match.conditions, match.dependence)
                outcomes = df[["subject_id"]].drop_duplicates().merge(outcomes, on="subject_id", how="left")
                assert len(outcomes) == df["subject_id"].nunique()

                outcomes = outcomes.drop(columns="code").rename(columns={"time": "outcome_date"})

                # Set index_date (absolute or relative to outcome_date)
                outcomes["index_date"] = set_dates(
                    date_type=index.type, # Absolute/relative
                    outcome_dates=outcomes["outcome_date"], # Required for relative
                    hour_shift=index.get("hour_shift"), # Required for relative
                    date=index.get("date") # Required for absolute
                )

                # Set censor_date (absolute or relative to outcome_date)
                outcomes["censor_date"] = set_dates(
                    date_type=censor.type, # Absolute/relative
                    outcome_dates=outcomes["outcome_date"], # Required for relative
                    hour_shift=censor.get("hour_shift"), # Required for relative
                    date=censor.get("date") # Required for absolute
                )

                all_outcomes[outcome] = pd.concat((all_outcomes[outcome], outcomes))

    path_outcomes.mkdir(parents=True, exist_ok=True)
    for outcome, df_out in all_outcomes.items():
        # TODO: implement logger?
        print(f"Saving {outcome} to {path_outcomes / outcome}.parquet")
        df_out.to_parquet(path_outcomes / f"{outcome}.parquet")


if __name__ == "__main__":
    main()