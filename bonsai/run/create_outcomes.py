from datetime import datetime
from typing import List, Literal
import pandas as pd
import hydra
from omegaconf import DictConfig
from bonsai.paths import get_config_path
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()


@hydra.main(
    config_path=get_config_path(),  # TODO: make this more flexible to allow for different config paths
    config_name="create_outcomes",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    path_data = Path(cfg.paths.data)
    path_outcomes = Path(cfg.paths.outcomes)

    # TODO: Single file for each outcome or one for each split_outcome combination?
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
                outcomes = outcomes.drop(columns="code").rename(columns={"time": "outcome_date"})

                # Set index_date (absolute or relative to outcome_date)
                if index.type == "absolute":
                    outcomes["index_date"] = datetime(**index.date)
                elif index.type == "relative":
                    outcomes["index_date"] = outcomes["outcome_date"] + pd.Timedelta(hours=index.hour_shift)
                else:
                    raise ValueError(f"Index.type only allowed [absolute, relative], not {index.type}")
            
                # Set censor_date (absolute or relative to outcome_date)
                if censor.type == "absolute":
                    outcomes["censor_date"] = datetime(**censor.date)
                elif censor.type == "relative":
                    outcomes["censor_date"] = outcomes["outcome_date"] + pd.Timedelta(hours=censor.hour_shift)
                else:    
                    raise ValueError(f"censor.type only allowed [absolute, relative], not {censor.type}")
                all_outcomes[outcome] = pd.concat((all_outcomes[outcome], outcomes))

    path_outcomes.mkdir(parents=True, exist_ok=True)
    for outcome, df_out in all_outcomes.items():
        # TODO: implement logger?
        print(f"Saving {outcome} to {path_outcomes / outcome}.parquet")
        # print(df_out)
        df_out.to_parquet(path_outcomes / f"{outcome}.parquet")


def find(df, conditions: List, dependence: Literal["independent", "dependent"]):
    """ Returns the first row (priority based on condition order) for each patient that matches the conditions"""
    # Initialization
    df["_prio"] = pd.Series()
    masks = False
    subject_sets = []

    for i, cond in enumerate(conditions):
        cond_mask = df[cond["col"]].isin(cond["vals"]) # Rows that meet condition
        masks |= cond_mask  # OR operation
        df["_prio"] = df["_prio"].mask(cond_mask, i) # Set priority (to take first row later)
        subject_sets.append(set(df.loc[cond_mask, "subject_id"])) # Get subject that match condition
    
    # Toggle betweens any or all conditions met
    if dependence == "independent":
        matched_subjects = set.union(*subject_sets) # Any condition met
    elif dependence == "dependent": # TODO: Implement time_window
        matched_subjects = set.intersection(*subject_sets) # All conditions met
    else:
        raise ValueError(f"Dependence can only be [independent, dependent], not {dependence}")

    # Get matched subjects AND rows
    res = df[df["subject_id"].isin(matched_subjects) & masks]

    # Take first row based on `conditions` ordering
    res = res.sort_values("_prio").groupby("subject_id", sort=False, as_index=False).first()
    res = res.drop(columns="_prio")
    return res

if __name__ == "__main__":
    main()