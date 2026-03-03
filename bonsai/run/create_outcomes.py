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

    # TODO: Single file for each outcome or one for each split_outcome combination?
    # TODO: Implement some logging?
    all_outcomes = pd.DataFrame()

    for split in cfg.splits:
        shards = [shard for shard in (input_dir / split).glob("*.parquet")]
        for shard in shards:
            df = pd.read_parquet(shard, columns=["subject_id", "time", "code"])
            print(len(df))

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
                all_outcomes[outcome] = pd.concat((all_outcomes[outcome], outcomes))
                print(outcomes)

    path_outcomes.mkdir(parents=True, exist_ok=True)
    for outcome, df_out in all_outcomes.items():
        # TODO: implement logger?
        print(f"Saving {outcome} to {path_outcomes / outcome}.parquet")
        # print(df_out)
        df_out.to_parquet(path_outcomes / f"{outcome}.parquet")


def find(df, conditions: List, dependence: Literal["independent", "dependent"]):
    """Returns the first row (priority based on condition order) for each patient that matches the conditions"""
    # Initialization
    df["_prio"] = pd.Series()
    masks = False
    subject_sets = []

    for i, cond in enumerate(conditions):
        cond_mask = df[cond["col"]].isin(cond["vals"])  # Rows that meet condition
        masks |= cond_mask  # OR operation
        df["_prio"] = df["_prio"].mask(
            cond_mask, i
        )  # Set priority (to take first row later)
        subject_sets.append(
            set(df.loc[cond_mask, "subject_id"])
        )  # Get subject that match condition

    # Toggle betweens any or all conditions met
    if dependence == "independent":
        matched_subjects = set.union(*subject_sets)  # Any condition met
    elif dependence == "dependent":  # TODO: Implement time_window
        matched_subjects = set.intersection(*subject_sets)  # All conditions met
    else:
        raise ValueError(
            f"Dependence can only be [independent, dependent], not {dependence}"
        )

    # Get matched subjects AND rows
    res = df[df["subject_id"].isin(matched_subjects) & masks]

    # Take first row based on `conditions` ordering
    res = (
        res.sort_values("_prio")
        .groupby("subject_id", sort=False, as_index=False)
        .first()
    )
    res = res.drop(columns="_prio")
    return res


if __name__ == "__main__":
    main()
