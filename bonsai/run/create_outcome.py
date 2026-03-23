import logging
from pathlib import Path
import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from hydra.core.plugins import Plugins
from bonsai.paths import get_config_path
from bonsai.functional.outcomes import match, set_dates
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
    outcome = cfg.outcome.outcome
    index = cfg.outcome.index
    censor = cfg.outcome.censor

    logging.info(f"Starting create_outcome for `{save_path.stem}`")
    logging.info(f"Excluding subjects with {exclude}")
    logging.info(f"Outcome date assigned with {outcome}")
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
                exclude_df = match(df, exclude.conditions, exclude.dependence)
                logging.info(f"Excluding {len(exclude_df)} subjects")
                df = df[~df["subject_id"].isin(exclude_df["subject_id"])]

            # Find the outcomes matching outcome.conditions
            outcomes = match(df, outcome.conditions, outcome.dependence)
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

            outcomes["index_date"] = set_dates(
                date_type=index.type,  # Absolute/relative/exposure
                relative_dates=outcomes["outcome_date"],  # Required for relative
                relative_hour_shift=index.get(
                    "relative_hour_shift"
                ),  # Required for relative
                absolute_date=index.get("absolute_date"),  # Required for absolute
                subjects=outcomes[["subject_id"]],  # Required for exposure
                df=df,  # Required for exposure
                exposure_conditions=index.get("conditions"),  # Required for exposure
                exposure_dependence=index.get("dependence"),  # Required for exposure
            )

            outcomes["split"] = split
            all_outcomes = pd.concat((all_outcomes, outcomes))
    all_outcomes = all_outcomes.reset_index(drop=True)

    if (dates := all_outcomes["index_date"]).isna().any():
        logging.warning(
            f"Found {dates.isna().sum()} NaN index dates -- Replacing them with randomly sampled {(~dates.isna()).sum()} non-NaNs"
        )
        if (~dates.isna()).sum() == 0:
            raise ValueError(f"No non-NaN indexing dates found using {index}")
        # Randomly sample from non-null
        samples = dates.dropna().sample(
            n=dates.isna().sum(), replace=True
        )  # TODO: Add seed?
        all_outcomes["index_date"] = dates.fillna(
            value=pd.Series(samples.values, index=dates[dates.isna()].index)
        )

    all_outcomes["censor_date"] = set_dates(
        date_type="relative",  # Only relative censoring
        relative_dates=all_outcomes["index_date"],  # Censoring is based on index_date
        relative_hour_shift=censor[
            "relative_hour_shift"
        ],  # 0 sets index_date=censor_date
    )

    logging.info(
        f"Total number of subjects: {len(all_outcomes):_} ({(~all_outcomes['outcome_date'].isna()).sum():_} positives)"
    )
    logging.info(f"Saving to {save_path}")
    all_outcomes.to_parquet(save_path)


if __name__ == "__main__":
    main()
