import pandas as pd
from corebehrt.constants.data import CONCEPT_COL


class ValueCreator:
    """
    A class to load normalise values in data frames.
    Expects a 'result' column and 'concept' column to be present.
    """

    @staticmethod
    def bin_results(
        concepts: pd.DataFrame,
        num_bins=100,
        add_prefix=False,
        separator_regex=None,
    ) -> pd.DataFrame:
        if concepts.empty:
            # Return empty DataFrame with same columns plus the expected new ones
            return concepts.assign(
                index=pd.Series(dtype="int64"),
                order=pd.Series(dtype="int64"),
                code=pd.Series(dtype="object"),
            )
        concepts["binned_value"] = ValueCreator.bin(
            concepts["numeric_value"], num_bins=num_bins
        )

        # Add index + order
        concepts["index"] = concepts.index
        concepts.loc[:, "order"] = 0
        values = concepts.dropna(subset=["binned_value"]).copy()

        # Extract prefix from concept and use it for values codes
        if add_prefix and separator_regex is not None:
            values["prefix"] = values[CONCEPT_COL].str.extract(separator_regex)
            # Handle cases where regex doesn't match
            prefix_na_mask = values["prefix"].isna()
            if prefix_na_mask.any():
                values.loc[prefix_na_mask, "prefix"] = "UNK"
            values.loc[:, "code"] = values["prefix"] + "/" + values["binned_value"]
        else:
            values.loc[:, "code"] = values["binned_value"]

        values.loc[:, "order"] = 1
        concatted = pd.concat([concepts, values])

        # Drop columns that are not needed
        columns_to_drop = ["numeric_value", "binned_value"]
        if add_prefix:
            columns_to_drop.append("prefix")

        return concatted.drop(columns=columns_to_drop)

    @staticmethod
    def bin(normalized_values: pd.Series, num_bins=100) -> pd.Series:
        """
        Bins the values in a series into num_bins bins. Expects the values to be normalised.
        """
        normalized_values = pd.to_numeric(normalized_values, errors="coerce")
        val_mask = normalized_values.notna()
        normalized_values[val_mask] = normalized_values[val_mask].mul(num_bins)
        normalized_values = normalized_values.astype(object)
        normalized_values[val_mask] = (
            normalized_values[val_mask].astype(int).astype(str)
        )
        normalized_values[val_mask] = "VAL_" + normalized_values[val_mask]
        return normalized_values
