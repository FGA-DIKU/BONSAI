import pandas as pd
from typing import Tuple

from bonsai.functional.creators import (
    assign_index_and_order,
    create_abspos,
    create_age_in_years,
    create_background,
    create_segments,
    sort_features,
)
from bonsai.functional.create_data import check_features_columns


class FeatureCreator:
    """
    A class to create features from patient information and concepts DataFrames.
    We create background, death, age, absolute position, and segments features.
    """

    def __call__(
        self,
        concepts: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        check_features_columns(concepts)
        features, patient_info = create_background(concepts)
        features = create_age_in_years(features)
        features = create_abspos(features)

        if (
            "index" in features.columns and "order" in features.columns
        ):  # TODO: Make default?
            features = assign_index_and_order(features)
        features = features.dropna(subset=["subject_id", "time", "code"])
        features = sort_features(features)  # TODO: Combine with above index/order

        features = create_segments(features)
        features = features.drop(columns=["admission_id", "time", "birthdate"])
        features["subject_id"] = features["subject_id"].astype(int)

        return features, patient_info
