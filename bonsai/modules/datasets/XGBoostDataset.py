
"""Dataset for XGBoost outcome prediction using processed BONSAI subjects."""

from typing import Dict, List

import numpy as np
from scipy import sparse

from bonsai.functional.censoring import censor_subject
from bonsai.modules.datasets.BaseDataset import BaseDataset


class XGBoostDataset(BaseDataset):
    def __init__(
        self,
        subjects: List[Dict],
        outcomes: Dict[int, dict],
        vocabulary: Dict[str, int],
        feature_mode: str = "one_hot",
    ):
        super().__init__(subjects)
        self.outcomes = outcomes
        self.predict_token_id = vocabulary["[CLS]"]

        if feature_mode == "demographics":
            codes = [code for code in vocabulary if code.startswith("BACKGROUND//SEX//")]
        else:
            codes = [code for code in vocabulary if not code.startswith("[")]

        codes.sort(key=vocabulary.get)
        self.feature_names = ["age"] + codes
        self.code_to_column = {
            vocabulary[code]: column for column, code in enumerate(codes, start=1)
        }

    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        outcome = self.outcomes[subject["subject_id"]]

        if outcome["censor_abspos"] is not None:
            subject = censor_subject(
                subject,
                censor_date_abspos=outcome["censor_abspos"],
                predict_token_id=self.predict_token_id,
            )

        age = float(subject["age"][-1])

        columns = [0] + sorted(
            self.code_to_column[code]
            for code in set(subject["code"].tolist())
            if code in self.code_to_column
        )
        values = [age] + [1.0] * (len(columns) - 1)
        features = sparse.csr_matrix(
            (values, ([0] * len(columns), columns)),
            shape=(1, len(self.feature_names)),
            dtype=np.float32,
        )

        return {
            "subject_id": subject["subject_id"],
            "target": outcome["label"],
            "features": features,
        }

    def to_matrix(self):
        """Return feature matrix, labels, and subject IDs in dataset order."""
        samples = [self[i] for i in range(len(self))]
        return (
            sparse.vstack([sample["features"] for sample in samples], format="csr"),
            np.asarray([sample["target"] for sample in samples], dtype=np.int64),
            np.asarray([sample["subject_id"] for sample in samples], dtype=np.int64),
        )
