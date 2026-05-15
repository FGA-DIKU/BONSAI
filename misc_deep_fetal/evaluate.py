import argparse

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import (
    BinarySensitivityAtSpecificity,
    BinarySpecificityAtSensitivity,
)

def main(predictions_file: str):
    predictions = pd.read_csv(predictions_file)
    auc = roc_auc_score(predictions["label"], predictions["prob"])
    sens_at_spec_metric = BinarySensitivityAtSpecificity(min_specificity=0.85)
    sens_at_spec, _ = sens_at_spec_metric(
        torch.tensor(predictions["prob"], dtype=torch.float32),
        torch.tensor(predictions["label"], dtype=torch.int64),
    )
    spec_at_sens_metric = BinarySpecificityAtSensitivity(min_sensitivity=0.70)
    spec_at_sens, _ = spec_at_sens_metric(
        torch.tensor(predictions["prob"], dtype=torch.float32),
        torch.tensor(predictions["label"], dtype=torch.int64),
    )
    print(f"AUC: {auc}")
    print(f"Sensitivity at specificity 0.85: {sens_at_spec}")
    print(f"Specificity at sensitivity 0.70: {spec_at_sens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument(
        "--predictions-file",
        required=True,
        help="Path to predictions CSV",
    )
    args = parser.parse_args()
    main(args.predictions_file)
