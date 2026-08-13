import subprocess

correct = subprocess.run(
    [
        "python",
        "bonsai/run/create_data.py",
        "--config-name",
        "examples/example_value_data",
        "dataset=lab_correlated_MEDS_data_not_fused",
    ],
    check=True,
    text=True,
)

correct = subprocess.run(
    [
        "python",
        "bonsai/run/create_outcome.py",
        "--config-name",
        "examples/example_outcome_val",
        "dataset=lab_correlated_MEDS_data_not_fused",
    ],
    check=True,
    text=True,
)

correct = subprocess.run(
    [
        "python",
        "bonsai/run/pretrain.py",
        "--config-name",
        "examples/example_pretrain_val",
        "dataset=lab_correlated_MEDS_data_not_fused",
    ],
    check=True,
    text=True,
)
#
# correct = subprocess.run(
#    [
#        "python",
#        "bonsai/run/finetune.py",
#        "--config-name",
#        "examples/example_finetune_val",
#        "dataset=lab_correlated_MEDS_data_not_fused",
#    ],
#    check=True,
#    text=True,
# )
