Basic usage:

1. Create data.
`python bonsai/run/create_data.py --config-name example_data dataset=correlated_MEDS_data`
We use the [example_data.yaml](./configs/data_creation/example_data.yaml) config which transforms the correlated_MEDS_data in the example_data folder into the training format. This data will be saved in `data/correlated_MEDS_data`

2. Pretrain model. 
`python bonsai/run/pretrain.py --config-name example_pretrain dataset=correlated_MEDS_data`
We use the [pretrain.yaml](./configs/pretrain.yaml) config to have a short resource-light training that can run locally and point it to the dataset created in step 1.

3. Create outcomes (labels for finetuning)
`python bonsai/run/create_outcome.py --config-name example_outcome1 dataset=correlated_MEDS_data`
We use the [example_outcome.yaml](./configs/data_creation/example_outcome1.yaml) config which processes the target outcomes for the correlated_MEDS_data in the example_data folder and saves them in an outcome file in `data/correlated_MEDS_data/outcomes/example_outcome1.parquet`

4. Finetune model.
`python bonsai/run/finetune.py --config-name example_finetune dataset=correlated_MEDS_data outcome=example_outcome1 pretrain_path=/path/to/your/pretrained/checkpoints/best.ckpt`
We use the [finetune.yaml](./configs/finetune.yaml) config to have a short resource-light training that can run locally and point it to the dataset created in step 1, the checkpoint created in step 2, and the labels created in step 3.

5. Train model.
`python bonsai/run/train.py --config-name example_finetune dataset=correlated_MEDS_data outcome=example_outcome1`
We use the [finetune.yaml](./configs/finetune.yaml) config to have a short resource-light no-pretraining training that can run locally and point it to the dataset created in step 1 and the labels created in step 3.

To use the old pre-lightning version use:
```
git checkout tags/pre-lightning
```

