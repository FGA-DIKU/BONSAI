# BONSAI

[![Pipeline tests](https://github.com/FGA-DIKU/EHR/actions/workflows/pipeline.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/pipeline.yml)
[![Unittests](https://github.com/FGA-DIKU/EHR/actions/workflows/unittests.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/unittests.yml)
[![Format](https://github.com/FGA-DIKU/EHR/actions/workflows/format.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/format.yml)
[![Lint](https://github.com/FGA-DIKU/EHR/actions/workflows/lint.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/lint.yml)

> **A framework for processing and analyzing Electronic Health Records (EHR) data using transformer-based models.**

BONSAI helps researchers and data scientists preprocess EHR data, train models, and generate outcomes for downstream clinical predictions and analyses.

### Setup (requires Python 3.12)

```
git clone https://github.com/FGA-DIKU/BONSAI.git
pip install -e .
cp template_env .env
```

You can adapt the paths in .env to specify alternative directories containing custom configs, input data or where model checkpoint should be saved.

### Basic usage:

1. Create data.
`python bonsai/run/create_data.py --config-name examples/example_data dataset=correlated_MEDS_data`
We use the [example_data.yaml](./configs/examples/example_data.yaml) config which transforms the correlated_MEDS_data in the example_data folder into the training format. This data will be saved in `data/correlated_MEDS_data`

2. Pretrain model. 
`python bonsai/run/pretrain.py --config-name examples/example_pretrain dataset=correlated_MEDS_data`
We use the [pretrain.yaml](./configs/examples/example_pretrain.yaml) config to have a short resource-light training that can run locally and point it to the dataset created in step 1.

3. Create outcomes (labels for finetuning)
`python bonsai/run/create_outcome.py --config-name examples/example_outcome1 dataset=correlated_MEDS_data`
We use the [example_outcome.yaml](./configs/examples/example_outcome1.yaml) config which processes the target outcomes for the correlated_MEDS_data in the example_data folder and saves them in an outcome file in `data/correlated_MEDS_data/outcomes/examples/example_outcome1.parquet`

4. Finetune model.
`python bonsai/run/finetune.py --config-name examples/example_finetune dataset=correlated_MEDS_data outcome=examples/example_outcome1 pretrain_path=/path/to/your/pretrained/checkpoints/best.ckpt`
We use the [finetune.yaml](./configs/examples/example_finetune.yaml) config to have a short resource-light training that can run locally and point it to the dataset created in step 1, the checkpoint created in step 2, and the labels created in step 3.

5. Train model.
`python bonsai/run/train.py --config-name examples/example_finetune dataset=correlated_MEDS_data outcome=examples/example_outcome1`
We use the [finetune.yaml](./configs/examples/example_finetune.yaml) config to have a short resource-light no-pretraining training that can run locally and point it to the dataset created in step 1 and the labels created in step 3.

To use the old pre-lightning version use:
```
git checkout tags/pre-lightning
```

## Outcomes creation
We provide a standardized script to generate outcomes in [create_outcome.py](/bonsai/run/create_outcome.py), however you can also provide your own, in case our script doesn't accommodate your needs. 

An outcome file requires the following 5 columns saved as a `.parquet` file:

1. A `subject_id` to define the person of interest
2. A `split` string (e.g. "train", "tuning", "held_out") that denotes which split the given row belongs to (i.e. we make one file for all splits)
3. A `outcome_date` that denotes when the outcome happened (can be null)
4. A `index_date` that denotes from when we consider the prediction (can't be null)
5. A `censor_date` that denotes the data cutoff (can't be null)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and formatting
- Testing requirements
- Pull request process
- Issue reporting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BONSAI in your research, please cite the following paper:

```bibtex
@article{Montgomery2025,
  author = {Montgomery, A. and others},
  title = {BONSAI: A framework for processing and analysing {E}lectronic {H}ealth {R}ecords ({EHR}) data using transformer-based models},
  journal = {Journal of Open Source Software},
  volume = {10},
  number = {114},
  pages = {8869},
  year = {2025},
  doi = {10.21105/joss.08869}
}
```
