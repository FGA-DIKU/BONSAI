"""Visualize BONSAI patient embeddings from pretrained or finetuned checkpoints."""

import json
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from umap import UMAP

from bonsai.functional.collate import dynamic_padding
from bonsai.functional.outcomes import binarize_outcomes, finalize_outcomes
from bonsai.functional.subject_data import filter_subject_data, prepare_subject_data
from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset
from bonsai.modules.datasets.PretrainDataset import PretrainDataset
from bonsai.modules.networks.bonsai_nets import (
    BonsaiBase,
    BonsaiFinetune,
    BonsaiValueFinetune,
    BonsaiPretrain,
    BonsaiValuePretrain,
)
from bonsai.paths import get_config_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Standard plotting settings. The YAML only needs paths and plot definitions.
DEFAULT_CONFIG = OmegaConf.create(
    {
        "paths": {
            "population": "auto",
            "run_config": None,
            "output_dir": "umap_results",
            "summary_filename": "run_summary.json",
        },
        "model": {"pooling": "auto"},
        "labels": {"start_include": "auto", "end_include": "auto"},
        "inference": {
            "max_len": None,
            "cutoff_date": None,
            "split": "held_out",
            "batch_size": 64,
            "num_workers": 0,
            "device": "auto",
            "amp_dtype": "auto",
        },
        "umap": {
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "seed": 42,
        },
        "figure": {
            "size": [8, 6.5],
            "dpi": 200,
            "font_size": 11,
            "background": "white",
            "grid": False,
            "point_size": 9,
            "point_linewidth": 0,
            "rasterized": True,
            "alpha": 0.85,
            "x_label": "UMAP 1",
            "y_label": "UMAP 2",
            "aspect": "equal",
            "legend": {
                "show_counts": True,
                "count_format": " (n={count:,})",
                "markerscale": 2,
                "frameon": False,
            },
        },
    }
)


def get_run_config(checkpoint):
    config_path = checkpoint.parent / "hydra" / "config.yaml"
    return OmegaConf.load(config_path)

def load_model(checkpoint: Path, attn_type=None):
    """Restore either BONSAI model type from its Lightning checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    hp = saved["hyper_parameters"]
    weights = {
        name.removeprefix("model."): tensor
        for name, tensor in saved["state_dict"].items()
        if name.startswith("model.")
    }
    stage = "finetuned" if "finetune_head.weight" in weights else "pretrained"
    value_mode = hp.get("value_embedding_mode")
    classes = {
        ("finetuned", False): BonsaiFinetune,
        ("finetuned", True): BonsaiValueFinetune,
        ("pretrained", False): BonsaiPretrain,
        ("pretrained", True): BonsaiValuePretrain,
    }
    params = {
        key: hp[key]
        for key in (
            "vocab_size",
            "max_seqlen",
            "hidden_size",
            "num_layers",
            "bias",
            "dropout",
            "causal",
            "attn_type",
        )
    }
    params["attn_type"] = attn_type or hp["attn_type"]
    params["attention_dropout"] = 0.0
    if value_mode is not None:
        params["value_embedding_mode"] = value_mode
    if stage == "finetuned":
        params["predict_token_id"] = hp["predict_token_id"]

    model = classes[(stage, value_mode is not None)](**params)
    model.load_state_dict(weights, strict=True)
    return model.eval(), hp, stage, params["attn_type"]


def collate_pretrained(batch):
    """PretrainDataset has no labels; BONSAI's dynamic_padding expects a target."""
    for patient in batch:
        patient["target"] = torch.tensor([-100])
    return dynamic_padding(batch)


def load_patients(
    cfg: DictConfig, run_cfg: DictConfig, vocabulary: dict, stage: str, hp: dict
):
    """Apply the same BONSAI censoring/truncation to tokenized patient shards."""
    data_path = Path(cfg.paths.dataset)
    shard_path = (
        data_path / "held_out" if (data_path / "held_out").is_dir() else data_path
    )
    dataset_root = shard_path.parent if shard_path.name == "held_out" else shard_path

    subjects = prepare_subject_data(shard_path)
    population = cfg.paths.population
    population_path = (
        dataset_root / "cohorts" / "population_full.csv"
        if population == "auto"
        else population
    )
    if population_path is not None and Path(population_path).exists():
        subjects = filter_subject_data(
            subjects, pl.read_csv(population_path)["subject_id"]
        )
    subjects.sort(key=lambda subject: subject["subject_id"])
    background_length = int((subjects[0]["segment"] == 1).sum())
    max_len = (
        run_cfg.training.max_len
        if cfg.inference.max_len is None
        else cfg.inference.max_len
    )

    if cfg.paths.outcomes is None:
        if stage == "finetuned":
            raise ValueError(
                "Finetuned checkpoints require an outcomes file for patient cutoffs"
            )
        dataset = PretrainDataset(
            subjects,
            max_len=max_len,
            background_length=background_length,
            cutoff_date=OmegaConf.to_container(cfg.inference.cutoff_date, resolve=True)
            if cfg.inference.cutoff_date is not None
            else None,
        )
        collate_fn = collate_pretrained
    else:
        outcomes = pl.read_parquet(cfg.paths.outcomes).filter(
            pl.col("split") == cfg.inference.split
        )
        if cfg.labels.start_include == "auto":
            if stage == "pretrained":
                raise ValueError(
                    "Set labels.start_include (and labels.end_include) for pretrained checkpoints"
                )
            start = OmegaConf.to_container(run_cfg.labels.start_include, resolve=True)
            end = OmegaConf.to_container(run_cfg.labels.end_include, resolve=True)
        else:
            start = OmegaConf.to_container(cfg.labels.start_include, resolve=True)
            end = OmegaConf.to_container(cfg.labels.end_include, resolve=True)
        labels = finalize_outcomes(
            binarize_outcomes(outcomes, start_include=start, end_include=end)
        )
        subjects = [subject for subject in subjects if subject["subject_id"] in labels]
        dataset = FinetuneDataset(
            subjects,
            outcomes=labels,
            predict_token_id=hp["predict_token_id"] if stage == "finetuned" else None,
            background_length=background_length,
            max_len=max_len,
        )
        collate_fn = dynamic_padding

    return DataLoader(
        dataset,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.inference.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )


def make_plot_specs(plots, vocabulary: dict) -> list[dict]:
    """Resolve the configured code names to IDs from vocabulary.pt."""
    for plot in plots:
        if plot["kind"] != "code":
            continue
        if "values" in plot:
            plot["mode"] = "category"
            plot["ids"] = [
                vocabulary[plot["prefix"] + value] for value in plot["values"]
            ]
        else:
            plot["mode"] = "presence"
            plot["ids"] = [
                token_id
                for code, token_id in vocabulary.items()
                if code == plot["prefix"]
                or (plot["match"] == "startswith" and code.startswith(plot["prefix"]))
            ]
            if not plot["ids"]:
                raise ValueError(f"No vocabulary codes match {plot['prefix']!r}")
    return plots


def code_labels(batch: dict, spec: dict) -> np.ndarray:
    """Label the censored and truncated tokens actually seen by the model."""
    tokens = batch["code"]
    attention = batch["attention_mask"]
    if spec["mode"] == "category":
        a, b = [
            (tokens == token_id).logical_and(attention).any(dim=1)
            for token_id in spec["ids"]
        ]
        result = torch.full_like(a, 2, dtype=torch.long)
        result[a & ~b] = 0
        result[b & ~a] = 1
        result[a & b] = 3
    else:
        ids = torch.tensor(spec["ids"], dtype=tokens.dtype)
        result = (torch.isin(tokens, ids) & attention).any(dim=1).long()
    return result.numpy()


def predict_embeddings(
    model,
    loader,
    plot_specs,
    device: torch.device,
    amp_dtype: str,
    attn_type: str,
    stage: str,
    pooling: str,
    has_outcomes: bool,
):
    """Collect backbone representations and configured labels in one model pass."""
    model = model.to(device).eval()
    hp = model.hparams
    if attn_type == "flash" and device.type != "cuda":
        raise RuntimeError(
            f"Attention backend '{attn_type}' is unavailable on {device}. "
            "Specify a supported backend in inference.attn_type."
        )
    use_amp = attn_type == "flash"
    precision = {
        "auto": (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[amp_dtype]
    code_specs = [spec for spec in plot_specs if spec["kind"] == "code"]
    results = {"subject_id": [], "embedding": []}
    if stage == "finetuned":
        results["risk"] = []
    if has_outcomes:
        results["outcome"] = []
    results.update({spec["name"]: [] for spec in code_specs})

    with torch.inference_mode():
        for batch in loader:
            for spec in code_specs:
                results[spec["name"]].append(code_labels(batch, spec))
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=precision, enabled=use_amp):
                hidden = BonsaiBase.forward(model, batch)
                if stage == "finetuned":
                    prediction_token = hidden[batch["code"] == hp["predict_token_id"]]
                    risk = model.finetune_head(prediction_token).sigmoid().flatten()
                    results["risk"].append(risk.float().cpu().numpy())
                mode = (
                    ("prediction_token" if stage == "finetuned" else "mean")
                    if pooling == "auto"
                    else pooling
                )
                if mode == "prediction_token":
                    embedding = hidden[batch["code"] == hp["predict_token_id"]]
                elif mode == "mean":
                    mask = batch["attention_mask"].unsqueeze(-1)
                    embedding = (hidden * mask).sum(1) / mask.sum(1)
                elif mode == "last":
                    last = batch["attention_mask"].sum(1).long() - 1
                    embedding = hidden[torch.arange(len(last), device=device), last]
                else:
                    raise ValueError(f"Unknown embedding pooling: {mode}")

            results["subject_id"].append(batch["subject_id"].cpu().numpy())
            results["embedding"].append(embedding.float().cpu().numpy())
            if "outcome" in results:
                results["outcome"].append(batch["target"].flatten().cpu().numpy())

    return {name: np.concatenate(chunks) for name, chunks in results.items()}


def plot_umap(
    coords: np.ndarray,
    values: np.ndarray,
    plot: dict,
    cfg: DictConfig,
    output_dir: Path,
):
    """Render the requested plot using only text, colors and styling from YAML."""
    plt.rcParams["font.size"] = cfg.figure.font_size
    fig, ax = plt.subplots(figsize=tuple(cfg.figure.size), layout="constrained")
    fig.patch.set_facecolor(cfg.figure.background)
    ax.set_facecolor(cfg.figure.background)
    common = dict(
        s=cfg.figure.point_size,
        alpha=cfg.figure.alpha,
        linewidths=cfg.figure.point_linewidth,
        rasterized=cfg.figure.rasterized,
    )

    if plot["kind"] == "risk":
        points = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=plot["colormap"],
            vmin=plot["vmin"],
            vmax=plot["vmax"],
            **common,
        )
        fig.colorbar(points, ax=ax, label=plot["colorbar_label"])
    else:
        for idx, category in enumerate(plot["categories"]):
            mask = values == idx
            label = category["label"]
            if cfg.figure.legend.show_counts:
                label += cfg.figure.legend.count_format.format(count=int(mask.sum()))
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=category["color"],
                label=label,
                **common,
            )
        ax.legend(
            title=plot.get("legend_title"),
            markerscale=cfg.figure.legend.markerscale,
            frameon=cfg.figure.legend.frameon,
        )

    ax.set(
        title=plot["title"],
        xlabel=plot.get("x_label", cfg.figure.x_label),
        ylabel=plot.get("y_label", cfg.figure.y_label),
    )
    ax.set_aspect(cfg.figure.aspect, adjustable="box")
    ax.grid(cfg.figure.grid)
    if plot.get("note"):
        fig.text(0.5, 0.01, plot["note"], ha="center")
    fig.savefig(output_dir / plot["filename"], dpi=cfg.figure.dpi, bbox_inches="tight")
    plt.close(fig)


@hydra.main(
    config_path=get_config_path(), config_name="plot_embeddings", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.merge(DEFAULT_CONFIG, cfg)
    checkpoint = Path(cfg.paths.checkpoint)
    data_path = Path(cfg.paths.dataset)
    root = data_path.parent if data_path.name == "held_out" else data_path

    vocabulary = torch.load(
        root / "vocabulary.pt", map_location="cpu", weights_only=True
    )
    model, hp, stage, attn_type = load_model(
        checkpoint,
        attn_type=cfg.get("inference", {}).get("attn_type"),
    )
    run_cfg = get_run_config(checkpoint)
    loader = load_patients(cfg, run_cfg, vocabulary, stage, hp)
    plots = make_plot_specs(OmegaConf.to_container(cfg.plots, resolve=True), vocabulary)

    available = {"code"} | ({"outcome"} if cfg.paths.outcomes is not None else set())
    if stage == "finetuned":
        available.add("risk")
    unavailable = {plot["kind"] for plot in plots} - available
    if unavailable:
        raise ValueError(
            f"Plot types {sorted(unavailable)} are not available for this checkpoint/data"
        )
    if stage == "pretrained" and cfg.model.pooling == "prediction_token":
        raise ValueError(
            "Pretrained checkpoints have no trained prediction token; use mean or last pooling"
        )

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(
        device_name if cfg.inference.device == "auto" else cfg.inference.device
    )
    results = predict_embeddings(
        model,
        loader,
        plots,
        device,
        cfg.inference.amp_dtype,
        attn_type,
        stage,
        cfg.model.pooling,
        cfg.paths.outcomes is not None,
    )
    embeddings = results["embedding"]
    coords = UMAP(
        n_components=cfg.umap.n_components,
        n_neighbors=min(cfg.umap.n_neighbors, len(embeddings) - 1),
        min_dist=cfg.umap.min_dist,
        metric=cfg.umap.metric,
        random_state=cfg.umap.seed,
    ).fit_transform(embeddings)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for plot in plots:
        source = plot["name"] if plot["kind"] == "code" else plot["kind"]
        plot_umap(coords, results[source], plot, cfg, output_dir)

    summary = {
        "checkpoint": str(checkpoint),
        "model_stage": stage,
        "embedding_pooling": cfg.model.pooling,
        "dataset": str(data_path),
        "outcomes": str(cfg.paths.outcomes),
        "patients": len(results["subject_id"]),
        "embedding_dim": embeddings.shape[1],
        "plots": [plot["filename"] for plot in plots],
    }
    (output_dir / cfg.paths.summary_filename).write_text(json.dumps(summary, indent=2))
    print(
        f"Saved {len(plots)} plots to {output_dir} using one UMAP projection ({stage})"
    )


if __name__ == "__main__":
    main()
