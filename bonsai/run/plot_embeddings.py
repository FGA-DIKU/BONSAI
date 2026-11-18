#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")  # headless plotting must be configured before importing pyplot
import matplotlib.pyplot as plt

LOG = logging.getLogger("bonsai_umap")

@dataclass(frozen=True)
class HorizonData:
    eligible: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    status: np.ndarray

def _censor_survival(times: np.ndarray, events: np.ndarray):
    order = np.argsort(times)
    unique = np.unique(times[order])
    survival = 1.0
    values = []
    for time in unique:
        at_risk = np.sum(times >= time)
        censored = np.sum((times == time) & (events == 0))
        values.append((time, survival))  # left-continuous G(t-)
        if at_risk:
            survival *= max(0.0, 1.0 - censored / at_risk)
    knots = np.asarray([item[0] for item in values], dtype=float)
    left = np.asarray([item[1] for item in values], dtype=float)

    def evaluate(query):
        query = np.asarray(query, dtype=float)
        # G(query-): include censoring at earlier knots, not at query itself.
        indices = np.searchsorted(knots, query, side="left")
        if not len(left):
            return np.ones_like(query)
        extended = np.append(left, survival)
        return extended[indices]

    return evaluate

def horizon_ipcw_data(
    times: np.ndarray,
    events: np.ndarray,
    horizon: float,
    *,
    min_censor_survival: float = 0.05,
) -> HorizonData:
    """Construct cumulative-incidence labels and IPCW weights at ``horizon``."""
    times = np.asarray(times, dtype=float)
    raw_events = np.asarray(events, dtype=float)
    valid = np.isfinite(times) & (times >= 0) & np.isin(raw_events, [0, 1, 2])
    events = np.where(np.isfinite(raw_events), raw_events, -1).astype(int)
    evaluate_g = _censor_survival(times[valid], events[valid])
    primary = valid & (events == 1) & (times <= horizon)
    competing = valid & (events == 2) & (times <= horizon)
    observed_through_horizon = valid & (times > horizon)
    eligible = primary | competing | observed_through_horizon
    target = primary.astype(float)
    evaluation_time = np.minimum(times, horizon)
    weights = np.zeros_like(times, dtype=float)
    weights[eligible] = 1.0 / np.clip(
        evaluate_g(evaluation_time[eligible]), min_censor_survival, None
    )
    status = np.full(times.shape, "ineligible/early censor", dtype=object)
    status[observed_through_horizon] = "known event-free"
    status[competing] = "competing death"
    status[primary] = "primary event"
    return HorizonData(eligible, target, weights, status)

def _default_bandwidth(coords: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors

    if len(coords) < 2:
        return 1.0
    k = min(50, len(coords) - 1)
    distances = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors()[0][:, -1]
    positive = distances[distances > 0]
    return float(np.median(positive)) if positive.size else 1.0

def _display_grid(coords: np.ndarray, grid_size: int, min_padding: float = 0.):
    padding = np.maximum(0.04 * np.maximum(np.ptp(coords, axis=0), 1e-6), min_padding)
    xs = np.linspace(
        coords[:, 0].min() - padding[0], coords[:, 0].max() + padding[0], grid_size
    )
    ys = np.linspace(
        coords[:, 1].min() - padding[1], coords[:, 1].max() + padding[1], grid_size
    )
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])

def patient_footprint_margin(coords, points, *, radius=None, neighbors=8):
    """Signed union-of-discs footprint; geometry only, never outcome dependent.

    A common radius derived from median k-neighbor distance follows the cloud
    without a convex hull bridging distant islands. Positive means inside.
    This is a display footprint, not an additional claim of statistical support.
    """
    from scipy.spatial import cKDTree
    coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    tree = cKDTree(coords)
    if radius is None:
        k = min(len(coords), max(2, int(neighbors) + 1))
        distances = tree.query(coords, k=k)[0]
        distances = distances[:, -1] if distances.ndim == 2 else distances
        positive = distances[distances > 0]
        radius = float(np.median(positive)) if positive.size else 1.
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Footprint radius must be positive and finite.")
    return (float(radius) - tree.query(np.asarray(points), k=1)[0]), float(radius)

def neighbor_operator(embedding: np.ndarray, n_neighbors: int):
    """Row-stochastic kNN kernel built in the *full* embedding.

    The two-dimensional layout only decides where a grid point sits among
    patients; this operator decides which patients are its neighbours, in the
    space the model actually uses. Passing ``None`` downstream reproduces the
    plain two-dimensional kernel exactly.
    """
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    embedding = np.asarray(embedding, dtype=float)
    size = len(embedding)
    k = int(min(max(int(n_neighbors), 1), size))
    fitted = NearestNeighbors(n_neighbors=k).fit(embedding)
    distances, indices = fitted.kneighbors(embedding)  # includes the patient
    furthest = distances[:, -1]
    positive = furthest[furthest > 0]
    scale = max(float(np.median(positive)) if positive.size else 1.0, 1e-8)
    kernel = np.exp(-0.5 * (distances / scale) ** 2)
    kernel /= np.clip(kernel.sum(axis=1, keepdims=True), 1e-12, None)
    rows = np.repeat(np.arange(size), k)
    return csr_matrix((kernel.ravel(), (rows, indices.ravel())), shape=(size, size))

def _kernel_block(
    block: np.ndarray, coords: np.ndarray, scale: float, operator=None
) -> np.ndarray:
    """Grid-point x patient kernel weights, optionally routed through kNN."""
    distance2 = ((block[:, None] - coords[None]) ** 2).sum(axis=2)
    kernel = np.exp(-0.5 * distance2 / (scale * scale))
    if operator is not None:
        kernel = operator.T.dot(kernel.T).T
    return kernel

def _kernel_surfaces(
    coords: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    bandwidth: float,
    grid_size: int,
    bounds_coords: np.ndarray | None = None,
    min_padding: float = 0.,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx, yy, grid = _display_grid(coords if bounds_coords is None else bounds_coords,
                                grid_size, min_padding=min_padding)
    numerator = np.zeros(len(grid))
    denominator = np.zeros(len(grid))
    squared = np.zeros(len(grid))
    scale = max(float(bandwidth), 1e-8)
    for start in range(0, len(grid), 512):
        stop = min(start + 512, len(grid))
        kernel = _kernel_block(grid[start:stop], coords, scale) * weights[None]
        denominator[start:stop] = kernel.sum(axis=1)
        numerator[start:stop] = kernel @ values
        squared[start:stop] = (kernel * kernel).sum(axis=1)
    surface = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    effective_support = np.divide(
        denominator**2, squared, out=np.zeros_like(denominator), where=squared > 0
    )
    return (
        xx,
        yy,
        surface.reshape(xx.shape),
        effective_support.reshape(xx.shape),
        denominator.reshape(xx.shape),
    )

@dataclass(frozen=True)
class CalibrationResidual:
    """Spatial calibration residual D(x) = R_obs(x) - R_pred(x)."""

    observed: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    standard_error: np.ndarray
    z: np.ndarray
    support: np.ndarray
    blank: np.ndarray
    display: np.ma.MaskedArray
    label: str
    n_patients: int
    neighborhood: str

def spatial_calibration_residual(
    coords: np.ndarray,
    predicted_cif: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    bandwidth: float,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
    grid_size: int = 80,
    embedding: np.ndarray | None = None,
    n_neighbors: int = 50,
    support_mask: np.ndarray | None = None,
    min_effective_support: float = 25.0,
    standardize: bool = True,
    n_bootstrap: int = 200,
    held_out: np.ndarray | None = None,
    random_state: int = 0,
) -> CalibrationResidual:
    """Local calibration of a risk model over its own embedding.

    For every grid point ``x`` of the display layout this smooths two
    per-patient quantities through *one* kernel pass, so the two surfaces
    differ only in what is averaged:

    ``R_obs(x)``
        the IPCW cause-specific cumulative incidence at the horizon, i.e. the
        surface panel C shows;
    ``R_pred(x)``
        the model's predicted cause-specific cumulative incidence at the same
        horizon, which must already be a CIF (see
        :func:`cause_specific_cif_from_hazards`).

    ``D(x) = R_obs(x) - R_pred(x)`` is positive where the model under-predicts
    risk in that region of its embedding and negative where it over-predicts.
    ``predicted_cif`` carries no sampling noise relative to the outcome labels,
    so the standard error of ``D`` is the standard error of the local IPCW
    estimate, taken here by a patient-level bootstrap that reuses the cached
    kernel.

    Neighbourhoods come from ``embedding`` (the full representation) when it is
    supplied; the two-dimensional ``coords`` then only place the answer for
    display. Where a grid point may be shown is still decided in two dimensions
    by ``support_mask`` -- panel E's effective-support mask -- because a kNN
    kernel is spread across its neighbours everywhere, including regions of the
    layout that hold no patients at all. With no ``embedding`` the pass reduces
    to panel C's own kernel and ``observed`` is panel C's surface exactly.

    ``held_out`` restricts the estimate to patients the model never saw, which
    is what makes this a calibration check rather than a re-reading of the
    training fit.
    """
    coords = np.asarray(coords, dtype=float)
    predicted_cif = np.asarray(predicted_cif, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(coords) == len(predicted_cif) == len(target) == len(weights):
        raise ValueError("coords, predicted_cif, target and weights must align.")
    finite = np.isfinite(predicted_cif)
    if finite.any() and (
        predicted_cif[finite].min() < -1e-9 or predicted_cif[finite].max() > 1 + 1e-9
    ):
        raise ValueError(
            "predicted_cif must be a cumulative incidence in [0, 1]; convert hazards "
            "or survival curves with cause_specific_cif_from_hazards() first."
        )
    if grid is None:
        xx, yy, points = _display_grid(coords, grid_size)
    else:
        xx, yy = grid
        points = np.column_stack([xx.ravel(), yy.ravel()])
    used = finite & np.isfinite(target) & np.isfinite(weights) & (weights > 0)
    if held_out is not None:
        used &= np.asarray(held_out, dtype=bool)
    if not used.any():
        raise ValueError("No patients remain for the calibration residual.")
    local_coords = coords[used]
    local_predicted = predicted_cif[used]
    local_target = target[used]
    local_weights = weights[used]
    operator = None
    neighborhood = "2D layout"
    if embedding is not None:
        operator = neighbor_operator(
            np.asarray(embedding, dtype=float)[used], n_neighbors
        )
        neighborhood = f"full embedding (k={min(int(n_neighbors), int(used.sum()))})"

    size = len(points)
    observed = np.full(size, np.nan)
    predicted = np.full(size, np.nan)
    support = np.zeros(size)
    standard_error = np.full(size, np.nan)
    draws = None
    if n_bootstrap and n_bootstrap > 0:
        generator = np.random.default_rng(random_state)
        share = np.full(len(local_coords), 1.0 / len(local_coords))
        draws = generator.multinomial(
            len(local_coords), share, size=int(n_bootstrap)
        ).T.astype(float)
    scale = max(float(bandwidth), 1e-8)
    for start in range(0, size, 512):
        stop = min(start + 512, size)
        kernel = (
            _kernel_block(points[start:stop], local_coords, scale, operator)
            * local_weights[None]
        )
        denominator = kernel.sum(axis=1)
        squared = (kernel * kernel).sum(axis=1)
        support[start:stop] = np.divide(
            denominator**2, squared, out=np.zeros_like(denominator), where=squared > 0
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            observed[start:stop] = np.where(
                denominator > 0, (kernel @ local_target) / denominator, np.nan
            )
            predicted[start:stop] = np.where(
                denominator > 0, (kernel @ local_predicted) / denominator, np.nan
            )
            if draws is not None:
                resampled_total = kernel @ draws
                # Bootstrap the paired observed-minus-predicted residual,
                # rather than only the outcome surface. Model and G stay fixed.
                resampled_event = (kernel * (local_target - local_predicted)[None]) @ draws
                replicates = np.where(
                    resampled_total > 0, resampled_event / resampled_total, np.nan
                )
                standard_error[start:stop] = np.nanstd(replicates, axis=1, ddof=1)

    shape = xx.shape
    observed = observed.reshape(shape)
    predicted = predicted.reshape(shape)
    support = support.reshape(shape)
    standard_error = standard_error.reshape(shape)
    residual = observed - predicted
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(standard_error > 0, residual / standard_error, np.nan)
    blank = ~np.isfinite(residual)
    if operator is None:
        blank = blank | (support < min_effective_support)
    if support_mask is not None:
        blank = blank | np.asarray(support_mask, dtype=bool)
    if standardize:
        blank = blank | ~np.isfinite(z)
        display = np.ma.masked_where(blank, z)
        label = "Observed − predicted (local z)"
    else:
        display = np.ma.masked_where(blank, residual)
        label = "Observed − predicted (CIF)"
    return CalibrationResidual(
        observed=observed,
        predicted=predicted,
        residual=residual,
        standard_error=standard_error,
        z=z,
        support=support,
        blank=blank,
        display=display,
        label=label,
        n_patients=int(used.sum()),
        neighborhood=neighborhood,
    )


def plot_risk_maps(coords, times, events, predicted_risk, *, horizon=365.,
                   bandwidth=None, grid_size=70, min_effective_support=25.,
                   min_density_fraction=.01, title='Outcome', risk_max=1.):
    """Return (figure, grid_data) for aligned evaluation-patient arrays.

    events: 0=censoring, 1=target event, 2=competing event.
    predicted_risk: cumulative incidence of event 1 by horizon (not a hazard).
    Kernel observed/predicted risks use identical IPCW-weighted neighborhoods.
    Sparse/extrapolated locations are blank. Residuals are descriptive; no
    pointwise or simultaneous significance claim is made.
    """
    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    predicted_risk = np.asarray(predicted_risk, dtype=float)
    n = len(coords)
    if coords.shape != (n, 2) or n < 3 or not np.isfinite(coords).all():
        raise ValueError('coords must contain at least three finite (x, y) rows')
    if any(v.shape != (n,) for v in (times, events, predicted_risk)):
        raise ValueError('All arrays must contain the same patients in the same order')
    if not np.isfinite(times).all() or (times < 0).any() or not np.isin(events, [0, 1, 2]).all():
        raise ValueError('Need finite nonnegative follow-up and events coded 0/1/2')
    if not np.isfinite(predicted_risk).all() or ((predicted_risk < 0) | (predicted_risk > 1)).any():
        raise ValueError('predicted_risk must contain finite probabilities in [0, 1]')
    if (not np.isfinite(horizon) or horizon <= 0 or grid_size < 3
            or min_effective_support <= 0 or not 0 <= min_density_fraction < 1
            or not np.isfinite(risk_max) or not 0 < risk_max <= 1):
        raise ValueError('Invalid horizon, grid size, support, density or colour scale')
    data = horizon_ipcw_data(times, events, horizon)
    used = data.eligible
    if used.sum() < 2:
        raise ValueError('Need at least two evaluable patients at the horizon')
    local = coords[used]
    bandwidth = _default_bandwidth(local) if bandwidth is None else float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('bandwidth must be positive and finite')
    xx, yy, observed, support, density = _kernel_surfaces(
        local, data.target[used], data.weights[used], bandwidth=bandwidth,
        grid_size=grid_size, bounds_coords=coords)
    geometry, _ = patient_footprint_margin(coords, np.column_stack([xx.ravel(), yy.ravel()]))
    blank = ((support < min_effective_support)
             | (density < min_density_fraction * density.max())
             | (geometry.reshape(xx.shape) < 0))
    calibration = spatial_calibration_residual(
        local, predicted_risk[used], data.target[used], data.weights[used],
        bandwidth=bandwidth, grid=(xx, yy), support_mask=blank,
        min_effective_support=min_effective_support, standardize=False, n_bootstrap=0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout='constrained')
    axes = axes.ravel()
    colours = {'ineligible/early censor': '#D0D0D0', 'known event-free': '#71869E',
               'competing death': '#9C83B5', 'primary event': '#B65B63'}
    for status, colour in colours.items():
        take = data.status == status
        if take.any():
            axes[0].scatter(*coords[take].T, s=6, color=colour, label=status, rasterized=True)
    axes[0].legend(fontsize=7, loc='best', markerscale=2)
    point = axes[1].scatter(*coords.T, c=predicted_risk, s=6, cmap='viridis',
                            vmin=0, vmax=risk_max, rasterized=True)
    fig.colorbar(point, ax=axes[1], label='Predicted probability', extend='max' if (predicted_risk > risk_max).any() else 'neither')
    for ax, values in ((axes[2], observed), (axes[3], calibration.predicted)):
        image = ax.pcolormesh(xx, yy, np.ma.masked_where(blank, values),
                              cmap='viridis', vmin=0, vmax=risk_max, shading='auto', rasterized=True)
        fig.colorbar(image, ax=ax, label='Probability', extend='max' if np.any(values[~blank] > risk_max) else 'neither')
    bound = max(float(np.nanmax(np.abs(calibration.residual[~blank]))) if (~blank).any() else .01, .01)
    image = axes[4].pcolormesh(xx, yy, calibration.display, cmap='RdBu_r',
                               vmin=-bound, vmax=bound, shading='auto', rasterized=True)
    fig.colorbar(image, ax=axes[4], label='Observed − predicted probability')
    image = axes[5].pcolormesh(xx, yy, np.ma.masked_where(blank, support),
                               cmap='plasma', shading='auto', rasterized=True)
    fig.colorbar(image, ax=axes[5], label='Effective sample size (Kish ESS)')
    titles = ['Observed outcome status', 'Individual predicted risk', 'Kernel observed risk (IPCW)',
              'Kernel predicted risk (same weights)', 'Local calibration: observed − predicted',
              'Local support']
    for letter, ax, panel_title in zip('ABCDEF', axes, titles):
        ax.set_title(f'{letter}  {panel_title}', loc='left', fontsize=10)
        ax.set(xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()), aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f'{title} — horizon {horizon:g} days', fontsize=14)
    grids = dict(x=xx, y=yy, observed=observed, predicted=calibration.predicted,
                 residual=calibration.residual, support=support, mask=blank,
                 bandwidth=bandwidth, n_total=n, n_eligible=int(used.sum()))
    return fig, grids



# =============================================================
# BONSAI inference and CLI integration
# =============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create BONSAI held-out UMAP plots for predicted risk and observed outcome.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained FINETUNED best.ckpt (not pretrained checkpoint)")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="BONSAI processed dataset root containing subject_data_held_out.pt, vocabulary.pt, cohorts/ and outcomes/")
    parser.add_argument("--outcomes", type=Path, default=None,
                        help="Outcome parquet; automatic when unique or identified in training config")
    parser.add_argument("--finetune-config", type=Path, default=None,
                        help="Original Hydra resolved or un-resolved config.yaml (auto-discovered next to checkpoint)")
    parser.add_argument("--output-dir", type=Path, default=Path("umap_results"))
    parser.add_argument("--max-len", type=int, default=None,
                        help="Override finetuning training.max_len if original Hydra config is missing")
    parser.add_argument("--label-start-hours", type=int, default=None,
                        help="Override training labels.start_include as hours")
    parser.add_argument("--label-end-days", type=int, default=None,
                        help="Override training labels.end_include as days (else use config)")
    parser.add_argument("--horizon-days", type=float, default=None,
                        help="Required for IPCW plots if training target has a fixed horizon, otherwise inferred from config")
    parser.add_argument("--followup-end-column", default="followup_end_date",
                        help="Genuine outcome follow-up end column, NOT BONSAI censor_date")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--attn-type", choices=["flash", "sdpa"], default=None,
                        help="Optional checkpoint attention backend override; validate parity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=70)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--min-effective-support", type=float, default=25.0)
    parser.add_argument("--risk-max", type=float, default=1.0)
    parser.add_argument("--save-patient-table", action="store_true",
                        help="Write an identifiable patient-level CSV (sensitive data)")
    return parser


def get_train_config(checkpoint: Path, override: Path | None):
    """Recover original max_len/label window if Hydra stored config next to ckpt."""
    from omegaconf import OmegaConf
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    if override:
        candidates = [override.expanduser().resolve()]
    else:
        # Hydra config path is 'hydra/config.yaml' for recent BONSAI, '.hydra' for older runs.
        candidates = [p / hidden / "config.yaml"
                      for p in (checkpoint.parent, checkpoint.parent.parent)
                      for hidden in ("hydra", ".hydra")]
    for path in candidates:
        if path.is_file():
            LOG.info("Using original finetune config: %s", path)
            return OmegaConf.load(path), path
    if override:
        raise FileNotFoundError(f"Finetuning config not found: {override}")
    LOG.warning("No original Hydra config found next to checkpoint. Training max_len and labels "
                "CANNOT be recovered from the model weights alone. Use --finetune-config "
                "(recommended), or explicitly set --max-len and label-window options.")
    return None, None


def config_setting(cfg, path, default=None):
    if cfg is None:
        return default
    from omegaconf import OmegaConf
    value = OmegaConf.select(cfg, path, default=default, throw_on_missing=False)
    return value


def resolve_inputs(args, train_cfg):
    root = args.dataset.expanduser().resolve()
    if root.name == "subject_data_held_out.pt":
        root = root.parent
    if not root.is_dir():
        raise FileNotFoundError(f"Processed dataset directory missing: {root}")
    patient_path = root / "subject_data_held_out.pt"
    vocab_path = root / "vocabulary.pt"
    population_path = root / "cohorts" / "population_full.csv"
    for path in (patient_path, vocab_path, population_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required BONSAI processed-data file missing: {path}")
    if args.outcomes is not None:
        outcome_path = args.outcomes.expanduser().resolve()
    else:
        outcome_name = config_setting(train_cfg, "outcome")
        outcome_path = root / "outcomes" / (str(outcome_name) + ".parquet") if outcome_name else None
        if outcome_path is None or not outcome_path.is_file():
            candidates = sorted((root / "outcomes").rglob("*.parquet")) if (root / "outcomes").exists() else []
            if len(candidates) != 1:
                raise ValueError(
                    f"Found {len(candidates)} possible outcome parquet files under {root / 'outcomes'}. "
                    "Specify --outcomes /path/to/the_same_target_used_for_training.parquet"
                )
            outcome_path = candidates[0]
    if not outcome_path.is_file():
        raise FileNotFoundError(f"Outcome file missing: {outcome_path}")
    return root, patient_path, vocab_path, population_path, outcome_path


def get_model_and_module(checkpoint: Path, vocabulary: dict, attention_override: str | None):
    """Instantiate repo's real model and Lightning module, load finetuned weights strictly."""
    import torch
    from bonsai.modules.lightningmodules.FinetuneModule import FinetuneModule
    from bonsai.modules.networks.bonsai_nets import BonsaiFinetune, BonsaiValueFinetune

    # Trusted local checkpoint only. Lightning checkpoints contain Python pickle.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = dict(checkpoint_data.get("hyper_parameters", {}))
    state = checkpoint_data.get("state_dict", {})
    if not hp or not any(k.startswith("model.finetune_head.") for k in state):
        raise ValueError("Not a BONSAI FINETUNED checkpoint (missing model.finetune_head weights)")
    if "[CLS]" not in vocabulary:
        raise ValueError("vocabulary.pt does not contain a [CLS] token")
    if int(vocabulary["[CLS]"]) != int(hp.get("predict_token_id", -1)):
        raise ValueError("Vocabulary's [CLS] token ID differs from checkpoint; wrong processed dataset?")
    if len(vocabulary) != int(hp.get("vocab_size", -1)):
        raise ValueError("Vocabulary size differs from checkpoint; wrong processed dataset?")
    is_value = hp.get("value_embedding_mode") is not None
    model_type = BonsaiValueFinetune if is_value else BonsaiFinetune
    attn_type = attention_override or hp.get("attn_type")
    if attention_override and attention_override != hp.get("attn_type"):
        LOG.warning("Changing attention backend %s -> %s; validate numerical parity", hp.get("attn_type"), attn_type)
    model_args = {
        "vocab_size": len(vocabulary),
        "max_seqlen": int(hp["max_seqlen"]),
        "hidden_size": int(hp["hidden_size"]),
        "num_layers": int(hp["num_layers"]),
        "bias": bool(hp["bias"]),
        "dropout": float(hp["dropout"]),
        # BONSAI's checkpoint hparams omit attention_dropout; eval mode disables it.
        "attention_dropout": 0.0,
        "causal": bool(hp["causal"]),
        "attn_type": attn_type,
        "predict_token_id": int(hp["predict_token_id"]),
    }
    if is_value:
        model_args["value_embedding_mode"] = hp["value_embedding_mode"]
    model = model_type(**model_args)
    # Preserve BCEWithLogitsLoss.pos_weight buffers for strict Lightning state loading.
    # Checkpoint hyperparameters serialize this as a scalar even though init expects Tensor.
    saved_pos_weight = hp.get("pos_weight")
    pos_weight = None if saved_pos_weight is None else torch.tensor(float(saved_pos_weight))
    module = FinetuneModule.load_from_checkpoint(
        str(checkpoint), model=model, pos_weight=pos_weight,
        predictions_output_path=None, strict=True, map_location="cpu"
    )
    module.eval()
    return module, hp


def make_predict_loader(patient_path: Path, population_path: Path, outcome_path: Path,
                        vocabulary: dict, max_len: int,
                        start_include: dict, end_include: dict | None,
                        batch_size: int, num_workers: int):
    """Call BONSAI's own binarization and FinetuneDataModule.predict_dataloader."""
    import polars as pl
    from bonsai.functional.outcomes import binarize_outcomes, finalize_outcomes
    from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule

    df = pl.read_parquet(outcome_path)
    required = {"split", "subject_id", "index_date", "outcome_date", "censor_date"}
    if required - set(df.columns):
        raise ValueError(f"Outcome parquet missing BONSAI columns: {sorted(required - set(df.columns))}")
    held_out = df.filter(pl.col("split") == "held_out")
    if held_out.is_empty():
        raise ValueError("No outcomes with split='held_out'")
    if held_out["subject_id"].n_unique() != held_out.height:
        raise ValueError("Multiple outcome rows for the same held-out subject_id")
    # Exact BONSAI label function and filtering; do NOT silently relabel all missing events zero.
    filtered = binarize_outcomes(held_out, start_include=start_include, end_include=end_include)
    outcome_lookup = finalize_outcomes(filtered)
    if not outcome_lookup:
        raise ValueError("No usable held-out patients after BONSAI label window filtering")
    # Training and val paths are not read when stage='predict'.
    dm = FinetuneDataModule(
        path_train_data=str(patient_path), path_val_data=str(patient_path),
        path_predict_data=str(patient_path), path_population=str(population_path),
        batch_size=batch_size, num_workers=num_workers,
        predict_token_id=int(vocabulary["[CLS]"]), max_len=max_len,
        train_outcomes={}, val_outcomes={}, predict_outcomes=outcome_lookup,
    )
    dm.setup("predict")
    if len(dm.predict_dataset) < 4:
        raise ValueError("Need at least four held-out patients for a UMAP; got " + str(len(dm.predict_dataset)))
    # A final alignment lookup: keep only eligible outcomes and preserve the original
    # outcomes' dates for optional IPCW, but never rely on row order for matching.
    records = {int(row["subject_id"]): row for row in filtered.to_dicts()}
    return dm.predict_dataloader(), records, len(dm.predict_dataset)


def infer_in_one_pass(module, dataloader, device):
    """BONSAI.predict_step handles sigmoid/labels; pre-hook captures its head input."""
    import torch
    model = module.model
    if not hasattr(model, "finetune_head"):
        raise TypeError("Model lacks finetune_head: cannot extract prediction-token embeddings")
    module.to(device)
    module.eval()
    capture = []

    def record_embedding(head, inputs):
        if len(inputs) != 1 or inputs[0].ndim != 2:
            raise RuntimeError("Expected [batch, hidden_dim] input to finetune_head")
        capture.append(inputs[0].detach().float().cpu())

    hook = model.finetune_head.register_forward_pre_hook(record_embedding)
    ids, embeddings, probs, labels = [], [], [], []
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(dataloader):
                moved = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
                capture.clear()
                prediction = module.predict_step(moved, batch_index)
                batch_ids = prediction["subject_id"].detach().cpu().reshape(-1).numpy()
                scores = prediction["prob"].detach().float().cpu().reshape(-1).numpy()
                outcomes = prediction["label"].detach().cpu().reshape(-1).numpy().astype(int)
                if len(capture) != 1 or capture[0].shape[0] != len(batch_ids):
                    raise RuntimeError("Prediction-head embeddings do not match this patient's batch")
                if len(scores) != len(batch_ids) or len(outcomes) != len(batch_ids):
                    raise RuntimeError("Batch predictions, labels and subject IDs do not align")
                ids.extend(batch_ids.tolist())
                embeddings.append(capture[0].numpy())
                probs.append(scores)
                labels.append(outcomes)
    finally:
        hook.remove()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate held-out subject IDs in prediction dataloader")
    return (np.asarray(ids, dtype=np.int64), np.concatenate(embeddings),
            np.concatenate(probs), np.concatenate(labels))


def parse_datetime(obj):
    if obj is None:
        return None
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).replace(tzinfo=None) if obj.tzinfo else obj
    if isinstance(obj, date):
        return datetime.combine(obj, time.min)
    if isinstance(obj, str):
        parsed = datetime.fromisoformat(obj.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
    raise TypeError(f"Unexpected date/datetime representation {type(obj)}")


def followup_arrays(ordered_records, column, horizon, model_labels):
    """IPCW available only with a true end-of-follow-up and a matching horizon."""
    if horizon is None:
        return None, None
    if not ordered_records or any(record.get(column) is None for record in ordered_records):
        LOG.warning("No complete genuine %s column; plotting recorded outcomes without IPCW", column)
        return None, None
    times, events = [], []
    for record in ordered_records:
        index = parse_datetime(record["index_date"])
        event_time = parse_datetime(record["outcome_date"])
        followup_end = parse_datetime(record[column])
        if index is None or followup_end is None:
            raise ValueError("Missing index_date/followup end date")
        duration = (followup_end - index).total_seconds() / 86400
        if duration < 0:
            raise ValueError("Follow-up ends before prediction starts")
        if event_time is not None:
            event_day = (event_time - index).total_seconds() / 86400
            if event_day < 0:
                raise ValueError("Outcome preceding index_date in held-out set")
        else:
            event_day = None
        if event_day is not None and event_day <= duration:
            times.append(event_day)
            events.append(1)
        else:
            times.append(duration)
            events.append(0)
    times, events = np.asarray(times), np.asarray(events)
    valid = horizon_ipcw_data(times, events, horizon)
    contradictory = valid.eligible & (valid.target.astype(int) != model_labels)
    if contradictory.any():
        raise ValueError(f"{contradictory.sum()} model target labels conflict with follow-up events "
                         "at this horizon. Check training label window / dates.")
    return times, events


def calculate_surfaces(coords, y, predictions, times, events, horizon,
                       bandwidth, grid_size, minimum_support):
    if times is None:
        keep = np.ones(len(coords), dtype=bool)
        targets = np.asarray(y, dtype=float)
        weights = np.ones(len(coords), dtype=float)
        status = np.where(y > 0, "recorded outcome", "no recorded outcome")
    else:
        ipcw = horizon_ipcw_data(times, events, horizon)
        keep, targets, weights, status = (ipcw.eligible, ipcw.target, ipcw.weights, ipcw.status)
    if keep.sum() < 3:
        raise ValueError("Not enough evaluable patients for smoothed plots")
    scale = _default_bandwidth(coords[keep]) if bandwidth is None else float(bandwidth)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Bandwidth must be positive")
    xx, yy, obs, support, density = _kernel_surfaces(
        coords[keep], targets[keep], weights[keep], bandwidth=scale,
        grid_size=grid_size, bounds_coords=coords)
    calibration = spatial_calibration_residual(
        coords[keep], predictions[keep], targets[keep], weights[keep],
        bandwidth=scale, grid=(xx, yy), min_effective_support=minimum_support,
        standardize=False, n_bootstrap=0)
    margin, _ = patient_footprint_margin(coords, np.column_stack([xx.ravel(), yy.ravel()]))
    mask = ((support < minimum_support) | (density < 0.01 * np.nanmax(density))
            | (margin.reshape(xx.shape) < 0))
    if mask.all():
        LOG.warning("All smoothed cells are masked; try a smaller --min-effective-support")
    return dict(x=xx, y=yy, predicted=calibration.predicted, observed=obs,
                mask=mask, support=support, eligible=keep, status=status,
                bandwidth=scale)


def write_plot(coords, points, surfaces, *, kind, ipcw, risk_max, output):
    from matplotlib.colors import Normalize
    xx, yy = surfaces["x"], surfaces["y"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
    if kind == "risk":
        artist = axes[0].scatter(coords[:, 0], coords[:, 1], c=points,
                                 s=8, cmap="viridis", vmin=0, vmax=risk_max,
                                 rasterized=True)
        fig.colorbar(artist, ax=axes[0], label="BONSAI predicted probability")
        axes[0].set_title("Patients colored by model-predicted risk")
        surface, title = surfaces["predicted"], "Smoothed predicted model probability"
    else:
        colors = ({"ineligible/early censor": "#d0d0d0", "known event-free": "#71869e",
                   "competing death": "#9c83b5", "primary event": "#b65b63"} if ipcw else
                  {"no recorded outcome": "#71869e", "recorded outcome": "#b65b63"})
        for key, color in colors.items():
            selected = surfaces["status"] == key
            if np.any(selected):
                axes[0].scatter(coords[selected, 0], coords[selected, 1],
                                s=8, color=color, label=key, rasterized=True)
        axes[0].legend(fontsize=8, markerscale=2)
        axes[0].set_title("Patients colored by observed outcome")
        surface = surfaces["observed"]
        title = ("Local IPCW outcome risk" if ipcw else
                 "Smoothed fraction with recorded outcome (not IPCW)")
    image = axes[1].pcolormesh(xx, yy, np.ma.masked_where(surfaces["mask"], surface),
                                cmap="viridis", norm=Normalize(0, risk_max),
                                shading="auto", rasterized=True)
    fig.colorbar(image, ax=axes[1], label="Local probability / fraction")
    axes[1].scatter(coords[:, 0], coords[:, 1], s=1.2, alpha=0.15, color="black",
                    rasterized=True)
    axes[1].set_title(title)
    for ax in axes:
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2", aspect="equal")
        ax.set(xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args):
    import torch
    import polars as pl
    from umap import UMAP

    if args.batch_size <= 0 or args.num_workers < 0 or args.umap_neighbors < 2:
        raise ValueError("Invalid batch size, workers or UMAP n_neighbors")
    if args.grid_size < 3 or args.min_effective_support <= 0 or not (0 < args.risk_max <= 1):
        raise ValueError("Invalid grid size, support or risk color range")
    if args.umap_min_dist < 0:
        raise ValueError("UMAP min-dist cannot be negative")
    ckpt = args.checkpoint.expanduser().resolve()
    config, config_path = get_train_config(ckpt, args.finetune_config)
    root, patient_path, vocab_path, population_path, outcome_path = resolve_inputs(args, config)
    LOG.info("Dataset: %s | outcomes: %s", root, outcome_path)
    # BONSAI training .pt artifacts are trusted pickle; load only own files.
    vocabulary = torch.load(vocab_path, map_location="cpu", weights_only=False)
    module, hp = get_model_and_module(ckpt, vocabulary, args.attn_type)
    config_max_len = config_setting(config, "training.max_len")
    max_len = args.max_len if args.max_len is not None else config_max_len
    if max_len is None:
        max_len = int(hp["max_seqlen"])
        LOG.warning("Using checkpoint max_seqlen=%s, which may not equal finetune training.max_len", max_len)
    if max_len > int(hp["max_seqlen"]) or max_len < 2:
        raise ValueError("Finetune max_len must be >=2 and <= checkpoint max_seqlen")

    start_include = config_setting(config, "labels.start_include", {"hours": 0})
    end_include = config_setting(config, "labels.end_include", None)
    from omegaconf import OmegaConf
    if OmegaConf.is_config(start_include):
        start_include = OmegaConf.to_container(start_include, resolve=True)
    if OmegaConf.is_config(end_include):
        end_include = OmegaConf.to_container(end_include, resolve=True)
    if args.label_start_hours is not None:
        start_include = {"hours": args.label_start_hours}
    if args.label_end_days is not None:
        end_include = {"days": args.label_end_days}
    # A fixed horizon is only established if it matches training labels.end_include.
    from datetime import timedelta
    cfg_horizon = timedelta(**end_include).total_seconds() / 86400 if end_include is not None else None
    horizon = args.horizon_days if args.horizon_days is not None else cfg_horizon
    if horizon is not None and (not np.isfinite(horizon) or horizon <= 0):
        raise ValueError("Horizon must be positive and finite")
    fixed_horizon_agrees = (cfg_horizon is not None and horizon is not None
                            and abs(horizon - cfg_horizon) < 1e-6)
    if args.horizon_days is not None and not fixed_horizon_agrees:
        LOG.warning("Requested --horizon-days=%s does not match finetune labels.end_include=%s. "
                    "Using BONSAI model labels for plots; IPCW calibration will NOT run.",
                    horizon, cfg_horizon)
    if cfg_horizon is None:
        LOG.warning("No fixed outcome horizon in finetuning labels. The sigmoid is not established "
                    "as a fixed-horizon risk estimate; plots show training-defined outcome labels.")

    loader, records, expected = make_predict_loader(
        patient_path, population_path, outcome_path, vocabulary, int(max_len),
        start_include, end_include, args.batch_size, args.num_workers)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ids, embeddings, probabilities, labels = infer_in_one_pass(module, loader, device)
    if len(ids) != expected or any(int(sid) not in records for sid in ids):
        raise RuntimeError("Inference did not match the filtered held-out cohort")
    if not (np.isfinite(embeddings).all() and np.isfinite(probabilities).all()):
        raise ValueError("Model produced NaN/inf values")
    ordered = [records[int(i)] for i in ids]
    times, events = (followup_arrays(ordered, args.followup_end_column, horizon, labels)
                     if fixed_horizon_agrees else (None, None))
    LOG.info("Embeddings: %d patients, %d dimensions; IPCW=%s", len(ids), embeddings.shape[1], times is not None)
    reducer = UMAP(n_components=2, n_neighbors=min(args.umap_neighbors, len(ids) - 1),
                   min_dist=args.umap_min_dist, random_state=args.seed,
                   transform_seed=args.seed)
    xy = reducer.fit_transform(embeddings)
    surfaces = calculate_surfaces(xy, labels, probabilities, times, events, horizon,
                                  args.bandwidth, args.grid_size, args.min_effective_support)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plot(xy, probabilities, surfaces, kind="risk", ipcw=times is not None,
               risk_max=args.risk_max, output=args.output_dir / "umap_predicted_risk.png")
    write_plot(xy, labels, surfaces, kind="outcome", ipcw=times is not None,
               risk_max=args.risk_max, output=args.output_dir / "umap_observed_outcome.png")
    if times is not None:
        figure, _ = plot_risk_maps(
            xy, times, events, probabilities, horizon=horizon,
            bandwidth=args.bandwidth, grid_size=args.grid_size,
            min_effective_support=args.min_effective_support,
            risk_max=args.risk_max, title="BONSAI held-out endpoint")
        figure.savefig(args.output_dir / "umap_six_panel_calibration.png", dpi=200)
        plt.close(figure)
    if args.save_patient_table:
        pl.DataFrame({"subject_id": ids, "umap_1": xy[:, 0], "umap_2": xy[:, 1],
                      "predicted_probability": probabilities,
                      "observed_outcome_label": labels}).write_csv(
                          args.output_dir / "patients_sensitive.csv")
    summary = {
        "checkpoint": str(ckpt), "dataset": str(root), "outcomes": str(outcome_path),
        "original_finetune_config": str(config_path) if config_path else None,
        "n_held_out": len(ids), "embedding_dim": int(embeddings.shape[1]),
        "n_positive_labels": int(labels.sum()), "max_len": int(max_len),
        "label_start_include": start_include, "label_end_include": end_include,
        "fixed_horizon_days": cfg_horizon, "ipcw_adjusted": times is not None,
        "bandwidth": float(surfaces["bandwidth"]),
        "notes": ("IPCW needs genuine end-of-follow-up and independent censoring. "
                  "Without follow-up end dates, 0 is no recorded outcome, not confirmed event-free. "
                  "If original Hydra config is absent, label window and max_len may differ from training."),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOG.info("Saved plots in %s", args.output_dir.resolve())
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
