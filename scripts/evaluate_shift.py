#!/usr/bin/env python3
import json, os, sys, csv
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "neural-lam-prob-model"))
sys.path.insert(0, REPO_ROOT)

from neural_lam import constants, metrics as nl_metrics
from neural_lam.era5_dataset import ERA5Dataset
from neural_lam.models.graph_efm import GraphEFM


@dataclass
class EvalResult:
    split: str
    calibration: str
    rmse: dict = field(default_factory=dict)
    bias: dict = field(default_factory=dict)
    crps: dict = field(default_factory=dict)
    spread_skill: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    rank_hist: dict = field(default_factory=dict)
    pred: np.ndarray | None = None
    target: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model_args(cfg: dict):
    """Build an argparse-like namespace for model loading."""
    mc = cfg["model"]
    fc = cfg["forecast"]
    gc = cfg["graph"]
    ev = cfg["evaluation"]

    class Args:
        pass
    a = Args()
    a.hidden_dim = mc["hidden_dim"]
    a.latent_dim = mc["latent_dim"]
    a.hidden_layers = mc["hidden_layers"]
    a.processor_layers = mc["decoder_processor_layers"]
    a.encoder_processor_layers = mc["encoder_processor_layers"]
    a.prior_processor_layers = mc["prior_processor_layers"]
    a.kl_beta = mc["kl_beta"]
    a.crps_weight = mc["crps_weight"]
    a.loss = mc["loss"]
    a.sample_obs_noise = mc["sample_obs_noise"]
    a.output_std = mc["output_std"]
    a.prior_dist = mc["prior_dist"]
    a.learn_prior = mc["learn_prior"]
    a.graph = gc["name"]
    a.dataset = cfg["dataset"]["name"]
    a.lr = cfg["training"]["phase_a"]["lr"]
    a.step_length = cfg["sampling"]["step_length"]
    a.eval_leads = fc["eval_leads"]
    a.ensemble_size = ev["ensemble_size"]
    a.n_example_pred = 0
    a.batch_size = 1
    return a


def dataset_base_times(dataset: ERA5Dataset) -> np.ndarray:
    """
    Base init time (np.datetime64) for each dataset index.

    For non-train splits the dataset initializes from the two consecutive
    6-hourly states ending at ``time_coords[1 + 2 * idx]``, so the base
    (init) time of dataset index ``idx`` is ``time_coords[1 + 2 * idx]``.
    """
    times = dataset.time_coords
    return np.array([times[1 + 2 * idx] for idx in range(len(dataset))])


def select_eval_indices(
    dataset: ERA5Dataset,
    seed: int = 42,
    dates_per_month: int = 2,
) -> list:
    """
    Deterministic subset of dataset indices to evaluate.

    Samples ``dates_per_month`` distinct days per calendar month (seeded),
    keeping every init time the dataset offers on those days. This derives
    the init times directly from the dataset's own time coordinate, so it
    stays consistent regardless of which hours the split starts on.
    """
    base_times = dataset_base_times(dataset)
    day_strs = np.datetime_as_string(base_times, unit="D")  # "YYYY-MM-DD"
    by_day = defaultdict(list)
    for idx, day_str in enumerate(day_strs):
        by_day[day_str].append(idx)

    rng = np.random.RandomState(seed)
    selected = []
    for month_key in sorted({s[:7] for s in by_day}):  # "YYYY-MM"
        days = sorted(s for s in by_day if s.startswith(month_key))
        n = min(dates_per_month, len(days))
        chosen = sorted(rng.choice(np.array(days), size=n, replace=False))
        for day in chosen:
            selected.extend(sorted(by_day[day]))
    return selected


def run_ensemble_forecast(
    model: GraphEFM,
    dataset: ERA5Dataset,
    indices: list,
    ensemble_size: int,
    pred_length: int,
    device: torch.device,
    mask: torch.Tensor | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run ensemble forecasts for the given dataset indices.

    Returns:
        predictions: (n_samples, ensemble_size, pred_steps, N_grid, d_state)
        targets: (n_samples, pred_steps, N_grid, d_state)
    """
    model.eval()
    all_preds = []
    all_targets = []

    for idx in tqdm(indices, desc="Forecasting"):
        init_states, target_states, forcing = dataset[idx]
        # init_states: (2, N_grid, d_state)
        # target_states: (pred_steps, N_grid, d_state)
        # forcing: (pred_steps, N_grid, forcing_dim)

        # Add batch dim
        init_states = init_states.unsqueeze(0).to(device)
        target_states_b = target_states.unsqueeze(0).to(device)
        forcing = forcing.unsqueeze(0).to(device)

        with torch.no_grad():
            traj_means, _ = model.sample_trajectories(
                init_states,
                forcing,
                target_states_b,
                ensemble_size,
                use_encoder=False,
            )
            # traj_means: (1, ensemble_size, pred_steps, N_grid, d_state)

        all_preds.append(traj_means.squeeze(0).cpu().numpy())
        all_targets.append(target_states.cpu().numpy())

    if not all_preds:
        raise RuntimeError("No valid init times to forecast.")

    preds = np.stack(all_preds, axis=0)
    targets = np.stack(all_targets, axis=0)
    return preds, targets


def compute_all_metrics(
    pred: torch.Tensor,         # (S, M, T, N, D)
    target: torch.Tensor,       # (S, T, N, D)
    grid_weights: torch.Tensor,  # (N,)
    var_names: list,
    lead_hours: list,
    mask: torch.Tensor | None = None,
) -> dict:
    """Compute per-variable, per-lead metrics from ensemble predictions."""
    results = defaultdict(lambda: defaultdict(list))

    n_samples, n_ens, n_steps, n_grid, d_state = pred.shape

    for lead_idx in range(n_steps):
        lead_h = lead_hours[lead_idx]
        p_lead = pred[:, :, lead_idx]   # (S, M, N, D)
        t_lead = target[:, lead_idx]    # (S, N, D)

        for var_idx, vname in enumerate(var_names):
            p_v = p_lead[..., var_idx:var_idx+1]  # (S, M, N, 1)
            t_v = t_lead[..., var_idx:var_idx+1]  # (S, N, 1)

            # RMSE (ensemble mean)
            ens_mean = p_v.mean(dim=1)  # (S, N, 1)
            mse_val = nl_metrics.mse(
                ens_mean, t_v, None,
                grid_weights=grid_weights, mask=mask,
                average_grid=True, sum_vars=True,
            )  # (S,)
            rmse = torch.sqrt(mse_val)  # (S,)
            results["rmse"][vname][lead_h] = rmse.cpu().numpy()

            # Bias (ensemble mean)
            bias_val = nl_metrics.ens_mean_bias(
                p_v, t_v,
                grid_weights=grid_weights, mask=mask,
                average_grid=True, sum_vars=True,
            )  # (S,)
            results["bias"][vname][lead_h] = bias_val.cpu().numpy()

            # CRPS
            crps_val = nl_metrics.crps_ens(
                p_v, t_v, None,
                grid_weights=grid_weights, mask=mask,
                average_grid=True, sum_vars=True,
            )  # (S,)
            results["crps"][vname][lead_h] = crps_val.cpu().numpy()

            # Spread-skill ratio
            ssr = nl_metrics.spread_skill_ratio(
                p_v, t_v,
                grid_weights=grid_weights, mask=mask,
            )  # (S, 1)
            results["spread_skill"][vname][lead_h] = ssr.squeeze(-1).cpu().numpy()

            # Interval coverage
            for pct, label in [(0.5, "50"), (0.8, "80"), (0.9, "90")]:
                cov = nl_metrics.interval_coverage(
                    p_v, t_v, central_fraction=pct,
                )  # (S, 1)
                results["coverage"][label][vname][lead_h] = (
                    cov.squeeze(-1).cpu().numpy()
                )

            # Rank histogram (aggregated, not per-sample)
            rh = nl_metrics.rank_histogram(p_v, t_v, n_bins=n_ens)
            # (1, n_bins+1)
            results["rank_hist"][vname][lead_h] = rh.squeeze(0).cpu().numpy()

    return results


def fit_calibration(
    pred_val: np.ndarray,       # (S, M, T, N, D)
    target_val: np.ndarray,     # (S, T, N, D)
    var_names: list,
    lead_hours: list,
    grid_weights: torch.Tensor,
    search_grid: list,
    mask: torch.Tensor | None = None,
) -> dict:
    """
    Fit post-hoc spread scaling multipliers on validation data.
    Returns {var_name: {lead_h: best_alpha}}.
    """
    multipliers = {}
    p_t = torch.tensor(pred_val, dtype=torch.float32)
    t_t = torch.tensor(target_val, dtype=torch.float32)

    for var_idx, vname in enumerate(var_names):
        multipliers[vname] = {}
        for lead_idx, lead_h in enumerate(lead_hours):
            p_v = p_t[:, :, lead_idx, ..., var_idx:var_idx+1]
            t_v = t_t[:, lead_idx, ..., var_idx:var_idx+1]

            best_alpha = 1.0
            best_crps = float("inf")

            for alpha in search_grid:
                # Calibrate: mean + alpha * (member - mean)
                p_cal = nl_metrics.apply_spread_scaling(
                    p_v, torch.tensor([alpha], dtype=torch.float32)
                )
                crps = nl_metrics.crps_ens(
                    p_cal, t_v, None,
                    grid_weights=grid_weights, mask=mask,
                    average_grid=True, sum_vars=True,
                )
                mean_crps = crps.mean().item()
                if mean_crps < best_crps:
                    best_crps = mean_crps
                    best_alpha = alpha

            multipliers[vname][lead_h] = best_alpha

    return multipliers


def apply_calibration(
    pred: np.ndarray,
    multipliers: dict,
    var_names: list,
    lead_hours: list,
) -> np.ndarray:
    """Apply fitted multipliers to ensemble predictions."""
    p_t = torch.tensor(pred, dtype=torch.float32)
    calibrated = p_t.clone()

    for var_idx, vname in enumerate(var_names):
        for lead_idx, lead_h in enumerate(lead_hours):
            alpha = multipliers[vname][lead_h]
            p_v = p_t[:, :, lead_idx, ..., var_idx:var_idx+1]
            alpha_t = torch.tensor([alpha], dtype=torch.float32)
            calibrated[:, :, lead_idx, ..., var_idx] = (
                nl_metrics.apply_spread_scaling(p_v, alpha_t).squeeze(-1)
            )

    return calibrated.cpu().numpy()


def bootstrap_diff(
    vals_a: np.ndarray,   # (S,)
    vals_b: np.ndarray,   # (S,)
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(vals_b - vals_a)."""
    diff = vals_b - vals_a
    rng = np.random.RandomState(seed)
    n = len(diff)
    means = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        means.append(diff[idx].mean())
    means = np.array(means)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


def compute_t2m_anomalies(
    cfg: dict, init_times: np.ndarray,
) -> dict:
    """
    Compute latitude-weighted t2m anomalies for each init time,
    relative to training-period monthly climatology.
    """
    static_dir = os.path.join(REPO_ROOT, cfg["dataset"]["static_dir"])
    clim_path = os.path.join(static_dir, "monthly_climatology.npz")
    if not os.path.exists(clim_path):
        print(f"  Warning: no climatology at {clim_path}, skipping anomalies")
        return {}

    clim = np.load(clim_path)
    if "2m_temperature" not in clim:
        print("  Warning: no 2m_temperature in climatology")
        return {}

    t2m_clim = clim["2m_temperature"]  # (12, lon, lat)
    lat_weights = np.cos(
        np.deg2rad(np.linspace(90, -90, t2m_clim.shape[-1]))
    )
    lat_weights = lat_weights / lat_weights.mean()

    # Open fields to read actual t2m at init times
    fields_path = os.path.join(REPO_ROOT, cfg["dataset"]["fields_zarr"])
    import xarray as xa
    ds = xa.open_zarr(fields_path)

    anomalies = []
    for t in init_times:
        t_str = np.datetime_as_string(t, unit="m") + ":00"
        try:
            t2m_val = ds["2m_temperature"].sel(
                time=t_str, method="nearest"
            ).values  # (lon, lat)
        except Exception:
            continue
        month = int(np.datetime_as_string(t, unit="M")[5:7])
        clim_val = t2m_clim[month - 1]  # (lon, lat)
        anom = t2m_val - clim_val
        # Latitude-weighted mean
        weighted_anom = np.average(anom, axis=-1, weights=lat_weights).mean()
        anomalies.append(weighted_anom)

    return {
        "mean": float(np.mean(anomalies)),
        "std": float(np.std(anomalies)),
        "values": [float(a) for a in anomalies],
    }


def compute_persistence_rmse(
    targets: np.ndarray,  # (S, T, N, D)
    init_states_for_persistence: np.ndarray,  # (S, N, D) — latest init
    grid_weights: torch.Tensor,
    var_names: list,
    lead_hours: list,
    mask: torch.Tensor | None = None,
) -> dict:
    """Compute RMSE for persistence baseline at each lead."""
    results = defaultdict(dict)
    for lead_idx, lead_h in enumerate(lead_hours):
        # Persistence: just repeat init state
        for var_idx, vname in enumerate(var_names):
            # Broadcast init_state to target shape
            pred_v = torch.tensor(
                init_states_for_persistence[:, var_idx:var_idx+1],
                dtype=torch.float32,
            )  # (S, N, 1)
            # Expand to match target shape
            targ_v = torch.tensor(
                targets[:, lead_idx, ..., var_idx:var_idx+1],
                dtype=torch.float32,
            )

            # Expand pred to match
            pred_v = pred_v.unsqueeze(1)  # (S, 1, N, 1)

            mse_v = nl_metrics.mse(
                pred_v.squeeze(1),
                targ_v,
                None,
                grid_weights=grid_weights,
                mask=mask,
                average_grid=True,
                sum_vars=True,
            )
            rmse = torch.sqrt(mse_v).cpu().numpy()
            results[vname][lead_h] = rmse

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(
        description="Evaluate Graph-EFM ensemble for temporal-shift experiment"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--ensemble_size", type=int, default=8,
        help="Number of ensemble members (default: 8)",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for metrics and calibration (default: results/)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (default: auto)",
    )
    args = parser.parse_args()

    # Load config and initialize constants
    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    constants.load_experiment_config(config_path)

    os.makedirs(args.output, exist_ok=True)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Build model args and load checkpoint
    model_args = build_model_args(cfg)
    model_args.ensemble_size = args.ensemble_size
    print(f"Loading checkpoint: {args.checkpoint}")
    model = GraphEFM.load_from_checkpoint(
        args.checkpoint, args=model_args, map_location=device,
    )
    model = model.to(device)
    model.eval()

    # Derived constants
    var_names = [sv["short_name"] for sv in cfg["state_variables"]]
    d_state = len(var_names)
    eval_leads = cfg["forecast"]["eval_leads"]
    lead_hours = [6 * (i + 1) for i in range(eval_leads)]
    dates_per_month = cfg["sampling"]["eval_dates_per_month"]
    seed = cfg["evaluation"]["seed"]
    ensemble_size = args.ensemble_size
    cal_grid = cfg["calibration"]["search_grid"]

    # Grid weights and mask (from model buffers). Fall back to None if the
    # model does not expose them (metrics then degrade to unweighted).
    grid_weights = getattr(model, "grid_weights", None)  # (N,) or None
    mask = getattr(model, "interior_mask_bool", None)  # (N,) or None

    # ---- 1. Build evaluation samples ----
    splits = ["val", "id", "ood"]
    datasets = {}
    for split_key in splits:
        datasets[split_key] = ERA5Dataset(
            cfg["dataset"]["name"], pred_length=eval_leads,
            split=split_key, standardize=True)

    eval_indices = {}
    for split_key in splits:
        eval_indices[split_key] = select_eval_indices(
            datasets[split_key], seed=seed,
            dates_per_month=dates_per_month)
        print(f"{split_key}: {len(eval_indices[split_key])} init times")

    predictions, targets = {}, {}
    for split_key in splits:
        pred_arr, targ_arr = run_ensemble_forecast(
            model, datasets[split_key], eval_indices[split_key],
            ensemble_size, eval_leads, device, mask)
        predictions[split_key] = pred_arr
        targets[split_key] = targ_arr
        print(f"  {split_key} pred: {pred_arr.shape}, target: {targ_arr.shape}")

    all_results = {}
    for split_key in splits:
        pred_t = torch.tensor(predictions[split_key], dtype=torch.float32)
        targ_t = torch.tensor(targets[split_key], dtype=torch.float32)
        metrics = compute_all_metrics(
            pred_t, targ_t, grid_weights, var_names, lead_hours, mask,
        )
        all_results[(split_key, "raw")] = metrics

    print("Fitting calibration on validation set...")
    multipliers = fit_calibration(
        predictions["val"], targets["val"],
        var_names, lead_hours, grid_weights, cal_grid, mask,
    )
    for vname, vmult in multipliers.items():
        best_leads = {str(k): v for k, v in vmult.items()}
        print(f"  {vname}: {best_leads}")

    # Save multipliers
    mult_save = {
        str(vname): {str(lh): float(a) for lh, a in vdict.items()}
        for vname, vdict in multipliers.items()
    }
    with open(os.path.join(args.output, "calibration_multipliers.json"),
              "w") as fh:
        json.dump(mult_save, fh, indent=2)

    # Apply calibration
    for split_key in ["id", "ood"]:
        cal_pred = apply_calibration(
            predictions[split_key], multipliers, var_names, lead_hours)
        pred_t = torch.tensor(cal_pred, dtype=torch.float32)
        targ_t = torch.tensor(targets[split_key], dtype=torch.float32)
        metrics = compute_all_metrics(
            pred_t, targ_t, grid_weights, var_names, lead_hours, mask)
        all_results[(split_key, "calibrated")] = metrics

    # Persistence baseline
    pers_rmse = defaultdict(dict)
    for split_key in ["id", "ood"]:
        dataset = datasets[split_key]
        init_latest, targ_all = [], []
        for idx in eval_indices[split_key]:
            init_states, target_states, _ = dataset[idx]
            init_latest.append(init_states[1].cpu().numpy())
            targ_all.append(target_states.cpu().numpy())
        if init_latest:
            init_arr = np.stack(init_latest, axis=0)
            targ_arr_full = np.stack(targ_all, axis=0)
            pers_metrics = compute_persistence_rmse(
                targ_arr_full, init_arr, grid_weights, var_names, lead_hours, mask)
            for vname in var_names:
                for lh in lead_hours:
                    pers_rmse[split_key][(vname, lh)] = pers_metrics[vname][lh]

    # t2m anomalies
    anomalies = {}
    for split_key in splits:
        base_times = dataset_base_times(datasets[split_key])
        init_times = base_times[eval_indices[split_key]]
        anom = compute_t2m_anomalies(cfg, init_times)
        anomalies[split_key] = anom
        if anom:
            print(f"  {split_key} t2m anomaly: {anom['mean']:.3f} K")

    # Bootstrap CIs
    bootstrap_results = []
    for calib in ["raw", "calibrated"]:
        for metric_name in ["rmse", "crps"]:
            id_metrics = all_results.get(("id", calib), {})
            ood_metrics = all_results.get(("ood", calib), {})
            if metric_name not in id_metrics:
                continue
            for vname in var_names:
                for lh in lead_hours:
                    id_vals = id_metrics[metric_name][vname].get(lh)
                    ood_vals = ood_metrics[metric_name][vname].get(lh)
                    if id_vals is None or ood_vals is None:
                        continue
                    # Make same length
                    min_len = min(len(id_vals), len(ood_vals))
                    est, lo, hi = bootstrap_diff(
                        id_vals[:min_len], ood_vals[:min_len],
                        n_resamples=cfg["evaluation"]["bootstrap_resamples"],
                        seed=seed,
                    )
                    bootstrap_results.append({
                        "metric": f"{calib}_{metric_name}",
                        "variable": vname,
                        "lead_hours": lh,
                        "ood_minus_id_estimate": est,
                        "ci_lower": lo,
                        "ci_upper": hi,
                    })

    # Export
    rows = []
    for (split_key, calib), metrics in all_results.items():
        for metric_name in ["rmse", "bias", "crps", "spread_skill"]:
            if metric_name not in metrics:
                continue
            for vname in var_names:
                for lh in lead_hours:
                    vals = metrics[metric_name][vname].get(lh)
                    if vals is None:
                        continue
                    rows.append({
                        "split": split_key,
                        "calibration": calib,
                        "variable": vname,
                        "lead_hours": lh,
                        "metric": metric_name,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    })

        # Coverage
        if "coverage" in metrics:
            for pct_label in ["50", "80", "90"]:
                if pct_label not in metrics["coverage"]:
                    continue
                for vname in var_names:
                    for lh in lead_hours:
                        vals = metrics["coverage"][pct_label][vname].get(lh)
                        if vals is None:
                            continue
                        rows.append({
                            "split": split_key,
                            "calibration": calib,
                            "variable": vname,
                            "lead_hours": lh,
                            "metric": f"coverage_{pct_label}",
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                        })

    if rows:
        with open(os.path.join(args.output, "metrics.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    if bootstrap_results:
        with open(os.path.join(args.output, "bootstrap_ci.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=bootstrap_results[0].keys())
            writer.writeheader()
            writer.writerows(bootstrap_results)

    anom_export = {k: {"mean": v.get("mean"), "std": v.get("std")}
                   for k, v in anomalies.items()}
    with open(os.path.join(args.output, "t2m_anomalies.json"), "w") as fh:
        json.dump(anom_export, fh, indent=2)

    pers_rows = []
    for split_key in ["id", "ood"]:
        for (vname, lh), rmse_vals in pers_rmse[split_key].items():
            pers_rows.append({"split": split_key, "variable": vname,
                              "lead_hours": lh, "rmse_mean": float(np.mean(rmse_vals))})
    if pers_rows:
        with open(os.path.join(args.output, "persistence_rmse.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=pers_rows[0].keys())
            writer.writeheader()
            writer.writerows(pers_rows)

    for (split_key, calib), metrics in all_results.items():
        if "rank_hist" not in metrics:
            continue
        rh_dict = {}
        for vname in var_names:
            rh_dict[vname] = {str(lh): metrics["rank_hist"][vname].get(lh).tolist()
                              for lh in lead_hours
                              if metrics["rank_hist"][vname].get(lh) is not None}
        with open(os.path.join(args.output, f"rank_hist_{split_key}_{calib}.json"), "w") as fh:
            json.dump(rh_dict, fh, indent=2)

    print("Evaluation done.")


if __name__ == "__main__":
    main()
