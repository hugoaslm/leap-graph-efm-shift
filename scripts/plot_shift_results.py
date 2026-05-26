#!/usr/bin/env python3
"""
Generate figures for the Graph-EFM temporal-shift experiment.

Reads evaluation metrics from results/ and produces the 5 agreed figures:
  1. Split timeline + t2m anomaly comparison
  2. Ensemble-mean RMSE vs lead time (ID vs OOD + persistence)
  3. Raw vs calibrated CRPS vs lead time (ID vs OOD)
  4. Calibrated spread-skill ratio + interval coverage vs lead time
  5. Calibrated t2m rank histograms at 72 h

Usage:
    python scripts/plot_shift_results.py \\
        --config configs/wb2_shift_64x32_graph_efm.yaml \\
        --metrics results/ \\
        --output figures/
"""
from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

COLORS = {
    "raw_id": "#2166ac",
    "raw_ood": "#b2182b",
    "cal_id": "#4393c3",
    "cal_ood": "#d6604d",
    "persistence": "#666666",
}

LINE_STYLES = {
    "raw_id": "-",
    "raw_ood": "-",
    "cal_id": "--",
    "cal_ood": "--",
    "persistence": ":",
}

VARIABLE_LABELS = {
    "z500": "Z500",
    "t850": "T850",
    "t2m": "T2M",
}

VARIABLE_UNITS = {
    "z500": "m²/s²",
    "t850": "K",
    "t2m": "K",
}


def load_metrics(metrics_dir: str) -> list[dict]:
    """Load metrics.csv as list of dicts."""
    import csv
    path = os.path.join(metrics_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return []
    with open(path, "r") as fh:
        return list(csv.DictReader(fh))


def load_bootstrap(metrics_dir: str) -> list[dict]:
    """Load bootstrap_ci.csv."""
    import csv
    path = os.path.join(metrics_dir, "bootstrap_ci.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r") as fh:
        return list(csv.DictReader(fh))


def load_persistence(metrics_dir: str) -> list[dict]:
    """Load persistence_rmse.csv."""
    import csv
    path = os.path.join(metrics_dir, "persistence_rmse.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r") as fh:
        return list(csv.DictReader(fh))


def load_anomalies(metrics_dir: str) -> dict:
    """Load t2m_anomalies.json."""
    path = os.path.join(metrics_dir, "t2m_anomalies.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        return json.load(fh)


def load_rank_histograms(metrics_dir: str, split: str, calib: str) -> dict:
    """Load rank histogram JSON."""
    path = os.path.join(metrics_dir, f"rank_hist_{split}_{calib}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        return json.load(fh)


def get_metric_values(rows: list[dict], split: str, calib: str, var: str,
                      metric: str) -> dict:
    """Return {lead_hours: mean_value} for a given metric slice."""
    result = {}
    for r in rows:
        if (r["split"] == split and r["calibration"] == calib
                and r["variable"] == var and r["metric"] == metric):
            result[int(float(r["lead_hours"]))] = float(r["mean"])
    return result


def get_bootstrap_ci(bs_rows: list[dict], calib: str, metric: str,
                     var: str) -> dict:
    """Return {lead_hours: (est, lo, hi)}."""
    key = f"{calib}_{metric}"
    result = {}
    for r in bs_rows:
        if (r["metric"] == key and r["variable"] == var):
            lh = int(float(r["lead_hours"]))
            result[lh] = (
                float(r["ood_minus_id_estimate"]),
                float(r["ci_lower"]),
                float(r["ci_upper"]),
            )
    return result


# ---------------------------------------------------------------------------
# Figure 1: Split timeline + t2m anomalies
# ---------------------------------------------------------------------------

def fig1_splits_anomalies(cfg: dict, anomalies: dict, out_dir: str):
    """Timeline bar of splits + t2m anomaly overlaid scatter."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [1, 3]},
    )

    splits = cfg["splits"]
    colors_split = {
        "train": "#4daf4a", "val": "#377eb8",
        "id": "#ff7f00", "ood": "#e41a1c",
    }

    # Top: horizontal bars
    y_positions = {"train": 3, "val": 2, "id": 1, "ood": 0}
    for name, (start, end) in splits.items():
        sy = int(start[:4])
        ey = int(end[:4])
        ax1.barh(
            y_positions[name], ey - sy + 1, left=sy, height=0.6,
            color=colors_split[name], alpha=0.8, label=name.upper(),
        )
    ax1.set_yticks(list(y_positions.values()))
    ax1.set_yticklabels([k.upper() for k in y_positions])
    ax1.set_xlim(1978, 2023)
    ax1.legend(loc="upper right", ncol=4, fontsize=8)
    ax1.set_title("Data splits")

    # Bottom: t2m anomalies
    # Anomalies dict has mean/std per split, but not per-year values.
    # We show the aggregate as a bar.
    split_order = ["val", "id", "ood"]
    means = [anomalies.get(s, {}).get("mean", 0) for s in split_order]
    stds = [anomalies.get(s, {}).get("std", 0) for s in split_order]
    x = np.arange(len(split_order))
    bars = ax2.bar(
        x, means, yerr=stds,
        color=[colors_split[s] for s in split_order],
        capsize=5, alpha=0.8,
    )
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in split_order])
    ax2.set_ylabel("T2M anomaly (K)")
    ax2.set_title(
        "T2M anomaly (vs 1979–2005 monthly climatology)"
    )

    fig.tight_layout()
    path = os.path.join(out_dir, "01_splits_anomalies.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: RMSE vs lead time
# ---------------------------------------------------------------------------

def fig2_rmse(rows: list[dict], pers_rows: list[dict],
              bs_rows: list[dict], cfg: dict, out_dir: str):
    """Ensemble-mean RMSE vs lead time, ID vs OOD + persistence."""
    var_names = [sv["short_name"] for sv in cfg["state_variables"]]
    fig, axes = plt.subplots(
        1, len(var_names), figsize=(5 * len(var_names), 4), squeeze=False,
    )

    for vi, vname in enumerate(var_names):
        ax = axes[0, vi]

        # ID raw
        id_vals = get_metric_values(rows, "id", "raw", vname, "rmse")
        ood_vals = get_metric_values(rows, "ood", "raw", vname, "rmse")
        # Persistence
        pers_id = {}
        pers_ood = {}
        for r in pers_rows:
            lh = int(float(r["lead_hours"]))
            if r["variable"] == vname:
                if r["split"] == "id":
                    pers_id[lh] = float(r["rmse_mean"])
                else:
                    pers_ood[lh] = float(r["rmse_mean"])

        leads = sorted(id_vals.keys())
        ax.plot(leads, [id_vals[l] for l in leads],
                color=COLORS["raw_id"], linestyle=LINE_STYLES["raw_id"],
                marker="o", label="ID (raw)")
        ax.plot(leads, [ood_vals[l] for l in leads],
                color=COLORS["raw_ood"], linestyle=LINE_STYLES["raw_ood"],
                marker="s", label="OOD (raw)")

        # Persistence (average ID/OOD or plot both)
        if pers_id:
            p_leads = sorted(pers_id.keys())
            ax.plot(p_leads, [pers_id[l] for l in p_leads],
                    color=COLORS["persistence"],
                    linestyle=LINE_STYLES["persistence"],
                    marker="x", label="Persistence")

        # Bootstrap CI bands
        ci = get_bootstrap_ci(bs_rows, "raw", "rmse", vname)
        if ci:
            ci_leads = sorted(ci.keys())
            id_arr = np.array([id_vals[l] for l in ci_leads])
            lo = np.array([ci[l][1] for l in ci_leads])
            hi = np.array([ci[l][2] for l in ci_leads])
            ax.fill_between(
                ci_leads, id_arr + lo, id_arr + hi,
                alpha=0.15, color=COLORS["raw_ood"],
                label="OOD−ID 95% CI",
            )

        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(f"RMSE ({VARIABLE_UNITS.get(vname, '')})")
        ax.set_title(VARIABLE_LABELS.get(vname, vname))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ensemble-mean RMSE", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "02_rmse_id_ood.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: CRPS vs lead time (raw vs calibrated)
# ---------------------------------------------------------------------------

def fig3_crps(rows: list[dict], bs_rows: list[dict],
              cfg: dict, out_dir: str):
    """Raw vs calibrated CRPS, ID vs OOD."""
    var_names = [sv["short_name"] for sv in cfg["state_variables"]]
    fig, axes = plt.subplots(
        1, len(var_names), figsize=(5 * len(var_names), 4), squeeze=False,
    )

    for vi, vname in enumerate(var_names):
        ax = axes[0, vi]
        for cal, split, key in [
            ("raw", "id", "raw_id"), ("raw", "ood", "raw_ood"),
            ("calibrated", "id", "cal_id"),
            ("calibrated", "ood", "cal_ood"),
        ]:
            vals = get_metric_values(rows, split, cal, vname, "crps")
            if not vals:
                continue
            leads = sorted(vals.keys())
            ax.plot(leads, [vals[l] for l in leads],
                    color=COLORS[key], linestyle=LINE_STYLES[key],
                    marker="o" if "id" in key else "s",
                    label=f"{split.upper()} ({cal})")

        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(f"CRPS ({VARIABLE_UNITS.get(vname, '')})")
        ax.set_title(VARIABLE_LABELS.get(vname, vname))
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CRPS: Raw vs Calibrated", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "03_crps_calibration.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Spread-skill ratio + interval coverage
# ---------------------------------------------------------------------------

def fig4_spread_coverage(rows: list[dict], cfg: dict, out_dir: str):
    """Calibrated spread-skill ratio (target=1) and coverage (target=P)."""
    var_names = [sv["short_name"] for sv in cfg["state_variables"]]
    fig, axes = plt.subplots(
        2, len(var_names), figsize=(5 * len(var_names), 7), squeeze=False,
    )

    for vi, vname in enumerate(var_names):
        # Top: spread-skill ratio
        ax_ss = axes[0, vi]
        for split, key in [("id", "raw_id"), ("ood", "raw_ood")]:
            vals = get_metric_values(
                rows, split, "calibrated", vname, "spread_skill",
            )
            if not vals:
                continue
            leads = sorted(vals.keys())
            ax_ss.plot(leads, [vals[l] for l in leads],
                       color=COLORS[key], marker="o",
                       label=split.upper())
        ax_ss.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax_ss.set_ylabel("Spread-skill ratio")
        ax_ss.set_title(f"{VARIABLE_LABELS.get(vname, vname)} — Spread/Skill")
        ax_ss.legend(fontsize=7)
        ax_ss.grid(True, alpha=0.3)

        # Bottom: interval coverage
        ax_cov = axes[1, vi]
        for pct_label, pct_val in [("50", 0.5), ("80", 0.8), ("90", 0.9)]:
            id_vals = get_metric_values(
                rows, "id", "calibrated", vname,
                f"coverage_{pct_label}",
            )
            ood_vals = get_metric_values(
                rows, "ood", "calibrated", vname,
                f"coverage_{pct_label}",
            )
            if not id_vals:
                continue
            leads = sorted(id_vals.keys())
            ax_cov.plot(
                leads, [id_vals[l] for l in leads],
                color=COLORS["raw_id"], linestyle="-",
                marker="o", markersize=4,
                label=f"ID {pct_label}%" if pct_label == "50" else "",
            )
            ax_cov.plot(
                leads, [ood_vals[l] for l in leads],
                color=COLORS["raw_ood"], linestyle="--",
                marker="s", markersize=4,
                label=f"OOD {pct_label}%" if pct_label == "50" else "",
            )
            ax_cov.axhline(
                pct_val, color="gray", linewidth=0.5, linestyle=":",
            )
        ax_cov.set_xlabel("Lead time (h)")
        ax_cov.set_ylabel("Coverage")
        ax_cov.set_title(f"{VARIABLE_LABELS.get(vname, vname)} — Coverage")
        ax_cov.legend(fontsize=6, ncol=2)
        ax_cov.grid(True, alpha=0.3)

    fig.suptitle("Calibrated Ensemble: Spread-Skill & Coverage",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "04_spread_skill_coverage.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Rank histograms at 72h
# ---------------------------------------------------------------------------

def fig5_rank_histograms(metrics_dir: str, cfg: dict, out_dir: str):
    """Calibrated t2m rank histograms at 72h, ID vs OOD."""
    var_names = [sv["short_name"] for sv in cfg["state_variables"]]
    lead_72 = 72

    rh_id = load_rank_histograms(metrics_dir, "id", "calibrated")
    rh_ood = load_rank_histograms(metrics_dir, "ood", "calibrated")

    fig, axes = plt.subplots(
        1, len(var_names), figsize=(4 * len(var_names), 3.5), squeeze=False,
    )

    for vi, vname in enumerate(var_names):
        ax = axes[0, vi]
        id_bins = None
        ood_bins = None

        if vname in rh_id and str(lead_72) in rh_id[vname]:
            id_bins = np.array(rh_id[vname][str(lead_72)])
        if vname in rh_ood and str(lead_72) in rh_ood[vname]:
            ood_bins = np.array(rh_ood[vname][str(lead_72)])

        if id_bins is None and ood_bins is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(VARIABLE_LABELS.get(vname, vname))
            continue

        n_bins = len(id_bins) if id_bins is not None else len(ood_bins)
        x = np.arange(n_bins)
        width = 0.35

        if id_bins is not None:
            ax.bar(x - width/2, id_bins, width, color=COLORS["raw_id"],
                   alpha=0.7, label="ID")
        if ood_bins is not None:
            ax.bar(x + width/2, ood_bins, width, color=COLORS["raw_ood"],
                   alpha=0.7, label="OOD")

        # Flat reference
        ax.axhline(1.0 / n_bins, color="black", linewidth=0.5,
                   linestyle="--", label="Uniform")

        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{VARIABLE_LABELS.get(vname, vname)} @ {lead_72}h")
        ax.legend(fontsize=7)

    fig.suptitle(f"Calibrated Rank Histograms at {lead_72}h", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "05_rank_histograms.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(
        description="Plot results for Graph-EFM temporal-shift experiment"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--metrics", type=str, default="results",
        help="Directory containing evaluation CSV/JSON files",
    )
    parser.add_argument(
        "--output", type=str, default="figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    os.makedirs(args.output, exist_ok=True)

    # Load data
    rows = load_metrics(args.metrics)
    bs_rows = load_bootstrap(args.metrics)
    pers_rows = load_persistence(args.metrics)
    anomalies = load_anomalies(args.metrics)

    if not rows:
        print("No metrics found. Run evaluate_shift.py first.")
        return

    print(f"Loaded {len(rows)} metric rows, {len(bs_rows)} bootstrap rows, "
          f"{len(pers_rows)} persistence rows")

    print("\nGenerating figures...")
    fig1_splits_anomalies(cfg, anomalies, args.output)
    fig2_rmse(rows, pers_rows, bs_rows, cfg, args.output)
    fig3_crps(rows, bs_rows, cfg, args.output)
    fig4_spread_coverage(rows, cfg, args.output)
    fig5_rank_histograms(args.metrics, cfg, args.output)

    print(f"\n✓ All figures saved to {args.output}/")


if __name__ == "__main__":
    main()
