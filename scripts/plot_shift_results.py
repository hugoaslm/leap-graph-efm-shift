#!/usr/bin/env python3
import json, os, csv
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import yaml

COLORS = {
    "raw_id": "#2166ac", "raw_ood": "#b2182b",
    "cal_id": "#4393c3", "cal_ood": "#d6604d",
    "persistence": "#666666",
}
LINE_STYLES = {
    "raw_id": "-", "raw_ood": "-",
    "cal_id": "--", "cal_ood": "--",
    "persistence": ":",
}
VAR_LABELS = {"z500": "Z500", "t850": "T850", "t2m": "T2M"}
VAR_UNITS = {"z500": "m²/s²", "t850": "K", "t2m": "K"}


def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _metric_vals(rows, split, calib, var, metric):
    result = {}
    for r in rows:
        if (r["split"] == split and r["calibration"] == calib
                and r["variable"] == var and r["metric"] == metric):
            result[int(float(r["lead_hours"]))] = float(r["mean"])
    return result


def _bootstrap_ci(bs_rows, calib, metric, var):
    key = f"{calib}_{metric}"
    result = {}
    for r in bs_rows:
        if r["metric"] == key and r["variable"] == var:
            lh = int(float(r["lead_hours"]))
            result[lh] = (float(r["ood_minus_id_estimate"]),
                          float(r["ci_lower"]), float(r["ci_upper"]))
    return result


def fig1_splits_anomalies(cfg: dict, anomalies: dict, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [1, 3]},
    )

    splits = cfg["splits"]
    colors_split = {
        "train": "#4daf4a", "val": "#377eb8",
        "id": "#ff7f00", "ood": "#e41a1c",
    }

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


def fig2_rmse(rows: list[dict], pers_rows: list[dict],
              bs_rows: list[dict], cfg: dict, out_dir: str):
    var_names = [sv["short_name"] for sv in cfg["state_variables"]]
    fig, axes = plt.subplots(
        1, len(var_names), figsize=(5 * len(var_names), 4), squeeze=False,
    )

    for vi, vname in enumerate(var_names):
        ax = axes[0, vi]

        id_vals = _metric_vals(rows, "id", "raw", vname, "rmse")
        ood_vals = _metric_vals(rows, "ood", "raw", vname, "rmse")
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

        if pers_id:
            p_leads = sorted(pers_id.keys())
            ax.plot(p_leads, [pers_id[l] for l in p_leads],
                    color=COLORS["persistence"],
                    linestyle=LINE_STYLES["persistence"],
                    marker="x", label="Persistence")

        ci = _bootstrap_ci(bs_rows, "raw", "rmse", vname)
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
        ax.set_ylabel(f"RMSE ({VAR_UNITS.get(vname, '')})")
        ax.set_title(VAR_LABELS.get(vname, vname))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ensemble-mean RMSE", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "02_rmse_id_ood.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig3_crps(rows: list[dict], bs_rows: list[dict],
              cfg: dict, out_dir: str):
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
            vals = _metric_vals(rows, split, cal, vname, "crps")
            if not vals:
                continue
            leads = sorted(vals.keys())
            ax.plot(leads, [vals[l] for l in leads],
                    color=COLORS[key], linestyle=LINE_STYLES[key],
                    marker="o" if "id" in key else "s",
                    label=f"{split.upper()} ({cal})")

        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(f"CRPS ({VAR_UNITS.get(vname, '')})")
        ax.set_title(VAR_LABELS.get(vname, vname))
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CRPS: Raw vs Calibrated", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "03_crps_calibration.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_spread_coverage(rows: list[dict], cfg: dict, out_dir: str):
    var_names = [sv["short_name"] for sv in cfg["state_variables"]]
    fig, axes = plt.subplots(
        2, len(var_names), figsize=(5 * len(var_names), 7), squeeze=False,
    )

    for vi, vname in enumerate(var_names):
        ax_ss = axes[0, vi]
        for split, key in [("id", "raw_id"), ("ood", "raw_ood")]:
            vals = _metric_vals(
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
        ax_ss.set_title(f"{VAR_LABELS.get(vname, vname)} — Spread/Skill")
        ax_ss.legend(fontsize=7)
        ax_ss.grid(True, alpha=0.3)

        ax_cov = axes[1, vi]
        for pct_label, pct_val in [("50", 0.5), ("80", 0.8), ("90", 0.9)]:
            id_vals = _metric_vals(
                rows, "id", "calibrated", vname,
                f"coverage_{pct_label}",
            )
            ood_vals = _metric_vals(
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
        ax_cov.set_title(f"{VAR_LABELS.get(vname, vname)} — Coverage")
        ax_cov.legend(fontsize=6, ncol=2)
        ax_cov.grid(True, alpha=0.3)

    fig.suptitle("Calibrated Ensemble: Spread-Skill & Coverage",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "04_spread_skill_coverage.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig5_rank_histograms(metrics_dir: str, cfg: dict, out_dir: str):
    var_names = [sv["short_name"] for sv in cfg["state_variables"]]
    lead_72 = 72

    rh_id = _load_json(os.path.join(metrics_dir, "rank_hist_id_calibrated.json"))
    rh_ood = _load_json(os.path.join(metrics_dir, "rank_hist_ood_calibrated.json"))

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
            ax.set_title(VAR_LABELS.get(vname, vname))
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

        ax.axhline(1.0 / n_bins, color="black", linewidth=0.5,
                   linestyle="--", label="Uniform")

        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{VAR_LABELS.get(vname, vname)} @ {lead_72}h")
        ax.legend(fontsize=7)

    fig.suptitle(f"Calibrated Rank Histograms at {lead_72}h", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "05_rank_histograms.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = ArgumentParser(
        description="Plot results for Graph-EFM temporal-shift experiment"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="results")
    parser.add_argument("--output", type=str, default="figures")
    args = parser.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    os.makedirs(args.output, exist_ok=True)

    rows = _load_csv(os.path.join(args.metrics, "metrics.csv"))
    bs_rows = _load_csv(os.path.join(args.metrics, "bootstrap_ci.csv"))
    pers_rows = _load_csv(os.path.join(args.metrics, "persistence_rmse.csv"))
    anomalies = _load_json(os.path.join(args.metrics, "t2m_anomalies.json"))

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
