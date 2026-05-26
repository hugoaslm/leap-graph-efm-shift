#!/usr/bin/env python3
"""
Preprocessing orchestrator for the Graph-EFM temporal-shift experiment.

Runs all data-preparation steps for the reduced WeatherBench 2 ERA5 64×32 setup:
  1. Download / verify the 64×32 zarr store
  2. Generate forcing features (TOA radiation, year/day progress)
  3. Generate grid static features (lat/lon encoding, geopotential, land-sea mask)
  4. Generate hierarchical mesh graph
  5. Compute parameter statistics and weights (train-set only)

Each step is skipped if its output already exists, making the script safe to
re-run (incremental / cache-friendly).

Usage:
    python scripts/prepare_wb2_subset.py \\
        --config configs/wb2_shift_64x32_graph_efm.yaml

Or run individual steps:
    python scripts/prepare_wb2_subset.py \\
        --config configs/wb2_shift_64x32_graph_efm.yaml \\
        --steps forcing,grid_features
"""
from __future__ import annotations

import os
import subprocess
import sys
from argparse import ArgumentParser

import yaml

# Path to the neural-lam-prob-model directory (where the generation scripts live)
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "neural-lam-prob-model")
REPO_ROOT = os.path.abspath(REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_python(script_name: str, *args: str) -> None:
    """Run a Python script inside the neural-lam-prob-model directory."""
    script_path = os.path.join(REPO_ROOT, script_name)
    cmd = [sys.executable, script_path] + list(args)
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def check_exists(*paths: str) -> bool:
    """Return True if all paths exist."""
    return all(os.path.exists(p) for p in paths)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_download(cfg: dict) -> None:
    """Download WB2 data (or verify it exists)."""
    fields_path = cfg["dataset"]["fields_zarr"]
    if check_exists(fields_path):
        print(f"[skip] fields.zarr already exists at {fields_path}")
        return

    print("Data download must be run manually or on Colab:")
    print(f"  python scripts/download_wb2_data.py "
          f"--output {fields_path} "
          f"--time_start {cfg['splits']['train'][0]} "
          f"--time_end {cfg['splits']['ood'][1]}")
    raise RuntimeError(
        "fields.zarr not found. Run scripts/download_wb2_data.py first."
    )


def step_forcing(cfg: dict) -> None:
    """Generate forcing features."""
    dataset = cfg["dataset"]["name"]
    output = cfg["dataset"]["forcing_zarr"]
    if check_exists(output):
        print(f"[skip] forcing.zarr already exists at {output}")
        return

    # Determine time range from splits
    train_start = cfg["splits"]["train"][0]
    ood_end = cfg["splits"]["ood"][1]

    run_python(
        "create_global_forcing.py",
        "--dataset", dataset,
        "--time_start", train_start,
        "--time_end", ood_end,
    )


def step_grid_features(cfg: dict) -> None:
    """Generate grid static features."""
    dataset = cfg["dataset"]["name"]
    static_dir = cfg["dataset"]["static_dir"]
    output = os.path.join(static_dir, "grid_features.pt")
    if check_exists(output):
        print(f"[skip] grid_features.pt already exists at {output}")
        return

    run_python(
        "create_global_grid_features.py",
        "--dataset", dataset,
    )


def step_mesh(cfg: dict) -> None:
    """Generate hierarchical mesh graph."""
    dataset = cfg["dataset"]["name"]
    graph_name = cfg["graph"]["name"]
    graph_dir = os.path.join(REPO_ROOT, "graphs", graph_name)
    output = os.path.join(graph_dir, "m2m_edge_index.pt")
    if check_exists(output):
        print(f"[skip] graph already exists at {graph_dir}")
        return

    run_python(
        "create_global_mesh.py",
        "--dataset", dataset,
        "--graph", graph_name,
        "--splits", str(cfg["graph"]["splits"]),
        "--levels", str(cfg["graph"]["levels"]),
        "--hierarchical", str(int(cfg["graph"]["hierarchical"])),
    )


def step_parameter_weights(cfg: dict, config_path: str) -> None:
    """Compute parameter statistics and weights (train-set only)."""
    dataset = cfg["dataset"]["name"]
    static_dir = cfg["dataset"]["static_dir"]
    outputs = [
        os.path.join(static_dir, "parameter_mean.pt"),
        os.path.join(static_dir, "parameter_std.pt"),
        os.path.join(static_dir, "diff_mean.pt"),
        os.path.join(static_dir, "diff_std.pt"),
        os.path.join(static_dir, "parameter_weights.npy"),
        os.path.join(static_dir, "grid_weights.pt"),
    ]
    if check_exists(*outputs):
        print(f"[skip] parameter stats already exist in {static_dir}")
        return

    run_python(
        "create_parameter_weights.py",
        "--dataset", dataset,
        "--config", config_path,
        "--global_dataset", "1",
        "--batch_size", "8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEP_REGISTRY = {
    "download": step_download,
    "forcing": step_forcing,
    "grid_features": step_grid_features,
    "mesh": step_mesh,
    "parameter_weights": step_parameter_weights,
}

ALL_STEPS = list(STEP_REGISTRY.keys())


def main():
    parser = ArgumentParser(
        description="Prepare data for the Graph-EFM temporal-shift experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help=f"Comma-separated list of steps to run. "
        f"Available: {', '.join(ALL_STEPS)}. "
        f"Default: all",
    )
    args = parser.parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve step list
    if args.steps == "all":
        steps_to_run = ALL_STEPS
    else:
        steps_to_run = [s.strip() for s in args.steps.split(",")]
        unknown = set(steps_to_run) - set(ALL_STEPS)
        if unknown:
            parser.error(f"Unknown step(s): {', '.join(unknown)}")

    # Ensure data directory exists
    os.makedirs(cfg["dataset"]["static_dir"], exist_ok=True)
    graph_dir = os.path.join(REPO_ROOT, "graphs", cfg["graph"]["name"])
    os.makedirs(graph_dir, exist_ok=True)

    # Run steps
    for step_name in steps_to_run:
        func = STEP_REGISTRY[step_name]
        if step_name == "parameter_weights":
            func(cfg, config_path)
        else:
            func(cfg)

    print("\n✓ All steps completed.")


if __name__ == "__main__":
    main()
