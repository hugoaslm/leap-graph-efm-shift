#!/usr/bin/env python3
"""
Two-phase training wrapper for the Graph-EFM temporal-shift experiment.

Phase A: train with ar_steps=1 (single-step), save best checkpoint.
Phase B: resume from Phase A best, train with ar_steps=4 (multi-step).

Supports Colab budget management via --max_minutes.

Usage:
    python scripts/train_shift_model.py \\
        --config configs/wb2_shift_64x32_graph_efm.yaml \\
        --phase a \\
        --max_minutes 120

    python scripts/train_shift_model.py \\
        --config configs/wb2_shift_64x32_graph_efm.yaml \\
        --phase b \\
        --resume checkpoints/best_phase_a.ckpt \\
        --max_minutes 120
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import yaml

# Path to the neural-lam-prob-model directory
REPO_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "neural-lam-prob-model"
)
REPO_ROOT = os.path.abspath(REPO_ROOT)

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints"
)
CHECKPOINT_DIR = os.path.abspath(CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_path(path: str) -> str:
    """Resolve a path relative to the workspace root."""
    workspace = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(workspace, path))


def build_train_args(cfg: dict, phase: str, config_path: str,
                     resume_ckpt: str | None = None) -> list[str]:
    """Build the argument list for train_model.py from the YAML config."""
    model_cfg = cfg["model"]
    train_cfg = cfg["training"][f"phase_{phase}"]
    forecast_cfg = cfg["forecast"]
    graph_cfg = cfg["graph"]
    eval_cfg = cfg["evaluation"]

    args = [
        sys.executable,
        os.path.join(REPO_ROOT, "train_model.py"),
        "--dataset", cfg["dataset"]["name"],
        "--config", resolve_path(config_path),
        "--model", model_cfg["name"],
        "--graph", graph_cfg["name"],
        "--hidden_dim", str(model_cfg["hidden_dim"]),
        "--latent_dim", str(model_cfg["latent_dim"]),
        "--hidden_layers", str(model_cfg["hidden_layers"]),
        "--processor_layers", str(model_cfg["decoder_processor_layers"]),
        "--encoder_processor_layers",
            str(model_cfg["encoder_processor_layers"]),
        "--prior_processor_layers",
            str(model_cfg["prior_processor_layers"]),
        "--loss", model_cfg["loss"],
        "--kl_beta", str(model_cfg["kl_beta"]),
        "--crps_weight", str(model_cfg["crps_weight"]),
        "--output_std", str(int(model_cfg["output_std"])),
        "--sample_obs_noise", str(int(model_cfg["sample_obs_noise"])),
        "--prior_dist", model_cfg["prior_dist"],
        "--learn_prior", str(int(model_cfg["learn_prior"])),
        "--precision", model_cfg["precision"],
        "--ar_steps", str(
            forecast_cfg[f"ar_steps_train_phase_{phase}"]
        ),
        "--eval_leads", str(forecast_cfg["eval_leads"]),
        "--step_length", str(cfg["sampling"]["step_length"]),
        "--batch_size", str(train_cfg["batch_size"]),
        "--epochs", str(train_cfg["epochs"]),
        "--lr", str(train_cfg["lr"]),
        "--ensemble_size", str(eval_cfg["ensemble_size"]),
        "--n_workers", str(eval_cfg.get("n_workers", 4)),
        "--seed", str(eval_cfg["seed"]),
        "--sanity_batches", "0",    # skip sanity check to save time
    ]

    if resume_ckpt is not None:
        args += ["--load", resolve_path(resume_ckpt)]

    return args


def run_with_timeout(cmd: list[str], max_minutes: int,
                     cwd: str) -> subprocess.CompletedProcess:
    """Run a subprocess with a wall-clock timeout."""
    print(f"\n{'=' * 60}")
    print(f"Training phase (max {max_minutes} min)")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working dir: {cwd}")
    print(f"{'=' * 60}\n")

    start = time.time()
    timeout_sec = max_minutes * 60

    if sys.platform == "win32":
        # Windows: no signal-based timeout; just run
        proc = subprocess.Popen(cmd, cwd=cwd)
    else:
        proc = subprocess.Popen(cmd, cwd=cwd, preexec_fn=os.setsid)

    try:
        remaining = timeout_sec
        while remaining > 0:
            try:
                ret = proc.wait(timeout=min(remaining, 30))
                elapsed = time.time() - start
                print(f"\nTraining completed in {elapsed / 60:.1f} min "
                      f"(exit code {ret})")
                return subprocess.CompletedProcess(
                    cmd, ret, stdout="", stderr=""
                )
            except subprocess.TimeoutExpired:
                remaining = timeout_sec - (time.time() - start)

        # Timeout reached
        print(f"\nTimeout ({max_minutes} min) reached — "
              f"terminating training...")
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()

        return subprocess.CompletedProcess(cmd, -1, stdout="", stderr="")

    except KeyboardInterrupt:
        print("\nInterrupted — terminating training...")
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        raise


def find_best_checkpoint() -> str | None:
    """Find the min_val_loss checkpoint from the latest run."""
    saved_models = os.path.join(REPO_ROOT, "saved_models")
    if not os.path.isdir(saved_models):
        return None

    # Find most recent run directory
    run_dirs = sorted(
        [d for d in os.listdir(saved_models)
         if os.path.isdir(os.path.join(saved_models, d))],
        reverse=True,
    )
    for run_dir in run_dirs:
        full = os.path.join(saved_models, run_dir)
        for fname in ["min_val_loss.ckpt", "last.ckpt"]:
            path = os.path.join(full, fname)
            if os.path.exists(path):
                return path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(
        description="Two-phase Graph-EFM training for temporal-shift experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["a", "b"],
        help="Training phase: 'a' (ar_steps=1) or 'b' (ar_steps=4)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (required for phase b)",
    )
    parser.add_argument(
        "--max_minutes",
        type=int,
        default=120,
        help="Maximum wall-clock time for this phase (default: 120)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the command without executing",
    )
    args = parser.parse_args()

    # Load config
    config_path = resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Validate phase
    if args.phase == "b" and args.resume is None:
        # Try to auto-find the Phase A checkpoint
        auto = find_best_checkpoint()
        if auto:
            print(f"Auto-resume from: {auto}")
            args.resume = auto
        else:
            parser.error(
                "Phase B requires --resume <checkpoint>. "
                "No checkpoint found automatically."
            )

    if args.phase == "b":
        assert args.resume, "Phase B requires --resume"
        print(f"Resuming from: {resolve_path(args.resume)}")

    # Build command
    cmd = build_train_args(
        cfg,
        args.phase,
        args.config,
        resume_ckpt=args.resume if args.phase == "b" else None,
    )

    # Ensure directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "saved_models"), exist_ok=True)

    if args.dry_run:
        print("\n[Dry run] Would execute:")
        print("  " + " ".join(cmd))
        return

    # Run training
    try:
        run_with_timeout(cmd, args.max_minutes, cwd=REPO_ROOT)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

    # After training, copy the best checkpoint to the checkpoints directory
    best = find_best_checkpoint()
    if best:
        phase_name = f"best_phase_{args.phase}.ckpt"
        dest = os.path.join(CHECKPOINT_DIR, phase_name)

        import shutil
        shutil.copy2(best, dest)
        print(f"\nBest checkpoint saved to: {dest}")
    else:
        print("\nWarning: No checkpoint found after training.")


if __name__ == "__main__":
    main()
