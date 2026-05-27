import os, signal, subprocess, sys, shutil, time
from argparse import ArgumentParser

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "neural-lam-prob-model"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))


def resolve_path(path: str) -> str:
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(workspace, path))


def build_train_args(cfg, phase, config_path, resume_ckpt=None):
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


def run_with_timeout(cmd, max_minutes, cwd):
    timeout_sec = max_minutes * 60
    start = time.time()
    if sys.platform == "win32":
        proc = subprocess.Popen(cmd, cwd=cwd)
    else:
        proc = subprocess.Popen(cmd, cwd=cwd, preexec_fn=os.setsid)
    try:
        remaining = timeout_sec
        while remaining > 0:
            try:
                ret = proc.wait(timeout=min(remaining, 30))
                elapsed = time.time() - start
                print(f"Training finished ({elapsed/60:.1f} min)")
                return subprocess.CompletedProcess(cmd, ret, stdout="", stderr="")
            except subprocess.TimeoutExpired:
                remaining = timeout_sec - (time.time() - start)
        print(f"Timeout ({max_minutes} min), stopping.")
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
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        raise


def find_best_checkpoint():
    saved_models = os.path.join(REPO_ROOT, "saved_models")
    if not os.path.isdir(saved_models):
        return None
    run_dirs = sorted(
        [d for d in os.listdir(saved_models) if os.path.isdir(os.path.join(saved_models, d))],
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

    config_path = resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.phase == "b" and args.resume is None:
        auto = find_best_checkpoint()
        if auto:
            args.resume = auto
        else:
            parser.error("Phase B needs --resume <checkpoint> or a completed Phase A run.")

    if args.phase == "b":
        assert args.resume, "Phase B requires --resume"

    cmd = build_train_args(cfg, args.phase, args.config,
                           resume_ckpt=args.resume if args.phase == "b" else None)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "saved_models"), exist_ok=True)

    if args.dry_run:
        print("Would run:", " ".join(cmd))
        return

    try:
        run_with_timeout(cmd, args.max_minutes, cwd=REPO_ROOT)
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)

    best = find_best_checkpoint()
    if best:
        dest = os.path.join(CHECKPOINT_DIR, f"best_phase_{args.phase}.ckpt")
        shutil.copy2(best, dest)
        print(f"Checkpoint: {dest}")
    else:
        print("No checkpoint found after training.")


if __name__ == "__main__":
    main()
