import os, subprocess, sys
from argparse import ArgumentParser

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "neural-lam-prob-model"))
if not os.path.isdir(REPO_ROOT):
    REPO_ROOT = os.getcwd()


def _run(script_name, *args):
    script_path = os.path.join(REPO_ROOT, script_name)
    cmd = [sys.executable, script_path] + list(args)
    print("Running", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _exists(*paths):
    return all(os.path.exists(p) for p in paths)


def step_download(cfg):
    fields_path = cfg["dataset"]["fields_zarr"]
    if _exists(fields_path):
        return
    raise RuntimeError("fields.zarr missing")


def step_forcing(cfg):
    dataset = cfg["dataset"]["name"]
    output = cfg["dataset"]["forcing_zarr"]
    if _exists(output):
        return
    _run("create_global_forcing.py", "--dataset", dataset,
         "--time_start", cfg["splits"]["train"][0],
         "--time_end", cfg["splits"]["ood"][1])


def step_grid_features(cfg):
    dataset = cfg["dataset"]["name"]
    static_dir = cfg["dataset"]["static_dir"]
    if _exists(os.path.join(static_dir, "grid_features.pt")):
        return
    _run("create_global_grid_features.py", "--dataset", dataset)


def step_mesh(cfg):
    dataset = cfg["dataset"]["name"]
    graph_name = cfg["graph"]["name"]
    graph_dir = os.path.join(REPO_ROOT, "graphs", graph_name)
    if _exists(os.path.join(graph_dir, "m2m_edge_index.pt")):
        return
    _run("create_global_mesh.py", "--dataset", dataset, "--graph", graph_name,
         "--splits", str(cfg["graph"]["splits"]),
         "--levels", str(cfg["graph"]["levels"]),
         "--hierarchical", str(int(cfg["graph"]["hierarchical"])))


def step_parameter_weights(cfg, config_path):
    dataset = cfg["dataset"]["name"]
    static_dir = cfg["dataset"]["static_dir"]
    needed = ["parameter_mean.pt", "parameter_std.pt", "diff_mean.pt",
              "diff_std.pt", "parameter_weights.npy", "grid_weights.pt"]
    if _exists(*[os.path.join(static_dir, f) for f in needed]):
        return
    _run("create_parameter_weights.py", "--dataset", dataset,
         "--config", config_path, "--global_dataset", "1", "--batch_size", "8")


STEP_REGISTRY = {
    "download": step_download,
    "forcing": step_forcing,
    "grid_features": step_grid_features,
    "mesh": step_mesh,
    "parameter_weights": step_parameter_weights,
}


def main():
    parser = ArgumentParser(description="Prepare data for Graph-EFM temporal-shift experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", default="all",
                        help=f"Comma-separated steps: {', '.join(STEP_REGISTRY)} (default: all)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.steps == "all":
        steps_to_run = list(STEP_REGISTRY.keys())
    else:
        steps_to_run = [s.strip() for s in args.steps.split(",")]
        unknown = set(steps_to_run) - set(STEP_REGISTRY)
        if unknown:
            parser.error(f"Unknown steps: {', '.join(unknown)}")

    os.makedirs(cfg["dataset"]["static_dir"], exist_ok=True)
    graph_dir = os.path.join(REPO_ROOT, "graphs", cfg["graph"]["name"])
    os.makedirs(graph_dir, exist_ok=True)

    for name in steps_to_run:
        func = STEP_REGISTRY[name]
        if name == "parameter_weights":
            func(cfg, config_path)
        else:
            func(cfg)


if __name__ == "__main__":
    main()
