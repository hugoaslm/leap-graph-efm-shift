import os, shutil, subprocess, sys, time
from argparse import ArgumentParser

import xarray as xr
import yaml

REQUIRED_VARS = [
    "geopotential", "temperature", "2m_temperature",
    "geopotential_at_surface", "land_sea_mask",
]


def valid_fields_store(path, cfg):
    if not os.path.exists(path):
        return False
    try:
        ds = xr.open_zarr(path, consolidated=False)
    except Exception as exc:
        print(f"Invalid fields cache at {path}: {exc}")
        return False
    missing = set(REQUIRED_VARS) - set(ds.data_vars)
    expected_grid = tuple(cfg["grid"]["shape"])
    actual_grid = (ds.sizes.get("longitude"), ds.sizes.get("latitude"))
    state_dims = ds["2m_temperature"].dims if "2m_temperature" in ds else ()
    static_dims = ds["land_sea_mask"].dims if "land_sea_mask" in ds else ()
    ordered = state_dims == ("time", "longitude", "latitude") and \
        static_dims == ("longitude", "latitude")
    if missing or actual_grid != expected_grid or not ordered:
        print(f"Invalid fields cache at {path}: missing={sorted(missing)}, "
              f"grid={actual_grid}, dims={state_dims}")
        return False
    return True


def retire_invalid_store(path, cfg):
    if os.path.lexists(path) and not valid_fields_store(path, cfg):
        backup = f"{path}.invalid_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"Moving invalid cache aside: {path} -> {backup}")
        os.rename(path, backup)


def needs_rechunk(path):
    try:
        with xr.open_zarr(path) as ds:
            da = ds.get("temperature")
            if da is None or not da.chunks:
                return False
            t_chunks = da.chunks[0]
            return len(set(t_chunks)) == 1 and t_chunks[0] == 1
    except Exception:
        return False


def copy_path(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def restore_from_drive(local_path, drive_path, nl_path=None):
    if not os.path.exists(local_path) and os.path.exists(drive_path):
        print(f"Copying from Drive: {drive_path}")
        copy_path(drive_path, local_path)
    if nl_path is not None and not os.path.lexists(nl_path) and \
            os.path.exists(local_path):
        os.symlink(local_path, nl_path)


def persist_output(nl_path, local_path, drive_path):
    if not os.path.exists(nl_path) and not os.path.exists(local_path):
        return
    if os.path.islink(nl_path):
        local_path = os.path.realpath(nl_path)
    elif os.path.exists(nl_path):
        if not os.path.exists(local_path):
            print(f"Copying to local: {nl_path}")
            copy_path(nl_path, local_path)
        if os.path.isdir(nl_path):
            shutil.rmtree(nl_path)
        else:
            os.remove(nl_path)
        os.symlink(local_path, nl_path)
    if not os.path.exists(drive_path):
        print(f"Copying to Drive: {local_path}")
        copy_path(local_path, drive_path)
    elif os.path.isdir(local_path):
        shutil.copytree(local_path, drive_path, dirs_exist_ok=True)


def main():
    parser = ArgumentParser(description="Prepare data for Colab run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--nl", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--drive", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    t_start = time.time()

    def report(phase):
        print(f"  [{phase} done in {time.time() - t_start:.0f}s]")

    data_name = cfg["dataset"]["name"]
    drive_data = os.path.join(args.drive, "data", data_name)
    local_data = os.path.join(args.project, "data", data_name)
    nl_data = os.path.join(args.nl, "data", data_name)
    os.makedirs(local_data, exist_ok=True)
    os.makedirs(nl_data, exist_ok=True)
    os.makedirs(drive_data, exist_ok=True)
    os.makedirs(os.path.join(args.nl, "graphs"), exist_ok=True)

    fields_zarr = os.path.join(local_data, "fields.zarr")
    drive_fields_zarr = os.path.join(drive_data, "fields.zarr")
    nl_fields_zarr = os.path.join(nl_data, "fields.zarr")
    retire_invalid_store(fields_zarr, cfg)
    retire_invalid_store(drive_fields_zarr, cfg)
    retire_invalid_store(nl_fields_zarr, cfg)
    if not valid_fields_store(fields_zarr, cfg):
        if valid_fields_store(drive_fields_zarr, cfg):
            print("Copying data from Drive cache...")
            shutil.copytree(drive_data, local_data, dirs_exist_ok=True)
        else:
            print("Downloading WB2 data (~4 GB; public bucket, no auth)...")
            proc = subprocess.run(
                [sys.executable, "-u",
                 os.path.join(args.repo, "scripts", "download_wb2_data.py"),
                 "--output", fields_zarr,
                 "--time_start", cfg["splits"]["train"][0],
                 "--time_end", cfg["splits"]["ood"][1],
                 "--method", "xarray"],
                capture_output=True, text=True)
            sys.stdout.write(proc.stdout)
            sys.stdout.flush()
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
                sys.stderr.flush()
                raise RuntimeError(
                    f"WB2 download script failed (exit {proc.returncode}).")
            os.makedirs(drive_data, exist_ok=True)
            shutil.copytree(fields_zarr, drive_fields_zarr,
                            dirs_exist_ok=True)
    else:
        print("Data already on local disk.")
    report("fields.zarr (download/copy)")

    if not valid_fields_store(fields_zarr, cfg):
        raise RuntimeError(f"Downloaded fields store failed validation: "
                           f"{fields_zarr}")
    if not os.path.lexists(nl_fields_zarr):
        os.symlink(fields_zarr, nl_fields_zarr)
    if not valid_fields_store(nl_fields_zarr, cfg):
        raise RuntimeError(f"neural-lam cannot read fields store: "
                           f"{nl_fields_zarr}")

    if needs_rechunk(fields_zarr):
        print("Rechunking fields.zarr to time:512 (one-time, ~5-10 min)...")
        t0 = time.time()
        tmp = fields_zarr + ".rechunk"
        with xr.open_zarr(fields_zarr) as ds:
            ds = ds.chunk({"time": 512})
            enc = {}
            for v in ds.data_vars:
                if "time" in ds[v].dims:
                    enc[v] = {"chunks": (512,) + (-1,) * (ds[v].ndim - 1)}
                else:
                    enc[v] = {"chunks": ds[v].shape}
            ds.to_zarr(tmp, mode="w", encoding=enc)
        shutil.rmtree(fields_zarr)
        os.rename(tmp, fields_zarr)
        print(f"Copying rechunked fields to Drive: {drive_fields_zarr}")
        shutil.rmtree(drive_fields_zarr)
        shutil.copytree(fields_zarr, drive_fields_zarr)
        print(f"Rechunk complete ({time.time() - t0:.0f}s).")
    report("fields.zarr (rechunk/validate)")

    graph_name = cfg["graph"]["name"]
    local_graph = os.path.join(args.nl, "graphs", graph_name)
    drive_graph = os.path.join(args.drive, "graphs", graph_name)
    restore_from_drive(os.path.join(local_data, "forcing.zarr"),
                       os.path.join(drive_data, "forcing.zarr"),
                       os.path.join(nl_data, "forcing.zarr"))
    restore_from_drive(os.path.join(local_data, "static"),
                       os.path.join(drive_data, "static"),
                       os.path.join(nl_data, "static"))
    restore_from_drive(local_graph, drive_graph)

    config_path = os.path.abspath(args.config)
    prep_script = os.path.join(args.repo, "scripts",
                               "prepare_wb2_subset.py")
    steps_pending = []
    for step in ["forcing", "grid_features", "mesh", "parameter_weights"]:
        steps_pending.append(step)
        t0 = time.time()
        print(f"Running preprocessing step: {step} "
              f"(pending: {', '.join(steps_pending)})")
        subprocess.run(
            [sys.executable, "-u", prep_script,
             "--config", config_path, "--steps", step],
            cwd=args.nl, check=True)
        steps_pending.remove(step)
        print(f"[{step} took {time.time() - t0:.0f}s]")
        if step == "forcing":
            persist_output(os.path.join(nl_data, "forcing.zarr"),
                           os.path.join(local_data, "forcing.zarr"),
                           os.path.join(drive_data, "forcing.zarr"))
        elif step in ("grid_features", "parameter_weights"):
            persist_output(os.path.join(nl_data, "static"),
                           os.path.join(local_data, "static"),
                           os.path.join(drive_data, "static"))
        elif step == "mesh":
            if os.path.isdir(local_graph) and not os.path.exists(drive_graph):
                print(f"Copying to Drive: {local_graph}")
                copy_path(local_graph, drive_graph)

    print(f"PREP FINISHED in {time.time() - t_start:.0f}s total.")


if __name__ == "__main__":
    main()
