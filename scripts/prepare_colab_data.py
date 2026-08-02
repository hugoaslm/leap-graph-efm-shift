import os, shutil, subprocess, sys, tarfile, time
from argparse import ArgumentParser

import xarray as xr
import yaml

try:
    from dask.diagnostics import ProgressBar
except ImportError:
    class ProgressBar:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

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


def copy_tree_with_progress(src, dst, label):
    total = sum(os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(src) for f in files)
    done = 0
    count = 0
    t0 = time.time()
    for root, _, files in os.walk(src):
        target = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(target, exist_ok=True)
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(target, f)
            sz = os.path.getsize(s)
            shutil.copy2(s, d)
            done += sz
            count += 1
            if count % 100 == 0 or done >= total:
                print(f"  {label}: {done / 1e9:.2f}/{total / 1e9:.2f} GB "
                      f"({done * 100 / max(total, 1):.0f}%), {count} files",
                      flush=True)
    print(f"  {label} copied in {time.time() - t0:.0f}s", flush=True)


def ensure_symlink(local_path, nl_path):
    if os.path.lexists(nl_path):
        return
    if os.path.exists(local_path):
        os.symlink(local_path, nl_path)


def build_archive(local_tar, local_data, graph_dir):
    tmp = local_tar + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    with tarfile.open(tmp, "w") as tar:
        for name in ["fields.zarr", "static"]:
            path = os.path.join(local_data, name)
            if os.path.isdir(path):
                tar.add(path, arcname=name)
        if graph_dir is not None and os.path.isdir(graph_dir):
            tar.add(graph_dir, arcname="graphs/" + os.path.basename(graph_dir))
    os.replace(tmp, local_tar)


def upload_to_drive(local_tar, drive_archive):
    tmp = drive_archive + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    shutil.copy2(local_tar, tmp)
    if os.path.exists(drive_archive):
        os.remove(drive_archive)
    os.replace(tmp, drive_archive)


def finalize_output(nl_path, local_path):
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
    local_forcing = os.path.join(local_data, "forcing.zarr")
    local_static = os.path.join(local_data, "static")
    nl_forcing = os.path.join(nl_data, "forcing.zarr")
    nl_static = os.path.join(nl_data, "static")
    drive_archive = os.path.join(args.drive, "data", data_name + ".tar")
    graph_name = cfg["graph"]["name"]
    local_graph = os.path.join(args.nl, "graphs", graph_name)
    project_graph = os.path.join(args.project, "graphs", graph_name)
    drive_graph = os.path.join(args.drive, "graphs", graph_name)
    local_tar = os.path.join(args.project, "data", data_name + ".tar")

    retire_invalid_store(fields_zarr, cfg)
    retire_invalid_store(drive_fields_zarr, cfg)
    retire_invalid_store(nl_fields_zarr, cfg)
    restored_from = "local"
    safety_uploaded = False
    if not valid_fields_store(fields_zarr, cfg):
        if os.path.exists(drive_archive) and \
                os.path.getsize(drive_archive) > 1_000_000_000:
            restored_from = "archive"
            staging = os.path.join(local_data, "_restore.tar")
            size_gb = os.path.getsize(drive_archive) / 1e9
            print(f"Copying data archive from Drive ({size_gb:.1f} GB)...")
            t0 = time.time()
            shutil.copy2(drive_archive, staging)
            print(f"  archive copied in {time.time() - t0:.0f}s")
            t0 = time.time()
            with tarfile.open(staging) as tar:
                tar.extractall(local_data, filter="data")
            os.remove(staging)
            print(f"  archive extracted in {time.time() - t0:.0f}s")
        elif valid_fields_store(drive_fields_zarr, cfg):
            restored_from = "legacy"
            print("Migrating fields.zarr from legacy Drive cache "
                  "(one-time; forcing/static regenerate after)...")
            copy_tree_with_progress(drive_fields_zarr, fields_zarr,
                                    "legacy fields.zarr")
        else:
            restored_from = "download"
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
    else:
        print("Data already on local disk.")
    report("fields.zarr (download)")

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
        if os.path.lexists(tmp):
            if os.path.isdir(tmp) and not os.path.islink(tmp):
                shutil.rmtree(tmp)
            else:
                os.remove(tmp)
        with xr.open_zarr(fields_zarr) as ds:
            ds = ds.chunk({"time": 512})
            enc = {}
            for v in ds.data_vars:
                if "time" in ds[v].dims:
                    enc[v] = {"chunks": (512,) + (-1,) * (ds[v].ndim - 1)}
                else:
                    enc[v] = {"chunks": ds[v].shape}
            with ProgressBar(minimum=5, dt=2):
                ds.to_zarr(tmp, mode="w", encoding=enc)
        shutil.rmtree(fields_zarr)
        os.rename(tmp, fields_zarr)
        print(f"Rechunk complete ({time.time() - t0:.0f}s).")
    report("fields.zarr (rechunk/validate)")

    if not os.path.exists(drive_archive) and restored_from != "legacy":
        print("Uploading fields-only safety archive to Drive...")
        safety_tar = os.path.join(args.project, "data", data_name + "_fields.tar")
        build_archive(safety_tar, local_data, None)
        upload_to_drive(safety_tar, drive_archive)
        safety_uploaded = True
        print("Drive safety archive ready.")

    ensure_symlink(local_forcing, nl_forcing)
    ensure_symlink(local_static, nl_static)
    if not os.path.isdir(local_graph):
        extracted_graph = os.path.join(local_data, "graphs", graph_name)
        if os.path.isdir(extracted_graph):
            print(f"Restoring graph from archive: {local_graph}")
            shutil.move(extracted_graph, local_graph)
        elif os.path.isdir(drive_graph):
            print(f"Copying graph from Drive: {local_graph}")
            copy_path(drive_graph, local_graph)

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
            finalize_output(nl_forcing, local_forcing)
        elif step in ("grid_features", "parameter_weights"):
            finalize_output(nl_static, local_static)

    if os.path.isdir(local_graph) and not os.path.lexists(project_graph):
        os.makedirs(os.path.dirname(project_graph), exist_ok=True)
        print(f"Linking graph into project dir: {project_graph}")
        os.symlink(local_graph, project_graph)

    if not os.path.exists(drive_archive) or safety_uploaded:
        print("Building full data archive and uploading to Drive...")
        build_archive(local_tar, local_data, local_graph)
        upload_to_drive(local_tar, drive_archive)
        print("Drive archive updated.")
    else:
        print("Drive archive up to date.")

    print(f"PREP FINISHED in {time.time() - t_start:.0f}s total.")


if __name__ == "__main__":
    main()
