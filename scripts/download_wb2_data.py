import os, shutil, subprocess, sys, time
from argparse import ArgumentParser

import numpy as np
import xarray as xr

REQUIRED_VARS = [
    "geopotential", "temperature", "2m_temperature",
    "geopotential_at_surface", "land_sea_mask",
]

WB2_64x32_SOURCE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"


def download_with_gsutil(source, dest):
    subprocess.run(["gsutil", "-m", "cp", "-r", source, dest], check=True)


def download_with_xarray(source, dest, time_start=None, time_end=None,
                         colab_auth=False, slice_years=2):
    if colab_auth:
        try:
            from google.colab import auth
            auth.authenticate_user()
        except Exception as exc:
            print(f"Google auth failed ({exc}); using anonymous access.")

    for token in ["anon", None]:
        try:
            kwargs = {}
            if "://" in source:
                kwargs["storage_options"] = {"token": token}
            ds = xr.open_zarr(source, consolidated=True, **kwargs)
            break
        except Exception:
            if token is None:
                raise
            print("Anonymous access failed, trying default credentials...")
    missing = set(REQUIRED_VARS) - set(ds.data_vars)
    if missing:
        raise RuntimeError(
            "WeatherBench source is missing required variables: "
            f"{sorted(missing)}. Available variables: {list(ds.data_vars)}"
        )
    ds = ds[REQUIRED_VARS]
    if "level" in ds.dims:
        ds = ds.sel(level=[500, 850], method="nearest")
    if time_start is not None or time_end is not None:
        ds = ds.sel(time=slice(time_start, time_end))

    ds = ds.transpose(
        "time", "longitude", "latitude", "level", missing_dims="ignore"
    )

    n_time = ds.sizes["time"]
    if n_time == 0:
        raise RuntimeError(f"Empty time selection ({time_start}..{time_end})")

    encoding = {}
    for var in ds.data_vars:
        shape = ds[var].shape
        if len(shape) >= 3:
            encoding[var] = {"chunks": (1,) + tuple(-1 for _ in shape[1:])}

    bytes_per_time = 0
    for var in ds.data_vars:
        nbytes_per_elem = ds[var].dtype.itemsize
        per_time = int(np.prod(
            [ds[var].sizes[k] for k in ds[var].dims if k != "time"]
        ))
        bytes_per_time += nbytes_per_elem * per_time
    est_gb = bytes_per_time * n_time / 1e9

    dest_dir = os.path.dirname(dest) or "."
    free_gb = shutil.disk_usage(dest_dir).free / 1e9
    print(f"Selected subset: {n_time} timesteps, ~{est_gb:.1f} GB "
          f"(available disk: {free_gb:.1f} GB)", flush=True)
    if est_gb > 0.85 * free_gb:
        raise RuntimeError(
            f"Estimated download ({est_gb:.1f} GB) exceeds 85% of free disk "
            f"({free_gb:.1f} GB)."
        )

    slice_len = max(1, int(slice_years * 4 * 365))
    start = 0
    first = True
    w_start = time.time()
    if os.path.isdir(dest):
        try:
            with xr.open_zarr(dest, consolidated=True) as prev:
                n_prev = prev.sizes.get("time", 0)
                if n_prev > 0:
                    matches = (n_prev <= n_time and np.array_equal(
                        prev.time.values,
                        ds.time.isel(time=slice(0, n_prev)).values))
                    if n_prev >= n_time:
                        print(f"Store already complete ({n_prev} timesteps).",
                              flush=True)
                        return
                    if matches:
                        start = n_prev
                        first = False
                        print(f"Resuming: {n_prev}/{n_time} timesteps already "
                              "present.", flush=True)
        except Exception:
            pass

    for t0 in range(start, n_time, slice_len):
        t1 = min(n_time, t0 + slice_len)
        sub = ds.isel(time=slice(t0, t1))
        if first:
            sub.to_zarr(dest, mode="w", encoding=encoding, consolidated=True)
            first = False
        else:
            sub.to_zarr(dest, mode="a", append_dim="time", consolidated=True)
        frac = (t1 - start) / (n_time - start)
        eta = ""
        if t0 > start and time.time() > w_start:
            rate = (t0 - start) / (time.time() - w_start)
            remaining_s = (n_time - t1) / rate if rate > 0 else 0
            eta = f", ~{remaining_s / 60:.0f} min left"
        print(f"  wrote timesteps {t0 + 1}-{t1} of {n_time} "
              f"({frac:.0%}{eta})", flush=True)
    print(f"Saved {dest} (~{est_gb:.1f} GB)", flush=True)


def main():
    parser = ArgumentParser(description="Download WB2 ERA5 64x32 subset")
    parser.add_argument("--output", default="data/wb2_era5_64x32/fields.zarr")
    parser.add_argument("--source", default=WB2_64x32_SOURCE)
    parser.add_argument("--time_start", default="1979-01-01")
    parser.add_argument("--time_end", default="2022-12-31")
    parser.add_argument("--method", default="xarray", choices=["xarray", "gsutil"])
    parser.add_argument("--slice_years", type=int, default=2)
    parser.add_argument("--colab_auth", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.method == "gsutil":
        download_with_gsutil(args.source, args.output)
    else:
        download_with_xarray(args.source, args.output, args.time_start, args.time_end,
                             colab_auth=args.colab_auth,
                             slice_years=args.slice_years)

    ds = xr.open_zarr(args.output, consolidated=True)
    print(f"Time: {ds.time.values[0]} .. {ds.time.values[-1]}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Grid: {ds.sizes.get('longitude','?')} x {ds.sizes.get('latitude','?')}")


if __name__ == "__main__":
    main()
