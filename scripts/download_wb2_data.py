import os, subprocess, sys
from argparse import ArgumentParser

import xarray as xr

# State variables + static fields needed for grid features.
REQUIRED_VARS = [
    "geopotential", "temperature", "2m_temperature",
    "geopotential_at_surface", "land_sea_mask",
]

WB2_64x32_SOURCE = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32.zarr/"


def download_with_gsutil(source, dest):
    print(f"Copying {source} -> {dest}")
    subprocess.run(["gsutil", "-m", "cp", "-r", source, dest], check=True)


def download_with_xarray(source, dest, time_start=None, time_end=None, colab_auth=False):
    if colab_auth:
        from google.colab import auth
        auth.authenticate_user()

    # Try anonymous access first; fall back to default credentials
    for token in ["anon", None]:
        try:
            ds = xr.open_zarr(source, consolidated=True,
                              storage_options={"token": token})
            break
        except Exception as e:
            if token is None:
                raise
            print(f"Anonymous access failed, trying default credentials...")
    available = [v for v in REQUIRED_VARS if v in ds.data_vars]
    missing = set(REQUIRED_VARS) - set(available)
    if missing:
        print(f"Variables not found: {missing}")
    ds = ds[available]
    if "level" in ds.dims:
        ds = ds.sel(level=[500, 850], method="nearest")
    if time_start is not None or time_end is not None:
        ds = ds.sel(time=slice(time_start, time_end))

    encoding = {}
    for var in ds.data_vars:
        shape = ds[var].shape
        if len(shape) >= 3:
            encoding[var] = {"chunks": (1,) + tuple(-1 for _ in shape[1:])}
    ds.to_zarr(dest, mode="w", encoding=encoding)
    print(f"Saved {dest} ({ds.nbytes/1e9:.1f} GB)")


def main():
    parser = ArgumentParser(description="Download WB2 ERA5 64x32 subset")
    parser.add_argument("--output", default="data/wb2_era5_64x32/fields.zarr")
    parser.add_argument("--source", default=WB2_64x32_SOURCE)
    parser.add_argument("--time_start", default="1979-01-01")
    parser.add_argument("--time_end", default="2022-12-31")
    parser.add_argument("--method", default="xarray", choices=["xarray", "gsutil"])
    parser.add_argument("--colab_auth", action="store_true",
                        help="Authenticate with Google before accessing GCS")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.method == "gsutil":
        download_with_gsutil(args.source, args.output)
    else:
        download_with_xarray(args.source, args.output, args.time_start, args.time_end,
                             colab_auth=args.colab_auth)

    ds = xr.open_zarr(args.output)
    print(f"Time: {ds.time.values[0]} .. {ds.time.values[-1]}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Grid: {ds.sizes.get('longitude','?')} x {ds.sizes.get('latitude','?')}")


if __name__ == "__main__":
    main()
