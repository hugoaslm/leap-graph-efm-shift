#!/usr/bin/env python3
"""
Download the WeatherBench 2 ERA5 64×32 subset for the temporal-shift experiment.

This script downloads only the variables and time range needed for the pilot
study and saves them as a local zarr store. Designed to run on Google Colab
(where ``gsutil`` is available) or any machine with GCS access.

Usage:
    python scripts/download_wb2_data.py \\
        --output data/wb2_era5_64x32/fields.zarr \\
        --time_start 1979-01-01 \\
        --time_end 2022-12-31
"""
from __future__ import annotations

import os
import subprocess
import sys
from argparse import ArgumentParser

import xarray as xr

# Variables needed for the reduced experiment.
# geopotential@500hPa, temperature@850hPa, 2m_temperature (surface)
# Plus static fields needed for grid features:
#   geopotential_at_surface, land_sea_mask
REQUIRED_VARS = [
    "geopotential",
    "temperature",
    "2m_temperature",
    "geopotential_at_surface",
    "land_sea_mask",
]

WB2_64x32_SOURCE = (
    "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32.zarr/"
)


def download_with_gsutil(source: str, dest: str) -> None:
    """Use gsutil to copy a zarr store from GCS."""
    print(f"Downloading {source} -> {dest} ...")
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", source, dest],
        check=True,
    )
    print("Download complete.")


def download_with_xarray(
    source: str,
    dest: str,
    time_start: str | None = None,
    time_end: str | None = None,
) -> None:
    """
    Open the remote zarr with xarray, select variables and time slice,
    then write a local copy.
    """
    print(f"Opening remote zarr: {source}")
    ds = xr.open_zarr(source, consolidated=True)

    # Select only the variables we need
    available = [v for v in REQUIRED_VARS if v in ds.data_vars]
    missing = set(REQUIRED_VARS) - set(available)
    if missing:
        print(f"Warning: variables not found in zarr: {missing}")
    ds = ds[available]

    # Select required pressure levels (geopotential@500, temperature@850)
    if "level" in ds.dims:
        ds = ds.sel(level=[500, 850], method="nearest")

    # Time slice
    if time_start is not None or time_end is not None:
        ds = ds.sel(time=slice(time_start, time_end))

    print(f"Saving to {dest} ...")
    # Use reasonable chunking
    encoding = {}
    for var in ds.data_vars:
        shape = ds[var].shape
        if len(shape) >= 3:
            # time, lon, lat, [level] -> chunk time=1, rest full
            chunks = (1,) + tuple(-1 for _ in shape[1:])
            encoding[var] = {"chunks": chunks}

    ds.to_zarr(dest, mode="w", encoding=encoding)
    print(f"Saved {dest} ({ds.nbytes / 1e9:.2f} GB)")


def main():
    parser = ArgumentParser(
        description="Download WeatherBench 2 ERA5 64×32 subset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wb2_era5_64x32/fields.zarr",
        help="Local path for the output zarr store",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=WB2_64x32_SOURCE,
        help="Remote zarr path (GCS or local)",
    )
    parser.add_argument(
        "--time_start",
        type=str,
        default="1979-01-01",
        help="Start date (ISO format, inclusive)",
    )
    parser.add_argument(
        "--time_end",
        type=str,
        default="2022-12-31",
        help="End date (ISO format, inclusive)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="xarray",
        choices=["xarray", "gsutil"],
        help="Download method: 'xarray' (selective, slower) or "
        "'gsutil' (full copy, faster if full zarr is needed)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.method == "gsutil":
        download_with_gsutil(args.source, args.output)
    else:
        download_with_xarray(
            args.source,
            args.output,
            time_start=args.time_start,
            time_end=args.time_end,
        )

    # Verify
    ds = xr.open_zarr(args.output)
    print(f"\nVerification — downloaded dataset:")
    print(f"  Time range: {ds.time.values[0]} .. {ds.time.values[-1]}")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Grid: {ds.sizes.get('longitude', '?')}×"
          f"{ds.sizes.get('latitude', '?')}")


if __name__ == "__main__":
    main()
