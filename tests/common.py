import os, sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import xarray as xr

CFG_PATH = os.path.join(REPO_DIR, "configs", "wb2_shift_64x32_graph_efm.yaml")


def make_fields(store, nt=100, grid=(4, 2), time_chunk=None):
    lon, lat = grid
    ds = xr.Dataset({
        "2m_temperature": (("time", "longitude", "latitude"),
                           np.zeros((nt, lon, lat))),
        "temperature": (("time", "longitude", "latitude"),
                        np.zeros((nt, lon, lat))),
        "geopotential": (("time", "longitude", "latitude"),
                         np.zeros((nt, lon, lat))),
        "geopotential_at_surface": (("longitude", "latitude"),
                                    np.zeros((lon, lat))),
        "land_sea_mask": (("longitude", "latitude"), np.zeros((lon, lat))),
    }, coords={"time": np.arange(nt),
               "longitude": np.arange(lon), "latitude": np.arange(lat)})
    enc = {}
    for v in ds.data_vars:
        if "time" in ds[v].dims:
            c = time_chunk if time_chunk is not None else ds[v].shape[0]
            enc[v] = {"chunks": (c,) + (-1,) * (ds[v].ndim - 1)}
        else:
            enc[v] = {"chunks": ds[v].shape}
    ds.to_zarr(store, mode="w", encoding=enc)
    return ds


def make_fields_slice(store, t0, t1, grid=(4, 2), value=0.0, mode="w"):
    lon, lat = grid
    nt = t1 - t0
    ds = xr.Dataset({
        "2m_temperature": (("time", "longitude", "latitude"),
                           np.full((nt, lon, lat), value, dtype=np.float32)),
        "temperature": (("time", "longitude", "latitude"),
                        np.full((nt, lon, lat), value, dtype=np.float32)),
        "geopotential": (("time", "longitude", "latitude"),
                         np.full((nt, lon, lat), value, dtype=np.float32)),
        "geopotential_at_surface": (("longitude", "latitude"),
                                    np.zeros((lon, lat))),
        "land_sea_mask": (("longitude", "latitude"), np.zeros((lon, lat))),
    }, coords={"time": np.arange(t0, t1),
               "longitude": np.arange(lon), "latitude": np.arange(lat)})
    enc = {}
    for v in ds.data_vars:
        if "time" in ds[v].dims:
            enc[v] = {"chunks": (1,) + (-1,) * (ds[v].ndim - 1)}
        else:
            enc[v] = {"chunks": ds[v].shape}
    if mode == "a":
        ds.to_zarr(store, mode="a", append_dim="time")
    else:
        ds.to_zarr(store, mode="w", encoding=enc)
    return ds
