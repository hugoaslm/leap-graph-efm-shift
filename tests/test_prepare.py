import os, shutil, tarfile, tempfile

import numpy as np
import xarray as xr

from common import make_fields, make_fields_slice
import prepare_colab_data as pcd

CFG = {"grid": {"shape": [4, 2]}}


def test_valid_fields_store():
    tmp = tempfile.mkdtemp(prefix="tprep_")
    fields = os.path.join(tmp, "fields.zarr")
    make_fields(fields, time_chunk=512)
    assert pcd.valid_fields_store(fields, CFG)
    assert not pcd.needs_rechunk(fields)


def test_needs_rechunk_time1():
    tmp = tempfile.mkdtemp(prefix="tprep_")
    fields = os.path.join(tmp, "fields.zarr")
    make_fields(fields, time_chunk=1)
    assert pcd.needs_rechunk(fields)


def test_rechunk_roundtrip():
    tmp = tempfile.mkdtemp(prefix="tprep_")
    fields = os.path.join(tmp, "fields.zarr")
    make_fields(fields, time_chunk=1, nt=64)
    with xr.open_zarr(fields) as d:
        expected = d["temperature"].isel(time=slice(0, 4)).values.copy()
    d = xr.open_zarr(fields).chunk({"time": 512})
    enc = {v: {"chunks": (512,) + (-1,) * (d[v].ndim - 1)}
           if "time" in d[v].dims else {"chunks": d[v].shape}
           for v in d.data_vars}
    d.to_zarr(fields + ".r", mode="w", encoding=enc)
    shutil.rmtree(fields)
    os.rename(fields + ".r", fields)
    assert not pcd.needs_rechunk(fields)
    with xr.open_zarr(fields) as d:
        assert np.allclose(d["temperature"].isel(time=slice(0, 4)).values,
                           expected)


def test_append_time1_chunks():
    tmp = tempfile.mkdtemp(prefix="tprep_")
    fields = os.path.join(tmp, "fields.zarr")
    make_fields_slice(fields, 0, 40, value=1.0)
    make_fields_slice(fields, 40, 64, value=2.0, mode="a")
    with xr.open_zarr(fields) as d:
        assert d.sizes["time"] == 64
        assert np.allclose(d["temperature"].isel(time=5).values, 1.0)
        assert np.allclose(d["temperature"].isel(time=41).values, 2.0)


def test_archive_roundtrip():
    tmp = tempfile.mkdtemp(prefix="tprep_")
    local_data = os.path.join(tmp, "local")
    os.makedirs(os.path.join(local_data, "static"))
    open(os.path.join(local_data, "static", "grid_features.pt"), "w").write("x")
    make_fields(os.path.join(local_data, "fields.zarr"), time_chunk=512)
    graphs = os.path.join(tmp, "graphs", "g1")
    os.makedirs(graphs)
    open(os.path.join(graphs, "m2m_edge_index.pt"), "w").write("y")
    tar = os.path.join(tmp, "cache", "ds.tar")
    os.makedirs(os.path.dirname(tar))
    pcd.build_archive(tar, local_data, graphs)
    tops = sorted({n.split("/")[0] for n in tarfile.open(tar).getnames()})
    assert tops == ["fields.zarr", "graphs", "static"]
    restored = os.path.join(tmp, "restored")
    os.makedirs(restored)
    with tarfile.open(tar) as t:
        t.extractall(restored, filter="data")
    assert pcd.valid_fields_store(os.path.join(restored, "fields.zarr"), CFG)
    assert os.path.exists(os.path.join(restored, "static",
                                       "grid_features.pt"))
