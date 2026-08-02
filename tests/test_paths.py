import os, tempfile

import yaml

from common import REPO_DIR, SCRIPTS_DIR, CFG_PATH
import _paths
import prepare_wb2_subset


def test_find_neural_lam_sibling_layout():
    tmp = tempfile.mkdtemp(prefix="tpaths_")
    nlam = os.path.join(tmp, "neural-lam-prob-model")
    os.makedirs(os.path.join(nlam, "neural_lam"))
    module = os.path.join(tmp, "leap-graph-efm-shift", "scripts",
                          "train_shift_model.py")
    found = _paths.find_neural_lam(module, tmp)
    assert os.path.abspath(found) == os.path.abspath(nlam)


def test_find_neural_lam_repo_resolution():
    module = os.path.join(SCRIPTS_DIR, "train_shift_model.py")
    found = _paths.find_neural_lam(module, "/tmp")
    assert os.path.isabs(found)
    assert found.endswith("neural-lam-prob-model")


def test_find_neural_lam_cwd_fallback():
    tmp = tempfile.mkdtemp(prefix="tpaths_")
    nlam = os.path.join(tmp, "neural-lam-prob-model")
    os.makedirs(os.path.join(nlam, "neural_lam"))
    module = os.path.join(tmp, "elsewhere", "x.py")
    found = _paths.find_neural_lam(module, tmp)
    assert os.path.abspath(found) == os.path.abspath(nlam)


def test_train_args_balanced():
    import train_shift_model
    cfg = yaml.safe_load(open(CFG_PATH))
    cmd = train_shift_model.build_train_args(cfg, "a", "config.yaml")
    assert len(cmd) >= 2
    assert cmd[1].endswith("train_model.py")
    assert (len(cmd) - 2) % 2 == 0
    for i in range(2, len(cmd), 2):
        assert cmd[i].startswith("--")
    for flag in ("--dataset", "--model", "--config", "--graph", "--ar_steps",
                 "--batch_size", "--epochs", "--lr", "--eval_leads",
                 "--step_length", "--seed", "--ensemble_size"):
        assert flag in cmd


def test_train_model_eval_assert_allows_training():
    src = open(os.path.join(REPO_DIR, "neural-lam-patches",
                            "train_model.py")).read()
    assert '(None,) + tuple(cfg["splits"].keys())' in src
    assert "args.eval in valid_eval_splits" in src


def test_step_forcing_skips_when_present():
    tmp = tempfile.mkdtemp(prefix="tpaths_")
    forcing = os.path.join(tmp, "forcing.zarr")
    os.makedirs(forcing)
    cfg = {"dataset": {"name": "test", "forcing_zarr": forcing}}
    calls = []
    prepare_wb2_subset._run = lambda *a, **k: calls.append(a)
    prepare_wb2_subset.step_forcing(cfg)
    assert calls == []


def test_step_grid_features_runs_when_missing():
    tmp = tempfile.mkdtemp(prefix="tpaths_")
    cfg = {"dataset": {"name": "test",
                       "static_dir": os.path.join(tmp, "static")}}
    calls = []
    prepare_wb2_subset._run = lambda *a, **k: calls.append(a)
    prepare_wb2_subset.step_grid_features(cfg)
    assert len(calls) == 1
    assert calls[0][0].endswith("create_global_grid_features.py")
