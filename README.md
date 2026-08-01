# Graph-EFM Temporal Shift

Probabilistic calibration transfer under temporal distribution shift.

A reduced-resolution pilot study adapting the Graph-EFM model (Oskarsson et
al., NeurIPS 2024) to WeatherBench 2 ERA5 64×32 data, evaluating whether
ensemble calibration fitted on earlier validation years transfers to later,
warmer test years.

## Quick Start (Google Colab)

1. Open `notebooks/graph_efm_temporal_shift_colab.ipynb` in Colab
2. In Cell 1, `GITHUB_REPO` is pre-filled with
   `https://github.com/hugoaslm/leap-graph-efm-shift.git` — just choose
   `RUN_PROFILE` if the default `l4_core` doesn't match your runtime
3. Run all cells

The notebook clones this repo and neural-lam from GitHub. Data, checkpoints,
and outputs are stored on Google Drive at `MyDrive/leap_project/`.

## Repository

- `configs/wb2_shift_64x32_graph_efm.yaml` — experiment configuration (grid, variables, splits, model hyperparameters)
- `scripts/download_wb2_data.py` — WeatherBench 2 data download
- `scripts/prepare_wb2_subset.py` — preprocessing (forcing, grid features, mesh, statistics)
- `scripts/train_shift_model.py` — two-phase training wrapper
- `scripts/evaluate_shift.py` — ensemble evaluation with post-hoc calibration
- `scripts/plot_shift_results.py` — generates the five publication figures
- `notebooks/graph_efm_temporal_shift_colab.ipynb` — Colab notebook that runs the full pipeline
- `neural-lam-patches/` — modifications applied over the upstream neural-lam `prob_model_global` branch at runtime

## Research question

> Does a small probabilistic Graph-EFM model become less well calibrated on
> later ERA5 years than on earlier held-out years, after calibration has been
> fitted on an earlier validation period?

- **Model:** Graph-EFM (hierarchical graph neural ensemble forecasting)
- **Data:** WeatherBench 2 ERA5 64×32, 1979–2022, 6-hourly
- **Variables:** z500, t850, t2m (3 output channels)
- **Training:** 1979–2005, single-step → 4-step autoregressive
- **Calibration:** Validation 2006–2010, tested on ID 2011–2015 and OOD
  2016–2022

## References

- Oskarsson et al., "Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks", NeurIPS 2024.
- Neural-LAM repository: https://github.com/mllam/neural-lam
- WeatherBench 2: https://weatherbench2.readthedocs.io/
