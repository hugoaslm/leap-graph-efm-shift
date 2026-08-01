# Graph-EFM Temporal Shift

Probabilistic calibration transfer under temporal distribution shift.

A small Graph-EFM model (Oskarsson et al., NeurIPS 2024) is trained on
WeatherBench 2 ERA5 64x32 (z500, t850, t2m). Ensemble calibration is fitted
on an early validation period and tested on later, warmer years.

## Layout

- `configs/` — experiment configuration (grid, variables, splits, hyperparameters)
- `scripts/` — data download, preprocessing, training, evaluation, plotting
- `notebooks/` — end-to-end Colab pipeline
- `neural-lam-patches/` — runtime overlay applied on top of neural-lam (branch `prob_model_global`)

## Reproduce (Colab)

1. Open `notebooks/graph_efm_temporal_shift_colab.ipynb` in Colab.
2. Run all cells.

The notebook clones this repository and neural-lam, downloads the WeatherBench 2
subset, and stores checkpoints, results, and figures on Google Drive
(`MyDrive/leap_project/`).

## Data and splits

- Data: WeatherBench 2 ERA5 64x32, 1979–2022, 6-hourly
- Variables: z500, t850, t2m
- Training: 1979–2005
- Calibration: 2006–2010 (validation)
- Test: 2011–2015 (in-distribution), 2016–2022 (out-of-distribution)

## References

- Oskarsson et al. "Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks." NeurIPS 2024.
- Neural-LAM: https://github.com/mllam/neural-lam
- WeatherBench 2: https://weatherbench2.readthedocs.io/
