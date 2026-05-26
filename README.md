# Graph-EFM Temporal Shift

Probabilistic calibration transfer under temporal distribution shift.

A reduced-resolution pilot study adapting the Graph-EFM model (Oskarsson et
al., NeurIPS 2024) to WeatherBench 2 ERA5 64×32 data, evaluating whether
ensemble calibration fitted on earlier validation years transfers to later,
warmer test years.

## Quick Start (Google Colab)

1. Open `notebooks/graph_efm_temporal_shift_colab.ipynb` in Colab
2. Edit Cell 1: set `GITHUB_REPO` to this repo's URL, choose `RUN_PROFILE`
3. Run all cells

The notebook clones this repo and neural-lam from GitHub. Data, checkpoints,
and outputs are stored on Google Drive at `MyDrive/leap_project/`.

## What's in this repo

```
├── .gitignore
├── README.md
├── configs/
│   └── wb2_shift_64x32_graph_efm.yaml   # Experiment configuration
├── scripts/
│   ├── download_wb2_data.py              # WB2 data download (Colab)
│   ├── prepare_wb2_subset.py             # Preprocessing orchestrator
│   ├── train_shift_model.py              # Two-phase training wrapper
│   ├── evaluate_shift.py                 # Ensemble eval + calibration
│   └── plot_shift_results.py             # 5 publication figures
├── notebooks/
│   └── graph_efm_temporal_shift_colab.ipynb
└── neural-lam-patches/                   # Modified files applied over
    ├── train_model.py                    #   cloned neural-lam at runtime
    ├── create_global_forcing.py
    ├── create_parameter_weights.py
    └── neural_lam/
        ├── constants.py                  # Config-driven dynamic dimensions
        ├── era5_dataset.py               # Custom splits + variable selection
        ├── forecast_to_xarr.py           # Config-mode guards
        └── metrics.py                    # Ensemble evaluation helpers
```

## Research Question

> Does a small probabilistic Graph-EFM model become less well calibrated on
> later ERA5 years than on earlier held-out years, after calibration has been
> fitted on an earlier validation period?

- **Model:** Graph-EFM (hierarchical graph neural ensemble forecasting)
- **Data:** WeatherBench 2 ERA5 64×32, 1979–2022, 6-hourly
- **Variables:** z500, t850, t2m (3 output channels)
- **Training:** 1979–2005, single-step → 4-step autoregressive
- **Calibration:** Validation 2006–2010, tested on ID 2011–2015 and OOD
  2016–2022

## Credits

- **Graph-EFM:** Oskarsson et al., "Probabilistic Weather Forecasting with
  Hierarchical Graph Neural Networks", NeurIPS 2024
- **Neural-LAM:** https://github.com/mllam/neural-lam
- **WeatherBench 2:** https://weatherbench2.readthedocs.io/
