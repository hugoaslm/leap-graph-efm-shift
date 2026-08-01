# Standard library
import os
import time
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
import xarray as xa
from tqdm import tqdm

# First-party
from neural_lam import constants
from neural_lam.era5_dataset import ERA5Dataset
from neural_lam.weather_dataset import WeatherDataset


def _compute_latitude_weights(lat_vals):
    """Compute normalized cosine-latitude weights for a 1-D latitude array."""
    lat_rad = np.deg2rad(lat_vals)
    weights = np.cos(lat_rad).astype(np.float32)
    return weights / weights.mean()  # mean ≈ 1.0


def main():
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_example",
        help="Dataset to compute weights for (default: meps_example)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment YAML config (for pilot experiments). "
        "If provided, overrides paper defaults via constants.load_experiment_config().",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step (for LAM only)"
        " (default: 3)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--global_dataset",
        type=int,
        default=1,
        help="Whether this is a global (ERA5) dataset (default: 1)",
    )
    args = parser.parse_args()

    # Load experiment config if provided (must happen before ERA5Dataset use)
    if args.config is not None:
        constants.load_experiment_config(args.config)

    static_dir_path = os.path.join("data", args.dataset, "static")
    os.makedirs(static_dir_path, exist_ok=True)

    global_ds = bool(args.global_dataset) or "global" in args.dataset
    cfg = constants.get_config()

    if global_ds:
        # ---- Parameter weights ----
        if cfg is not None:
            # Config-driven: equal weight for all state variables
            vert_weights = np.ones(
                constants.GRID_STATE_DIM, dtype=np.float32
            )
        else:
            # Paper: pressure-proportional for atmos levels,
            # hand-designed for surface
            pres_levels_np = np.array(
                constants.PRESSURE_LEVELS, dtype=np.float32
            )
            pres_levels_norm = pres_levels_np / pres_levels_np.sum()
            atm_weights = np.tile(
                pres_levels_norm, len(constants.ATMOSPHERIC_PARAMS)
            )
            surface_weights = np.array(
                [
                    1.0 if var_name == "2t" else 0.1
                    for var_name in constants.SURFACE_PARAMS_SHORT
                ],
                dtype=np.float32,
            )
            vert_weights = np.concatenate(
                (atm_weights, surface_weights), axis=0
            )

        # ---- Grid (spatial) weights: cosine-latitude, flattened ----
        fields_group_path = os.path.join(
            "data", args.dataset, "fields.zarr"
        )
        xds = xa.open_zarr(fields_group_path)
        lat_vals = xds.coords["latitude"].values.astype(np.float32)
        num_lon = len(xds.coords["longitude"])

        lat_weights = _compute_latitude_weights(lat_vals)  # (num_lat,)
        lat_weights_torch = torch.tensor(lat_weights, dtype=torch.float32)
        grid_weights = (
            lat_weights_torch.unsqueeze(0).repeat(num_lon, 1).flatten()
        )  # (num_grid,)
        torch.save(
            grid_weights, os.path.join(static_dir_path, "grid_weights.pt")
        )

    else:
        # Create parameter weights based on height
        # based on fig A.1 in graph cast paper
        w_dict = {
            "2": 1.0,
            "0": 0.1,
            "65": 0.065,
            "1000": 0.1,
            "850": 0.05,
            "500": 0.03,
        }
        vert_weights = np.array(
            [w_dict[par.split("_")[-2]] for par in constants.PARAM_NAMES_SHORT],
            dtype=np.float32,
        )
    print("Saving parameter weights...")
    np.save(
        os.path.join(static_dir_path, "parameter_weights.npy"), vert_weights
    )

    if global_ds and cfg is not None:
        try:
            from dask.diagnostics import ProgressBar
        except ImportError:
            ProgressBar = None
        fields_group_path = os.path.join(
            "data", args.dataset, "fields.zarr"
        )
        print("Loading training fields into memory...", flush=True)
        t0 = time.time()
        fields_xds = xa.open_zarr(fields_group_path)
        train_start, train_end = cfg["splits"]["train"]
        fields_xds = fields_xds.sel(
            time=slice(train_start, train_end)
        )
        if ProgressBar is not None:
            with ProgressBar():
                fields_xds = fields_xds.load()
        else:
            fields_xds = fields_xds.load()
        print(f"Loaded in {time.time() - t0:.0f}s.", flush=True)
        print("Computing monthly climatology...", flush=True)
        t0 = time.time()
        monthly_clim = fields_xds.groupby("time.month").mean(dim="time")
        clim_dict = {}
        for var_name in monthly_clim.data_vars:
            clim_dict[var_name] = monthly_clim[var_name].values.astype(
                np.float32
            )
        np.savez(
            os.path.join(static_dir_path, "monthly_climatology.npz"),
            **clim_dict,
        )
        print(f"  climatology done in {time.time() - t0:.0f}s", flush=True)

        print("Computing mean and std.-dev. for parameters...", flush=True)
        t0 = time.time()
        state_parts = []
        for sv in cfg["state_variables"]:
            da = fields_xds[sv["name"]]
            if sv["level"] is not None:
                da = da.sel(level=sv["level"], method="nearest")
            arr = da.values.reshape(
                -1, da.sizes["longitude"] * da.sizes["latitude"], 1
            )
            state_parts.append(arr)
        state = np.concatenate(state_parts, axis=-1).astype(np.float32)

        mean = state[2:].mean(axis=(0, 1))
        second_moment = (state[2:] ** 2).mean(axis=(0, 1))
        std = np.sqrt(np.maximum(second_moment - mean ** 2, 0))
        torch.save(
            torch.tensor(mean),
            os.path.join(static_dir_path, "parameter_mean.pt"),
        )
        torch.save(
            torch.tensor(std),
            os.path.join(static_dir_path, "parameter_std.pt"),
        )
        print(f"  mean/std done in {time.time() - t0:.0f}s", flush=True)

        print("Computing mean and std.-dev. for one-step differences...",
              flush=True)
        t0 = time.time()
        standardized = (state - mean) / std
        diffs = standardized[2:] - standardized[1:-1]
        diff_mean = diffs.mean(axis=(0, 1))
        diff_second_moment = (diffs ** 2).mean(axis=(0, 1))
        diff_std = np.sqrt(
            np.maximum(diff_second_moment - diff_mean ** 2, 0)
        )
        torch.save(
            torch.tensor(diff_mean),
            os.path.join(static_dir_path, "diff_mean.pt"),
        )
        torch.save(
            torch.tensor(diff_std),
            os.path.join(static_dir_path, "diff_std.pt"),
        )
        print(f"  diff stats done in {time.time() - t0:.0f}s", flush=True)
    else:

        if global_ds:
            ds = ERA5Dataset(
                args.dataset,
                split="train",
                pred_length=1,  # Use 1 to get each time step only once
                standardize=False,
            )
        else:
            ds = WeatherDataset(
                args.dataset,
                split="train",
                subsample_step=1,
                pred_length=63,
                standardize=False,
            )  # Without standardization
        loader = torch.utils.data.DataLoader(
            ds, args.batch_size, shuffle=False, num_workers=args.n_workers
        )
        # Compute mean and std.-dev. of each parameter (+ flux forcing)
        # across full dataset
        print("Computing mean and std.-dev. for parameters...")
        means = []
        squares = []
        flux_means = []
        flux_squares = []
        for init_batch, target_batch, forcing_batch in tqdm(loader):
            if global_ds:
                batch = target_batch  # (N_batch, N_t=1, N_grid, d_features)
            else:
                batch = torch.cat(
                    (init_batch, target_batch), dim=1
                )  # (N_batch, N_t, N_grid, d_features)
            means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
            squares.append(
                torch.mean(batch**2, dim=(1, 2))
            )  # (N_batch, d_features,)

            if not global_ds:
                # Flux at 1st windowed position is index 1 in forcing
                flux_batch = forcing_batch[:, :, :, 1]
                flux_means.append(torch.mean(flux_batch))  # (,)
                flux_squares.append(torch.mean(flux_batch**2))  # (,)

        mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
        second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
        std = torch.sqrt(second_moment - mean**2)  # (d_features)

        print("Saving mean, std.-dev, flux_stats...")
        torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
        torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))

        if not global_ds:
            flux_mean = torch.mean(torch.stack(flux_means))  # (,)
            flux_second_moment = torch.mean(torch.stack(flux_squares))  # (,)
            flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
            flux_stats = torch.stack((flux_mean, flux_std))
            torch.save(flux_stats, os.path.join(static_dir_path, "flux_stats.pt"))

        # Compute mean and std.-dev. of one-step differences across the dataset
        print("Computing mean and std.-dev. for one-step differences...")
        # Re-load dataset with standardization
        if global_ds:
            ds_standard = ERA5Dataset(
                args.dataset,
                split="train",
                pred_length=1,  # Use 1 to get each time step only once
                standardize=True,
            )
        else:
            ds_standard = WeatherDataset(
                args.dataset,
                split="train",
                subsample_step=1,
                pred_length=63,
                standardize=True,
            )
            used_subsample_len = (65 // args.step_length) * args.step_length
        loader_standard = torch.utils.data.DataLoader(
            ds_standard, args.batch_size, shuffle=False, num_workers=args.n_workers
        )

        diff_means = []
        diff_squares = []
        for init_batch, target_batch, _ in tqdm(loader_standard):
            batch = torch.cat(
                (init_batch, target_batch), dim=1
            )  # (N_batch, N_t', N_grid, d_features)

            if global_ds:
                # Only extract state at init time and target at next time
                stepped_batch = batch[:, 1:]  # (N_batch, 2, N_grid, d_features)
            else:
                # Note: batch contains only 1h-steps
                stepped_batch = torch.cat(
                    [
                        batch[:, ss_i : used_subsample_len : args.step_length]
                        for ss_i in range(args.step_length)
                    ],
                    dim=0,
                )
                # (N_batch', N_t, N_grid, d_features),
                # N_batch' = args.step_length*N_batch

            batch_diffs = stepped_batch[:, 1:] - stepped_batch[:, :-1]
            # (N_batch', N_t-1, N_grid, d_features)

            diff_means.append(
                torch.mean(batch_diffs, dim=(1, 2))
            )  # (N_batch', d_features,)
            diff_squares.append(
                torch.mean(batch_diffs**2, dim=(1, 2))
            )  # (N_batch', d_features,)

        diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
        diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
        diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

        print("Saving one-step difference mean and std.-dev...")
        torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
        torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))


if __name__ == "__main__":
    main()
