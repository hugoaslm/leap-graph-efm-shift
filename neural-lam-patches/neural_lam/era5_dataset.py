# Standard library
import os

# Third-party
import numpy as np
import torch
import xarray as xa

# First-party
from neural_lam import constants, utils


class ERA5Dataset(torch.utils.data.Dataset):
    """
    Dataset loading ERA5 from Zarr.

    Supports both the original paper-scale setup (83 variables, hard-coded
    splits) and a config-driven pilot setup when ``constants.load_experiment_config()``
    has been called before instantiation.
    """

    def __init__(
        self,
        dataset_name,
        pred_length=40,
        split="train",
        standardize=True,
        expanded_test=False,
        **kwarg,  # pylint: disable=unused-argument
    ):
        super().__init__()

        self._config = constants.get_config()

        # ---- Determine valid split names ----
        if self._config is not None:
            valid_splits = tuple(self._config["splits"].keys())
        else:
            valid_splits = ("train", "val", "test")
        assert split in valid_splits, (
            f"Unknown dataset split '{split}'; valid: {valid_splits}"
        )

        # ---- Open zarrs ----
        fields_path = os.path.join("data", dataset_name, "fields.zarr")
        fields_xds = xa.open_zarr(fields_path)
        forcing_path = os.path.join("data", dataset_name, "forcing.zarr")
        forcing_xda = xa.open_dataarray(forcing_path, engine="zarr")

        # ---- Build split time slices ----
        split_slices = self._build_split_slices(
            dataset_name, split, expanded_test
        )
        fields_ds_split = fields_xds.sel(time=split_slices[split])
        forcing_ds_split = forcing_xda.sel(time=split_slices[split])

        # Compute dataset length
        timesteps_in_split = len(fields_ds_split.coords["time"])
        self.pred_length = pred_length
        # -1 for AR-2, - pred_length for target states
        ds_timesteps = timesteps_in_split - 1 - pred_length
        assert ds_timesteps > 0, (
            f"Dataset too small for pred_length={pred_length} "
            f"({ds_timesteps} valid init timesteps in split '{split}')"
        )
        if split == "train":
            self.ds_len = ds_timesteps
            self.init_all = True
        else:  # val, test, id, ood
            # Init only from 00/12 UTC
            self.ds_len = int(np.ceil(ds_timesteps / 2))
            self.init_all = False

        # ---- Standardization ----
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean = ds_stats["data_mean"]
            self.data_std = ds_stats["data_std"]

        # ---- Build the ordered list of (var_name, level_or_None) that
        #      defines the state dimension ----
        if self._config is not None:
            self._state_var_specs = [
                (sv["name"], sv["level"])
                for sv in self._config["state_variables"]
            ]
            self._use_config_vars = True
        else:
            # Paper mode: all atmos params at all levels, then all surface
            self._use_config_vars = False

        # Pre-load zarr data arrays
        if self._use_config_vars:
            # Load each (var, level) pair individually
            self._state_arrays = []
            for var_name, level in self._state_var_specs:
                if level is not None:
                    da = (
                        fields_ds_split[var_name]
                        .sel(level=level, method="nearest")
                    )
                else:
                    da = fields_ds_split[var_name]
                # da shape: (time, longitude, latitude) or
                #           (time, longitude, latitude) for surface
                self._state_arrays.append(da)
        else:
            # Original paper behaviour
            self.atm_xda = (
                fields_ds_split[constants.ATMOSPHERIC_PARAMS]
                .to_dataarray("state_var")
                .transpose(
                    "time", "longitude", "latitude", "state_var", "level"
                )
            )
            self.surface_xda = (
                fields_ds_split[constants.SURFACE_PARAMS]
                .to_dataarray("state_var")
                .transpose("time", "longitude", "latitude", "state_var")
            )
            self.atm_total_dim = len(self.atm_xda.coords["level"]) * len(
                self.atm_xda.coords["state_var"]
            )
            self.surface_total_dim = len(self.surface_xda.coords["state_var"])

        # Forcing
        self.forcing_xda = forcing_ds_split

    # ------------------------------------------------------------------
    def _build_split_slices(self, dataset_name, split, expanded_test):
        """Return a dict split_name -> slice for time selection."""
        if self._config is not None:
            # Config-driven splits
            split_dates = self._config["splits"]
            slices = {}
            for split_key, (start, end) in split_dates.items():
                # ERA5 6-hourly: ensure time strings include a time part
                s_str = start if "T" in start else f"{start}T00"
                e_str = end if "T" in end else f"{end}T18"
                slices[split_key] = slice(s_str, e_str)
            return slices

        # ---- Original paper splits ----
        if "example" in dataset_name:
            return {
                "train": slice("1959-01-01T12", "1959-01-03T12"),
                "val": slice("1959-01-03T18", "1959-01-04T18"),
                "test": slice("1959-01-03T18", "1959-01-04T18"),
            }

        slices = {
            "train": slice("1959-01-01T12", "2017-12-31T12"),
            "val": slice("2017-12-31T18", "2019-12-31T12"),
        }
        if expanded_test:
            slices["test"] = slice("2019-12-31T18", "2023-12-31T18")
        else:
            slices["test"] = slice("2019-12-31T18", "2021-01-10T18")
        return slices

    # ------------------------------------------------------------------
    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        # Forecast t=(s+1):(s+pred_length) from init states at t=s-1,s
        if self.init_all:
            init_i = idx + 1  # s = idx+1
        else:
            init_i = 1 + idx * 2  # s = 1 + 2idx
        sample_slice = slice(init_i - 1, init_i + self.pred_length + 1)
        full_series_len = self.pred_length + 2

        # === State ===
        if self._use_config_vars:
            # Stack config-specified variables in order
            state_parts = []
            for da in self._state_arrays:
                arr = da[sample_slice].to_numpy()  # (time, lon, lat)
                # Flatten spatial dims
                arr_flat = arr.reshape(full_series_len, -1, 1)
                state_parts.append(arr_flat)
            full_state_np = np.concatenate(
                state_parts, axis=-1
            )  # (2+pred_length, num_grid, d_state)
        else:
            # Original paper code path
            atm_sample_np = self.atm_xda[sample_slice].to_numpy()
            surface_sample_np = self.surface_xda[sample_slice].to_numpy()
            full_state_np = np.concatenate(
                (
                    atm_sample_np.reshape(
                        (full_series_len, -1, self.atm_total_dim)
                    ),
                    surface_sample_np.reshape(
                        (full_series_len, -1, self.surface_total_dim)
                    ),
                ),
                axis=-1,
            )

        # Convert to torch + standardize
        full_state_torch = torch.tensor(full_state_np, dtype=torch.float32)
        if self.standardize:
            full_state_torch = (
                full_state_torch - self.data_mean
            ) / self.data_std

        init_states = full_state_torch[:2]
        target_states = full_state_torch[2:]

        # === Forcing features ===
        forcing_np = self.forcing_xda[sample_slice].to_numpy()
        forcing_flat_np = forcing_np.reshape(
            full_series_len, -1, forcing_np.shape[-1]
        )
        forcing_windowed = np.concatenate(
            (
                forcing_flat_np[:-2],
                forcing_flat_np[1:-1],
                forcing_flat_np[2:],
            ),
            axis=2,
        )
        forcing_torch = torch.tensor(forcing_windowed, dtype=torch.float32)

        return init_states, target_states, forcing_torch
