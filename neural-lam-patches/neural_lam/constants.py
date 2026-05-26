# Standard library
import os

# Third-party
import cartopy
import numpy as np
import yaml

WANDB_PROJECT = "neural-lam"

# ---------------------------------------------------------------------------
# Default (paper-scale) constants — overridden when load_experiment_config()
# is called with a pilot experiment YAML.
# ---------------------------------------------------------------------------

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 5, 10, 20, 40])
# Also save checkpoints for minimum loss at these lead times
VAL_STEP_CHECKPOINTS = np.array([1, 20, 40])

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = [
    "val_spsk_ratio",
    "val_rmse",
]
# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {
    7: [1, 10],  # z500
    78: [1, 10],  # 2t
    79: [1, 10],  # 10u
}

# Plot forecasts for these variables at given lead times during validation step
# Format is a dictionary that maps from a variable index to a list of
# lead time steps
VAL_PLOT_VARS = {
    7: np.array([2, 20]),  # z500
    22: np.array([2, 20]),  # q700
    36: np.array([2, 20]),  # t850
    78: np.array([2, 20]),  # 2t
    79: np.array([2, 20]),  # 10u
    80: np.array([2, 20]),  # 10v
    82: np.array([2, 20]),  # tp
}

# During validation, plot example samples of latent variable from prior and
# variational distribution
LATENT_SAMPLES_PLOT = 4  # Number of samples to plot

# Following table 2 in GC
# Keys to read from fields zarr
ATMOSPHERIC_PARAMS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]  # times 13 pressure levels = 78 params

SURFACE_PARAMS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]  # = 5 params
# Total = 83 params

# Variable names
ATMOSPHERIC_PARAMS_SHORT = [
    "z",
    "q",
    "t",
    "u",
    "v",
    "w",
]
SURFACE_PARAMS_SHORT = ["2t", "10u", "10v", "msl", "tp"]
PRESSURE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]  # 13 levels
PARAM_NAMES_SHORT = [
    f"{param}{level}"
    for param in ATMOSPHERIC_PARAMS_SHORT
    for level in PRESSURE_LEVELS
] + SURFACE_PARAMS_SHORT

ATMOSPHERIC_PARAMS_UNITS = [
    "m²/s²",
    "kg/kg",
    "K",
    "m/s",
    "m/s",
    "Pa/s",
]
PARAM_UNITS = [
    unit for unit in ATMOSPHERIC_PARAMS_UNITS for level in PRESSURE_LEVELS
] + ["K", "m/s", "m/s", "Pa", "m"]

# What variables (index) to plot during evaluation

EVAL_PLOT_VARS = np.concatenate(
    [
        level_start_i
        + np.arange(0, len(ATMOSPHERIC_PARAMS)) * len(PRESSURE_LEVELS)
        for level_start_i in (
            PRESSURE_LEVELS.index(level) for level in (200, 500, 850)
        )
    ]
    + [np.arange(78, 83)]  # Surface
)

# Projection and grid
GRID_SHAPE = (240, 121)  # (long, lat)

# Create projection
MAP_PROJ = cartopy.crs.Robinson()
GRID_LIMITS = [
    -0.75,
    359.25,
    -90,
    90,
]

# Time step length (hours)
TIME_STEP_LENGTH = 6

# Data dimensions
GRID_ORIGINAL_FORCING_DIM = 5  # 5 features
GRID_FORCING_DIM = GRID_ORIGINAL_FORCING_DIM * 3
# 5 features for 3 time-step window
GRID_STATE_DIM = 6 * 13 + 5  # 83

# ---------------------------------------------------------------------------
# Experiment config loading — call once at startup for pilot experiments
# ---------------------------------------------------------------------------

_CONFIG = None  # Holds the parsed YAML dict after load_experiment_config()
_DEFAULT_VAR_UNITS = {
    "geopotential": "m²/s²",
    "specific_humidity": "kg/kg",
    "temperature": "K",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
    "vertical_velocity": "Pa/s",
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m/s",
    "10m_v_component_of_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation_6hr": "m",
}


def get_config():
    """Return the loaded experiment config dict, or None."""
    return _CONFIG


def load_experiment_config(config_path):
    """
    Load a pilot-experiment YAML config and override module-level constants.

    Must be called BEFORE any other module imports that capture constants
    by value (e.g. ``from neural_lam.constants import GRID_SHAPE``).
    All neural-lam code accesses constants via ``constants.GRID_SHAPE``
    (qualified name), so overriding the module attribute is sufficient.

    If no config is loaded, the original paper-scale defaults remain active.
    """
    # pylint: disable=global-statement
    global _CONFIG
    global GRID_SHAPE, GRID_STATE_DIM
    global ATMOSPHERIC_PARAMS, SURFACE_PARAMS, PRESSURE_LEVELS
    global ATMOSPHERIC_PARAMS_SHORT, SURFACE_PARAMS_SHORT
    global PARAM_NAMES_SHORT, PARAM_UNITS
    global EVAL_PLOT_VARS, VAL_PLOT_VARS, VAR_LEADS_METRICS_WATCH
    global VAL_STEP_LOG_ERRORS, VAL_STEP_CHECKPOINTS
    global GRID_ORIGINAL_FORCING_DIM, GRID_FORCING_DIM

    with open(config_path, "r", encoding="utf-8") as fh:
        _CONFIG = yaml.safe_load(fh)

    cfg = _CONFIG

    # ---- Grid ----
    GRID_SHAPE = tuple(cfg["grid"]["shape"])  # (lon, lat)

    # ---- State variables ----
    state_vars = cfg["state_variables"]
    GRID_STATE_DIM = len(state_vars)

    # Determine which ERA5 variable names are atmospheric vs surface,
    # and which pressure levels are present.
    atmos_set = {}
    surface_set = {}
    levels_set = set()
    for sv in state_vars:
        if sv["level"] is not None:
            atmos_set[sv["name"]] = True
            levels_set.add(sv["level"])
        else:
            surface_set[sv["name"]] = True

    ATMOSPHERIC_PARAMS = sorted(atmos_set.keys())
    SURFACE_PARAMS = sorted(surface_set.keys())
    PRESSURE_LEVELS = sorted(levels_set)

    # Short names and units per state-variable index
    state_short_names = [sv["short_name"] for sv in state_vars]
    state_units = [
        _DEFAULT_VAR_UNITS.get(sv["name"], "unknown") for sv in state_vars
    ]

    # Keep ATMOSPHERIC_PARAMS_SHORT / SURFACE_PARAMS_SHORT minimal
    ATMOSPHERIC_PARAMS_SHORT = [
        n[0] for n in ATMOSPHERIC_PARAMS
    ]  # first letter fallback
    SURFACE_PARAMS_SHORT = list(SURFACE_PARAMS)
    PARAM_NAMES_SHORT = state_short_names
    PARAM_UNITS = state_units

    # ---- Forcing ----
    GRID_ORIGINAL_FORCING_DIM = cfg["forcing"]["num_features"]
    GRID_FORCING_DIM = (
        GRID_ORIGINAL_FORCING_DIM * cfg["forcing"]["window"]
    )

    # ---- Forecast horizons ----
    eval_leads = cfg["forecast"]["eval_leads"]

    # Val-step logging: subset of paper values that fit within eval_leads
    VAL_STEP_LOG_ERRORS = np.array(
        [s for s in [1, 2, 4, 6, 8, 10, 12, 20, 40] if s <= eval_leads]
    )
    VAL_STEP_CHECKPOINTS = np.array(
        [s for s in [1, 4, 12] if s <= eval_leads]
    )

    # Plot + metrics-watch: use all variables at key leads
    EVAL_PLOT_VARS = np.arange(GRID_STATE_DIM)
    key_leads = [
        l for l in [1, 2, 4, 6, 12] if l <= eval_leads
    ]
    VAL_PLOT_VARS = {
        i: np.array(key_leads) for i in range(GRID_STATE_DIM)
    }
    VAR_LEADS_METRICS_WATCH = {
        i: [1, min(12, eval_leads)] for i in range(GRID_STATE_DIM)
    }

    # ---- Split info ----
    # Stored on _CONFIG for era5_dataset to read; also exposed as
    # convenience attributes.
    cfg["_split_dates"] = cfg["splits"]

    print(f"[constants] Loaded experiment config from {config_path}")
    print(f"  GRID_SHAPE = {GRID_SHAPE}")
    print(f"  GRID_STATE_DIM = {GRID_STATE_DIM}")
    print(f"  ATMOSPHERIC_PARAMS = {ATMOSPHERIC_PARAMS}")
    print(f"  SURFACE_PARAMS = {SURFACE_PARAMS}")
    print(f"  PRESSURE_LEVELS = {PRESSURE_LEVELS}")
    print(f"  PARAM_NAMES_SHORT = {PARAM_NAMES_SHORT}")
    print(f"  VAL_STEP_LOG_ERRORS = {VAL_STEP_LOG_ERRORS}")
