# Third-party
import torch


def get_metric(metric_name):
    """
    Get a defined metric with given name

    metric_name: str, name of the metric

    Returns:
    metric: function implementing the metric
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(
    metric_entry_vals, mask, grid_weights, average_grid, sum_vars
):
    """
    Masks and (optionally) reduces entry-wise metric values

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    metric_entry_vals: (..., N, d_state), prediction
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Perform weighting before masking
    if grid_weights is not None:
        metric_entry_vals = metric_entry_vals * grid_weights.unsqueeze(-1)
        # (..., N', d_state)

    # Only keep grid nodes in mask
    if mask is not None:
        metric_entry_vals = metric_entry_vals[
            ..., mask, :
        ]  # (..., N', d_state)

    # Optionally reduce last two dimensions
    if average_grid:  # Reduce grid first
        metric_entry_vals = torch.mean(
            metric_entry_vals, dim=-2
        )  # (..., d_state)
    if sum_vars:  # Reduce vars second
        metric_entry_vals = torch.sum(
            metric_entry_vals, dim=-1
        )  # (..., N) or (...,)

    return metric_entry_vals


def wmse(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    Weighted Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    entry_mse = torch.nn.functional.mse_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mse_weighted = entry_mse / (pred_std**2)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mse_weighted,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


# Allow for unused pred_std for consistent signature
def mse(
    pred,
    target,
    pred_std,  # pylint: disable=unused-argument
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    (Unweighted) Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev. (unused)
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred,
        target,
        torch.ones_like(pred),
        mask,
        grid_weights,
        average_grid,
        sum_vars,
    )


def wmae(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    Weighted Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    entry_mae = torch.nn.functional.l1_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mae_weighted = entry_mae / pred_std  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mae_weighted,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


# Allow for unused pred_std for consistent signature
def mae(
    pred,
    target,
    pred_std,  # pylint: disable=unused-argument
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    (Unweighted) Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev. (unused)
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred,
        target,
        torch.ones_like(pred),
        mask,
        grid_weights,
        average_grid,
        sum_vars,
    )


def nll(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    Negative Log Likelihood loss, for isotropic Gaussian likelihood

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Broadcast pred_std if shaped (d_state,), done internally in Normal class
    dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
    entry_nll = -dist.log_prob(target)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_nll,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def crps_gauss(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    (Negative) Continuous Ranked Probability Score (CRPS)
    Closed-form expression based on Gaussian predictive distribution

    (...,) is any number of batch dimensions, potentially different
            but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    std_normal = torch.distributions.Normal(
        torch.zeros((), device=pred.device), torch.ones((), device=pred.device)
    )
    target_standard = (target - pred) / pred_std  # (..., N, d_state)

    entry_crps = -pred_std * (
        torch.pi ** (-0.5)
        - 2 * torch.exp(std_normal.log_prob(target_standard))
        - target_standard * (2 * std_normal.cdf(target_standard) - 1)
    )  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_crps,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


# Allow for unused pred_std for consistent signature
def crps_ens(
    pred,
    target,
    pred_std,  # pylint: disable=unused-argument
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
    ens_dim=1,
):
    """
    (Negative) Continuous Ranked Probability Score (CRPS)
    Unbiased estimator from samples. See e.g. Weatherbench 2.

    (..., M, ...,) is any number of batch dimensions, including ensemble
        dimension M
    pred: (..., M, ..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., M, ..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced
        (sum over d_state)
    ens_dim: batch dimension where ensemble members are laid out, to reduce over

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    num_ens = pred.shape[ens_dim]  # Number of ensemble members
    if num_ens == 1:
        # With one sample CRPS reduces to MAE
        return mae(
            pred.squeeze(ens_dim),
            target,
            None,
            mask=mask,
            average_grid=average_grid,
        )

    if num_ens == 2:
        mean_mae = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
        )  # (..., N, d_state)

        # Use simpler estimator
        pair_diffs_term = -0.5 * torch.abs(
            pred.select(ens_dim, 0) - pred.select(ens_dim, 1)
        )  # (..., N, d_state)

        crps_estimator = mean_mae + pair_diffs_term  # (..., N, d_state)
    elif num_ens < 10:
        # This is the rank-based implementation with O(M*log(M)) compute and
        # O(M) memory. See Zamo and Naveau and WB2 for explanation.
        # For smaller ensemble we can compute all of this directly in memory.
        mean_mae = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
        )  # (..., N, d_state)

        # Ranks start at 1, two argsorts will compute entry ranks
        ranks = pred.argsort(dim=ens_dim).argsort(ens_dim) + 1

        pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
            (num_ens + 1 - 2 * ranks) * pred,
            dim=ens_dim,
        )  # (..., N, d_state)

        crps_estimator = mean_mae + pair_diffs_term  # (..., N, d_state)
    else:
        # For large ensembles we batch this over the variable dimension
        crps_res = []
        for var_i in range(pred.shape[-1]):
            pred_var = pred[..., var_i]
            target_var = target[..., var_i]

            mean_mae = torch.mean(
                torch.abs(pred_var - target_var.unsqueeze(ens_dim)), dim=ens_dim
            )  # (..., N)

            # Ranks start at 1, two argsorts will compute entry ranks
            ranks = pred_var.argsort(dim=ens_dim).argsort(ens_dim) + 1
            # (..., M, ..., N)

            pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
                (num_ens + 1 - 2 * ranks) * pred_var,
                dim=ens_dim,
            )  # (..., N)
            crps_res.append(mean_mae + pair_diffs_term)

        crps_estimator = torch.stack(crps_res, dim=-1)

    return mask_and_reduce_metric(
        crps_estimator, mask, grid_weights, average_grid, sum_vars
    )


def spread_squared(
    pred,
    target,  # pylint: disable=unused-argument
    pred_std,  # pylint: disable=unused-argument
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
    ens_dim=1,
):
    """
    (Squared) spread of ensemble.
    Similarly to RMSE, we want to take sqrt after spatial and sample averaging,
    so we need to average the squared spread.

    (..., M, ...,) is any number of batch dimensions, including ensemble
        dimension M
    pred: (..., M, ..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., M, ..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    grid_weights: (N,), weighting to apply over grid nodes
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced
        (sum over d_state)
    ens_dim: batch dimension where ensemble members are laid out, to reduce over

    Returns:
    metric_val: One of (...,), (..., d_state) depending on reduction arguments.
    """
    entry_var = torch.var(pred, dim=ens_dim)  # (..., N, d_state)
    return mask_and_reduce_metric(
        entry_var, mask, grid_weights, average_grid, sum_vars
    )


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    "crps_ens": crps_ens,
    "spread_squared": spread_squared,
}


# ---------------------------------------------------------------------------
# Ensemble evaluation helpers (for external evaluation scripts)
# ---------------------------------------------------------------------------

def ens_mean_mse(
    pred, target, mask=None, grid_weights=None, average_grid=True,
    sum_vars=True, ens_dim=1,
):
    """
    MSE of the ensemble mean.

    pred: (..., M, ..., N, d_state)
    target: (..., N, d_state)
    """
    ens_mean = torch.mean(pred, dim=ens_dim)  # (..., N, d_state)
    return mse(
        ens_mean, target, None,
        mask=mask, grid_weights=grid_weights,
        average_grid=average_grid, sum_vars=sum_vars,
    )


def ens_mean_bias(
    pred, target, mask=None, grid_weights=None, average_grid=True,
    sum_vars=True, ens_dim=1,
):
    """Bias of ensemble mean (pred - target)."""
    ens_mean = torch.mean(pred, dim=ens_dim)
    bias = ens_mean - target  # (..., N, d_state)
    if mask is not None:
        bias = bias[..., mask, :]
    if grid_weights is not None:
        bias = bias * grid_weights.unsqueeze(-1)
    if average_grid:
        bias = torch.mean(bias, dim=-2)
    if sum_vars:
        bias = torch.sum(bias, dim=-1)
    return bias


def spread_skill_ratio(
    pred, target, mask=None, grid_weights=None, ens_dim=1,
):
    """
    Spread-skill ratio: sqrt(ensemble variance / ensemble-mean MSE).
    Returns per-variable ratio. Ideal: ~1.0.

    pred: (..., M, ..., N, d_state)
    target: (..., N, d_state)

    Returns: (..., d_state)
    """
    spread_sq = spread_squared(
        pred, target, None, mask=mask, grid_weights=grid_weights,
        average_grid=True, sum_vars=False, ens_dim=ens_dim,
    )  # (..., d_state)
    ens_mse_val = ens_mean_mse(
        pred, target, mask=mask, grid_weights=grid_weights,
        average_grid=True, sum_vars=False, ens_dim=ens_dim,
    )  # (..., d_state)
    return torch.sqrt(spread_sq / (ens_mse_val + 1e-8))


def interval_coverage(
    pred, target, central_fraction=0.9, ens_dim=1,
):
    """
    Fraction of targets falling within the central P% interval of the
    ensemble.

    pred: (..., M, ..., N, d_state)
    target: (..., N, d_state)
    central_fraction: float in (0, 1)

    Returns: (..., d_state) — coverage per variable
    """
    lower_quantile = (1.0 - central_fraction) / 2.0
    upper_quantile = 1.0 - lower_quantile

    lower = torch.quantile(pred, lower_quantile, dim=ens_dim)
    upper = torch.quantile(pred, upper_quantile, dim=ens_dim)

    within = (
        (target >= lower) & (target <= upper)
    ).float()  # (..., N, d_state)

    # Average over spatial dim
    return torch.mean(within, dim=-2)  # (..., d_state)


def rank_histogram(
    pred, target, n_bins=None, ens_dim=1,
):
    """
    Compute rank histogram counts for each variable.

    pred: (..., M, ..., N, d_state)
    target: (..., N, d_state)

    Returns: (d_state, n_bins+1) — counts per rank per variable,
             normalized to sum to 1 along last dim.
    """
    num_ens = pred.shape[ens_dim]
    if n_bins is None:
        n_bins = num_ens

    # For each spatial location and variable, count how many ensemble
    # members are below the target → rank (0..num_ens)
    below = (pred < target.unsqueeze(ens_dim)).float()
    ranks = torch.sum(below, dim=ens_dim).long()  # (..., N, d_state), 0..M

    # Assign to bins: 0..M into n_bins+1 bins
    bin_edges = torch.linspace(0, num_ens, n_bins + 1, device=pred.device)
    # For each rank, find which bin
    hist = torch.zeros(
        pred.shape[-1], n_bins + 1, device=pred.device
    )
    for b in range(n_bins + 1):
        if b == 0:
            mask_bin = ranks <= bin_edges[0]
        elif b == n_bins:
            mask_bin = ranks > bin_edges[-2]
        else:
            mask_bin = (ranks > bin_edges[b - 1]) & (
                ranks <= bin_edges[b]
            )
        for v in range(pred.shape[-1]):
            hist[v, b] = mask_bin[..., v].sum()

    # Normalize
    hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
    return hist


def apply_spread_scaling(
    pred, alpha, ens_dim=1,
):
    """
    Apply a multiplicative spread scaling to ensemble predictions.
    Preserves the ensemble mean.

    pred: (..., M, ..., N, d_state)
    alpha: (d_state,) — scaling factor per variable

    Returns: (..., M, ..., N, d_state) — rescaled ensemble
    """
    ens_mean = torch.mean(pred, dim=ens_dim, keepdim=True)
    deviation = pred - ens_mean
    return ens_mean + alpha[..., None, None] * deviation
