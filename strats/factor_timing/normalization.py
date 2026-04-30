import math

from dataclasses import dataclass

import torch


_MISSING_POLICIES = ("propagate", "raise", "omit", "zero")


@dataclass(frozen=True, slots=True)
class StandardizationStats:
    """
    Train-window scaling statistics.
    """

    mean: torch.Tensor
    scale: torch.Tensor
    clip_value: float | None = None
    missing_policy: str = "propagate"
    fill_value: float = 0.0
    method: str = "standard"
    winsor_lower: torch.Tensor | None = None
    winsor_upper: torch.Tensor | None = None


def _float_values(values: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(values):
        return values
    return values.to(dtype=torch.float32)


def _check_missing_policy(missing_policy: str) -> None:
    if missing_policy not in _MISSING_POLICIES:
        raise ValueError(
            "Missing-value policy must be one of: " + ", ".join(_MISSING_POLICIES)
        )


def _fit_values(values: torch.Tensor, *, missing_policy: str) -> torch.Tensor:
    values = _float_values(values)
    if missing_policy == "raise" and not bool(torch.isfinite(values).all().item()):
        raise ValueError("Cannot fit normalizer with missing or non-finite values.")
    if missing_policy == "zero":
        return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def _sample_view(
    values: torch.Tensor,
    sample_dims: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if not sample_dims:
        raise ValueError("At least one sample dimension is required.")

    ndim = values.ndim
    dims = tuple(sorted(dim % ndim for dim in sample_dims))
    if len(set(dims)) != len(dims):
        raise ValueError("Sample dimensions must be unique.")

    retained_dims = tuple(index for index in range(ndim) if index not in dims)
    order = dims + retained_dims
    sample_count = math.prod(values.shape[index] for index in dims)
    retained_shape = tuple(values.shape[index] for index in retained_dims)
    stat_shape = tuple(
        1 if index in dims else values.shape[index] for index in range(ndim)
    )
    return values.permute(order).reshape(sample_count, *retained_shape), stat_shape


def _reshape_stat(statistic: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return statistic.reshape(shape)


def _missing_fill(values: torch.Tensor, fill_value: float) -> torch.Tensor:
    return torch.nan_to_num(
        values, nan=fill_value, posinf=fill_value, neginf=fill_value
    )


def _winsor_bounds(
    values: torch.Tensor,
    sample_dims: tuple[int, ...],
    quantiles: tuple[float, float] | None,
    *,
    missing_policy: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if quantiles is None:
        return None, None
    lower_quantile, upper_quantile = quantiles
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1.")

    flat, shape = _sample_view(values, sample_dims)
    quantile = torch.nanquantile if missing_policy == "omit" else torch.quantile
    if missing_policy == "omit":
        flat = torch.where(
            torch.isfinite(flat), flat, torch.full_like(flat, float("nan"))
        )
    lower = quantile(flat, lower_quantile, dim=0)
    upper = quantile(flat, upper_quantile, dim=0)
    if missing_policy == "omit":
        lower = _missing_fill(lower, 0.0)
        upper = _missing_fill(upper, 0.0)
    return _reshape_stat(lower, shape), _reshape_stat(upper, shape)


def _apply_winsor(
    values: torch.Tensor,
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
) -> torch.Tensor:
    if lower is None or upper is None:
        return values
    lower = lower.to(device=values.device, dtype=values.dtype)
    upper = upper.to(device=values.device, dtype=values.dtype)
    return torch.minimum(torch.maximum(values, lower), upper)


def fit_standardizer(
    values: torch.Tensor,
    *,
    sample_dims: tuple[int, ...] = (0,),
    epsilon: float = 1e-6,
    clip_value: float | None = None,
    winsor_quantiles: tuple[float, float] | None = None,
    missing_policy: str = "propagate",
    fill_value: float = 0.0,
) -> StandardizationStats:
    """
    Fits standardization statistics on the supplied train-window values only.
    """

    _check_missing_policy(missing_policy)
    values = _fit_values(values, missing_policy=missing_policy)
    winsor_lower, winsor_upper = _winsor_bounds(
        values,
        sample_dims,
        winsor_quantiles,
        missing_policy=missing_policy,
    )
    values = _apply_winsor(values, winsor_lower, winsor_upper)
    if missing_policy == "omit":
        flat, shape = _sample_view(values, sample_dims)
        finite = torch.isfinite(flat)
        clean = torch.where(finite, flat, torch.zeros_like(flat))
        count = finite.to(dtype=flat.dtype).sum(dim=0).clamp_min(1.0)
        mean = clean.sum(dim=0) / count
        centered = torch.where(finite, flat - mean.unsqueeze(0), torch.zeros_like(flat))
        variance = centered.square().sum(dim=0) / count
        scale = variance.sqrt().clamp_min(epsilon)
        mean = _reshape_stat(mean, shape)
        scale = _reshape_stat(scale, shape)
    else:
        mean = values.mean(dim=sample_dims, keepdim=True)
        scale = values.std(dim=sample_dims, unbiased=False, keepdim=True).clamp_min(
            epsilon
        )
    return StandardizationStats(
        mean=mean,
        scale=scale,
        clip_value=clip_value,
        missing_policy=missing_policy,
        fill_value=fill_value,
        method="standard",
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
    )


def fit_robust_standardizer(
    values: torch.Tensor,
    *,
    sample_dims: tuple[int, ...] = (0,),
    epsilon: float = 1e-6,
    clip_value: float | None = None,
    winsor_quantiles: tuple[float, float] | None = None,
    missing_policy: str = "propagate",
    fill_value: float = 0.0,
) -> StandardizationStats:
    """
    Fits median/IQR scaling statistics on the supplied train-window values only.
    """

    _check_missing_policy(missing_policy)
    values = _fit_values(values, missing_policy=missing_policy)
    winsor_lower, winsor_upper = _winsor_bounds(
        values,
        sample_dims,
        winsor_quantiles,
        missing_policy=missing_policy,
    )
    values = _apply_winsor(values, winsor_lower, winsor_upper)
    flat, shape = _sample_view(values, sample_dims)
    quantile = torch.nanquantile if missing_policy == "omit" else torch.quantile
    if missing_policy == "omit":
        flat = torch.where(
            torch.isfinite(flat), flat, torch.full_like(flat, float("nan"))
        )

    median = quantile(flat, 0.50, dim=0)
    lower = quantile(flat, 0.25, dim=0)
    upper = quantile(flat, 0.75, dim=0)
    scale = upper - lower
    if missing_policy == "omit":
        median = _missing_fill(median, 0.0)
        scale = _missing_fill(scale, 1.0)

    return StandardizationStats(
        mean=_reshape_stat(median, shape),
        scale=_reshape_stat(scale.clamp_min(epsilon), shape),
        clip_value=clip_value,
        missing_policy=missing_policy,
        fill_value=fill_value,
        method="robust",
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
    )


def apply_standardizer(
    values: torch.Tensor,
    stats: StandardizationStats,
) -> torch.Tensor:
    _check_missing_policy(stats.missing_policy)
    dtype = values.dtype if torch.is_floating_point(values) else stats.mean.dtype
    values = values.to(dtype=dtype)
    valid = torch.isfinite(values)
    if stats.missing_policy == "raise" and not bool(valid.all().item()):
        raise ValueError("Cannot transform missing or non-finite values.")
    if stats.missing_policy in {"omit", "zero"}:
        values = _missing_fill(values, stats.fill_value)
    values = _apply_winsor(values, stats.winsor_lower, stats.winsor_upper)

    normalized = (values - stats.mean.to(device=values.device, dtype=dtype)) / (
        stats.scale.to(device=values.device, dtype=dtype)
    )
    if stats.clip_value is None:
        output = normalized
    else:
        output = normalized.clamp(-stats.clip_value, stats.clip_value)
    if stats.missing_policy in {"omit", "zero"}:
        output = torch.where(valid, output, torch.full_like(output, stats.fill_value))
    return output


def fit_apply_train_standardizer(
    train_values: torch.Tensor,
    validation_values: torch.Tensor,
    test_values: torch.Tensor,
    *,
    sample_dims: tuple[int, ...] = (0,),
    clip_value: float | None = None,
    winsor_quantiles: tuple[float, float] | None = None,
    method: str = "standard",
    epsilon: float = 1e-6,
    missing_policy: str = "propagate",
    fill_value: float = 0.0,
) -> tuple[StandardizationStats, torch.Tensor, torch.Tensor, torch.Tensor]:
    if method == "standard":
        stats = fit_standardizer(
            train_values,
            sample_dims=sample_dims,
            epsilon=epsilon,
            clip_value=clip_value,
            winsor_quantiles=winsor_quantiles,
            missing_policy=missing_policy,
            fill_value=fill_value,
        )
    elif method == "robust":
        stats = fit_robust_standardizer(
            train_values,
            sample_dims=sample_dims,
            epsilon=epsilon,
            clip_value=clip_value,
            winsor_quantiles=winsor_quantiles,
            missing_policy=missing_policy,
            fill_value=fill_value,
        )
    else:
        raise ValueError("Standardizer method must be either 'standard' or 'robust'.")
    return (
        stats,
        apply_standardizer(train_values, stats),
        apply_standardizer(validation_values, stats),
        apply_standardizer(test_values, stats),
    )
