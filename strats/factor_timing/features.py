from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """
    Lightweight feature declaration for panel research fixtures.
    """

    name: str
    role: str = "entity"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureSpec name must be non-empty.")
        if not self.role:
            raise ValueError("FeatureSpec role must be non-empty.")


def _check_epsilon(value: float, *, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")


def _check_window(length: int, window: int, *, name: str) -> None:
    if window < 1:
        raise ValueError(f"{name} window must be greater than zero.")
    if length < window:
        raise ValueError(f"{name} length must be at least the window.")


def _clip(values: torch.Tensor, clip_value: float | None) -> torch.Tensor:
    if clip_value is None:
        return values
    if clip_value <= 0.0:
        raise ValueError("clip_value must be greater than zero.")
    return values.clamp(-clip_value, clip_value)


def log_returns(prices: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    _check_epsilon(epsilon, name="epsilon")
    if prices.shape[-1] < 2:
        raise ValueError("prices must contain at least two timesteps.")
    safe_prices = prices.clamp_min(epsilon)
    return torch.log(safe_prices[..., 1:] / safe_prices[..., :-1])


def delta_log(values: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Log change for non-negative scale or volume-like series.
    """

    _check_epsilon(epsilon, name="epsilon")
    if values.shape[-1] < 2:
        raise ValueError("values must contain at least two timesteps.")
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() > 0 and bool((finite_values < 0.0).any().item()):
        raise ValueError("delta_log expects non-negative scale values.")
    safe_values = values.clamp_min(epsilon)
    return torch.log(safe_values[..., 1:] / safe_values[..., :-1])


def vol_scaled_returns(
    returns: torch.Tensor,
    volatility: torch.Tensor,
    *,
    floor: float = 1e-6,
    clip_value: float | None = None,
) -> torch.Tensor:
    _check_epsilon(floor, name="floor")
    return _clip(returns / volatility.clamp_min(floor), clip_value)


def realized_volatility(
    returns: torch.Tensor,
    *,
    window: int,
    floor: float = 0.0,
) -> torch.Tensor:
    if floor < 0.0:
        raise ValueError("Realized volatility floor cannot be negative.")
    _check_window(returns.shape[-1], window, name="Realized volatility")
    windows = returns.unfold(dimension=-1, size=window, step=1)
    volatility = windows.std(dim=-1, unbiased=False)
    return volatility if floor == 0.0 else volatility.clamp_min(floor)


def downside_volatility(
    returns: torch.Tensor,
    *,
    window: int,
    threshold: float = 0.0,
    floor: float = 0.0,
) -> torch.Tensor:
    if floor < 0.0:
        raise ValueError("Downside volatility floor cannot be negative.")
    _check_window(returns.shape[-1], window, name="Downside volatility")
    downside = torch.minimum(
        returns - threshold,
        torch.zeros((), device=returns.device, dtype=returns.dtype),
    )
    windows = downside.unfold(dimension=-1, size=window, step=1)
    volatility = windows.square().mean(dim=-1).sqrt()
    return volatility if floor == 0.0 else volatility.clamp_min(floor)


def ewma_volatility(
    returns: torch.Tensor,
    *,
    decay: float | None = 0.94,
    half_life: float | None = None,
    floor: float = 1e-8,
) -> torch.Tensor:
    _check_epsilon(floor, name="floor")
    if returns.shape[-1] < 1:
        raise ValueError("returns must contain at least one timestep.")
    if half_life is not None:
        if half_life <= 0.0:
            raise ValueError("EWMA half_life must be greater than zero.")
        decay = math.exp(math.log(0.5) / half_life)
    if decay is None:
        raise ValueError("EWMA decay or half_life must be supplied.")
    if not 0.0 < decay < 1.0:
        raise ValueError("EWMA decay must be in the interval (0, 1).")
    variance = returns[..., 0].square()
    values = [variance.sqrt().clamp_min(floor)]
    for index in range(1, returns.shape[-1]):
        variance = decay * variance + (1.0 - decay) * returns[..., index].square()
        values.append(variance.sqrt().clamp_min(floor))
    return torch.stack(values, dim=-1)


def volatility_ratio(
    fast_volatility: torch.Tensor,
    slow_volatility: torch.Tensor,
    *,
    floor: float = 1e-8,
    clip_value: float | None = None,
) -> torch.Tensor:
    _check_epsilon(floor, name="floor")
    return _clip(fast_volatility / slow_volatility.clamp_min(floor), clip_value)


def ewma_volatility_ratio(
    returns: torch.Tensor,
    *,
    fast_half_life: float = 20.0,
    slow_half_life: float = 60.0,
    floor: float = 1e-8,
    clip_value: float | None = None,
) -> torch.Tensor:
    fast = ewma_volatility(returns, half_life=fast_half_life, floor=floor)
    slow = ewma_volatility(returns, half_life=slow_half_life, floor=floor)
    return volatility_ratio(fast, slow, floor=floor, clip_value=clip_value)


def drawdown_state(returns: torch.Tensor, *, floor: float = 1e-12) -> torch.Tensor:
    _check_epsilon(floor, name="floor")
    if returns.shape[-1] < 1:
        raise ValueError("returns must contain at least one timestep.")
    wealth = torch.cumprod(1.0 + returns, dim=-1)
    wealth = torch.cat((torch.ones_like(wealth[..., :1]), wealth), dim=-1)
    peak = torch.cummax(wealth, dim=-1).values
    return wealth[..., 1:] / peak[..., 1:].clamp_min(floor) - 1.0


def cross_sectional_dispersion(
    values: torch.Tensor,
    *,
    dim: int = -2,
    keepdim: bool = False,
) -> torch.Tensor:
    return values.to(dtype=torch.float32).std(dim=dim, unbiased=False, keepdim=keepdim)


def relative_change(values: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    _check_epsilon(epsilon, name="epsilon")
    if values.shape[-1] < 2:
        raise ValueError("values must contain at least two timesteps.")
    previous = values[..., :-1]
    return (values[..., 1:] - previous) / previous.abs().clamp_min(epsilon)


def zscore(
    values: torch.Tensor,
    *,
    dim: int = -1,
    epsilon: float = 1e-6,
    clip_value: float | None = None,
) -> torch.Tensor:
    _check_epsilon(epsilon, name="epsilon")
    centered = values - values.mean(dim=dim, keepdim=True)
    scale = values.std(dim=dim, unbiased=False, keepdim=True).clamp_min(epsilon)
    return _clip(centered / scale, clip_value)


def robust_zscore(
    values: torch.Tensor,
    *,
    dim: int = -1,
    epsilon: float = 1e-6,
    clip_value: float | None = None,
) -> torch.Tensor:
    _check_epsilon(epsilon, name="epsilon")
    median = values.median(dim=dim, keepdim=True).values
    centered = values - median
    lower = torch.quantile(values, 0.25, dim=dim, keepdim=True)
    upper = torch.quantile(values, 0.75, dim=dim, keepdim=True)
    return _clip(centered / (upper - lower).clamp_min(epsilon), clip_value)


def rolling_zscore(
    values: torch.Tensor,
    *,
    window: int,
    epsilon: float = 1e-6,
    clip_value: float | None = None,
) -> torch.Tensor:
    _check_epsilon(epsilon, name="epsilon")
    _check_window(values.shape[-1], window, name="Rolling z-score")
    windows = values.unfold(dimension=-1, size=window, step=1)
    current = windows[..., -1]
    centered = current - windows.mean(dim=-1)
    scale = windows.std(dim=-1, unbiased=False).clamp_min(epsilon)
    return _clip(centered / scale, clip_value)


def rolling_robust_zscore(
    values: torch.Tensor,
    *,
    window: int,
    epsilon: float = 1e-6,
    clip_value: float | None = None,
) -> torch.Tensor:
    _check_epsilon(epsilon, name="epsilon")
    _check_window(values.shape[-1], window, name="Rolling robust z-score")
    windows = values.unfold(dimension=-1, size=window, step=1)
    current = windows[..., -1]
    median = windows.median(dim=-1).values
    lower = torch.quantile(windows, 0.25, dim=-1)
    upper = torch.quantile(windows, 0.75, dim=-1)
    return _clip((current - median) / (upper - lower).clamp_min(epsilon), clip_value)


def group_spillover_summary(
    returns: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    window: int | None = 20,
) -> torch.Tensor:
    """
    Returns per-group latest mean and trailing volatility from `[entity, time]`.
    """

    if returns.ndim != 2:
        raise ValueError("Spillover returns must have shape [entity, time].")
    if group_ids.ndim != 1 or group_ids.shape[0] != returns.shape[0]:
        raise ValueError("Group ids must have shape [entity].")
    if returns.shape[1] < 1:
        raise ValueError("Spillover returns must contain at least one timestep.")
    if window is not None and window < 1:
        raise ValueError("Spillover summary window must be greater than zero.")

    trailing_length = (
        returns.shape[1] if window is None else min(returns.shape[1], window)
    )
    summaries = []
    for group_id in torch.unique(group_ids, sorted=True):
        group_returns = returns.index_select(
            0, (group_ids == group_id).nonzero().squeeze(-1)
        )
        summaries.append(
            torch.stack(
                (
                    group_returns[:, -1].mean(),
                    group_returns[:, -trailing_length:].std(unbiased=False),
                )
            )
        )
    return torch.stack(summaries)


def regional_spillover_summary(
    returns: torch.Tensor,
    region_ids: torch.Tensor,
    *,
    window: int | None = 20,
) -> torch.Tensor:
    return group_spillover_summary(returns, region_ids, window=window)
