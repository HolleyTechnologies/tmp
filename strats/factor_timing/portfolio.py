import math

import torch

from ...contracts import SACActionDistribution
from .schema import ForecastMoments, PortfolioConstraints, PositionSignal


def _raw_action(
    signal: torch.Tensor | PositionSignal,
) -> tuple[torch.Tensor, ForecastMoments | None]:
    if isinstance(signal, PositionSignal):
        return signal.action, signal.moments
    return signal, None


def _moment(
    value: torch.Tensor | None,
    action: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value.to(device=action.device, dtype=action.dtype)
    if tensor.shape != action.shape:
        raise ValueError(f"{name} must have shape [action_dim].")
    return tensor


def _confidence_scale(
    action: torch.Tensor,
    moments: ForecastMoments | None,
    *,
    minimum: float,
) -> torch.Tensor:
    scale = torch.ones_like(action)
    if moments is None:
        return scale
    conviction = _moment(moments.conviction, action, name="conviction")
    probability_positive = _moment(
        moments.probability_positive,
        action,
        name="probability_positive",
    )
    expected_return = _moment(moments.expected_return, action, name="expected_return")
    forecast_uncertainty = _moment(
        moments.forecast_uncertainty,
        action,
        name="forecast_uncertainty",
    )
    if conviction is not None:
        scale = scale * conviction.abs().clamp(0.0, 1.0)
    if probability_positive is not None:
        probability = probability_positive
        scale = scale * (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
    if expected_return is not None and forecast_uncertainty is not None:
        expected = expected_return.abs()
        uncertainty = forecast_uncertainty.abs()
        scale = scale * (expected / (expected + uncertainty + 1e-8)).clamp(0.0, 1.0)
    return scale.clamp_min(minimum)


def _apply_net_limit(
    positions: torch.Tensor,
    *,
    net_limit: float | None,
    active: torch.Tensor,
    max_position: float,
) -> torch.Tensor:
    if net_limit is None:
        return positions
    for _ in range(2):
        net = positions.sum()
        if float(net.abs().item()) <= net_limit + 1e-8:
            return positions
        active_count = active.to(dtype=positions.dtype).sum().clamp_min(1.0)
        target_net = net.sign() * net.new_tensor(net_limit)
        positions = positions - active.to(dtype=positions.dtype) * (
            (net - target_net) / active_count
        )
        positions = torch.where(
            active,
            positions.clamp(-max_position, max_position),
            positions,
        )
    return positions


def _scale_active_abs_sum(
    positions: torch.Tensor,
    limit: float | None,
    *,
    active: torch.Tensor,
) -> torch.Tensor:
    if limit is None:
        return positions
    frozen_exposure = positions.masked_fill(active, 0.0).abs().sum()
    remaining = limit - float(frozen_exposure.item())
    if remaining <= 0.0:
        return positions.masked_fill(active, 0.0)
    active_exposure = positions.masked_fill(~active, 0.0).abs().sum()
    if (
        float(active_exposure.item()) <= remaining
        or float(active_exposure.item()) <= 1e-12
    ):
        return positions
    scale = positions.new_tensor(remaining) / active_exposure
    return torch.where(active, positions * scale, positions)


def _apply_concentration_limit(
    positions: torch.Tensor,
    concentration_limit: float | None,
    *,
    active: torch.Tensor,
) -> torch.Tensor:
    if concentration_limit is None:
        return positions
    concentration = positions.square().sum()
    if float(concentration.item()) <= concentration_limit:
        return positions
    frozen_concentration = positions.masked_fill(active, 0.0).square().sum()
    remaining = concentration_limit - float(frozen_concentration.item())
    if remaining <= 0.0:
        return positions.masked_fill(active, 0.0)
    active_concentration = positions.masked_fill(~active, 0.0).square().sum()
    if float(active_concentration.item()) <= 1e-12:
        return positions
    scale = math.sqrt(remaining / float(active_concentration.item()))
    return torch.where(active, positions * scale, positions)


def _single_action_row(
    values: torch.Tensor,
    *,
    name: str,
    batch_index: int | None,
) -> torch.Tensor:
    if values.ndim == 1:
        if batch_index is not None:
            raise ValueError(f"{name} is unbatched but batch_index was provided.")
        return values
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [action_dim] or [batch, action_dim].")
    if batch_index is None:
        if values.shape[0] != 1:
            raise ValueError(f"{name} with multiple batch rows requires batch_index.")
        return values.squeeze(0)
    if batch_index < 0 or batch_index >= values.shape[0]:
        raise ValueError(f"{name} batch_index is out of range.")
    return values[batch_index]


def position_signal_from_sac_distribution(
    distribution: SACActionDistribution,
    *,
    action: torch.Tensor | None = None,
    batch_index: int | None = None,
) -> PositionSignal:
    """
    Converts a SAC actor distribution into strategy-local sizing information.
    """

    mean = _single_action_row(
        distribution.mean,
        name="SACActionDistribution.mean",
        batch_index=batch_index,
    )
    log_std = _single_action_row(
        distribution.log_std,
        name="SACActionDistribution.log_std",
        batch_index=batch_index,
    )
    bounded_mean = mean.tanh()
    if action is None:
        bounded_action = bounded_mean
    else:
        bounded_action = _single_action_row(
            action.to(device=mean.device, dtype=mean.dtype),
            name="action",
            batch_index=batch_index,
        ).clamp(-1.0, 1.0)
        if bounded_action.shape != bounded_mean.shape:
            raise ValueError("Action must share the distribution action shape.")

    raw_std = log_std.exp()
    bounded_std = (raw_std * (1.0 - bounded_mean.square())).clamp_min(0.0)
    return PositionSignal(
        action=bounded_action,
        moments=ForecastMoments(
            action_mean=bounded_mean,
            action_std=bounded_std,
            forecast_uncertainty=bounded_std,
            conviction=(1.0 / (1.0 + bounded_std)).clamp(0.0, 1.0),
        ),
    )


def project_positions(
    signal: torch.Tensor | PositionSignal,
    *,
    previous_positions: torch.Tensor | None = None,
    constraints: PortfolioConstraints | None = None,
    tradable_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Projects bounded actions into strategy-layer target positions.
    """

    constraints = PortfolioConstraints() if constraints is None else constraints
    action, moments = _raw_action(signal)
    positions = action.to(dtype=torch.float32).clamp(-1.0, 1.0)
    if positions.ndim != 1:
        raise ValueError("Factor timing actions must have shape [action_dim].")
    if positions.numel() < 1:
        raise ValueError("Factor timing actions must contain at least one value.")
    positions = positions * _confidence_scale(
        positions,
        moments,
        minimum=constraints.min_confidence_scale,
    )

    previous = None
    if previous_positions is not None:
        previous = previous_positions.to(device=positions.device, dtype=positions.dtype)
        if previous.shape != positions.shape:
            raise ValueError("previous_positions must have shape [action_dim].")

    if tradable_mask is None:
        active = torch.ones_like(positions, dtype=torch.bool)
    else:
        active = tradable_mask.to(device=positions.device, dtype=torch.bool)
        if active.shape != positions.shape:
            raise ValueError("tradable_mask must have shape [action_dim].")
        if previous is None and bool((~active).any().item()):
            raise ValueError(
                "previous_positions are required when tradable_mask freezes action slots."
            )
        if previous is not None:
            positions = torch.where(active, positions, previous)

    positions = torch.where(
        active,
        positions.clamp(-constraints.max_position, constraints.max_position),
        positions,
    )
    positions = _apply_net_limit(
        positions,
        net_limit=constraints.net_limit,
        active=active,
        max_position=constraints.max_position,
    )
    positions = _apply_concentration_limit(
        positions,
        constraints.concentration_limit,
        active=active,
    )

    gross_limit = constraints.gross_limit
    if constraints.leverage_limit is not None:
        gross_limit = min(gross_limit, constraints.leverage_limit)
    positions = _scale_active_abs_sum(positions, gross_limit, active=active)

    if previous is not None and constraints.max_turnover is not None:
        delta = positions - previous
        turnover = delta.abs().sum()
        if float(turnover.item()) > constraints.max_turnover:
            positions = previous + delta * (constraints.max_turnover / turnover)
    return (
        torch.where(active, positions, previous) if previous is not None else positions
    )


def portfolio_exposures(positions: torch.Tensor) -> dict[str, float]:
    gross = float(positions.abs().sum().item())
    net = float(positions.sum().item())
    concentration = float(positions.square().sum().item())
    effective_bets = 0.0 if concentration <= 1e-12 else gross * gross / concentration
    return {
        "concentration": concentration,
        "effective_bets": effective_bets,
        "gross_exposure": gross,
        "net_exposure": net,
    }
