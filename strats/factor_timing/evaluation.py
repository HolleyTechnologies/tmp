import math

from collections.abc import Mapping, Sequence

import torch

from .portfolio import portfolio_exposures
from .schema import RewardTerms, WalkForwardWindow


def _returns_vector(returns: torch.Tensor) -> torch.Tensor:
    return returns.to(dtype=torch.float32).reshape(-1)


def max_drawdown(returns: torch.Tensor) -> float:
    returns = _returns_vector(returns)
    wealth = torch.cat(
        (
            torch.ones(1, device=returns.device, dtype=returns.dtype),
            torch.cumprod(1.0 + returns, dim=0),
        )
    )
    peak = torch.cummax(wealth, dim=0).values
    drawdown = wealth / peak.clamp_min(1e-12) - 1.0
    return float(drawdown.min().item())


def t_stat(returns: torch.Tensor) -> float:
    """
    Diagnostic t-statistic for the average return, not proof of alpha.
    """

    returns = _returns_vector(returns)
    scale = returns.std(unbiased=False).clamp_min(1e-12)
    return float((returns.mean() / scale * math.sqrt(returns.numel())).item())


def hac_t_stat(returns: torch.Tensor, *, max_lag: int | None = None) -> float:
    """
    Newey-West-style diagnostic t-statistic for autocorrelated returns.
    """

    returns = _returns_vector(returns)
    count = returns.numel()
    if count < 2:
        return 0.0

    lag_count = (
        round(4.0 * (count / 100.0) ** (2.0 / 9.0)) if max_lag is None else max_lag
    )
    lag_count = max(0, min(lag_count, count - 1))
    centered = returns - returns.mean()
    long_run_variance = (centered * centered).sum() / count
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        covariance = (centered[lag:] * centered[:-lag]).sum() / count
        long_run_variance = long_run_variance + 2.0 * weight * covariance

    standard_error = (long_run_variance.clamp_min(1e-12) / count).sqrt()
    return float((returns.mean() / standard_error).item())


def _shape_metrics(
    returns: torch.Tensor, volatility: torch.Tensor
) -> tuple[float, float]:
    if float(volatility.item()) <= 1e-12:
        return 0.0, 0.0
    standardized = (returns - returns.mean()) / volatility
    skew = standardized.pow(3).mean()
    excess_kurtosis = standardized.pow(4).mean() - 3.0
    return float(skew.item()), float(excess_kurtosis.item())


def _position_summary(positions: torch.Tensor) -> dict[str, float]:
    positions = positions.to(dtype=torch.float32)
    if positions.ndim == 1:
        exposures = portfolio_exposures(positions)
        return {
            **exposures,
            "average_concentration": exposures["concentration"],
            "average_effective_bets": exposures["effective_bets"],
            "average_gross_exposure": exposures["gross_exposure"],
            "average_net_exposure": exposures["net_exposure"],
        }
    if positions.ndim != 2:
        raise ValueError("Positions must have shape [time, action] or [action].")

    gross = positions.abs().sum(dim=1)
    net = positions.sum(dim=1)
    concentration = positions.square().sum(dim=1)
    effective_bets = torch.where(
        concentration > 1e-12,
        gross.square() / concentration.clamp_min(1e-12),
        torch.zeros_like(concentration),
    )
    return {
        "average_concentration": float(concentration.mean().item()),
        "average_effective_bets": float(effective_bets.mean().item()),
        "average_gross_exposure": float(gross.mean().item()),
        "average_net_exposure": float(net.mean().item()),
        "concentration": float(concentration.mean().item()),
        "effective_bets": float(effective_bets.mean().item()),
        "gross_exposure": float(gross.mean().item()),
        "net_exposure": float(net.mean().item()),
    }


def performance_summary(
    returns: torch.Tensor,
    *,
    periods_per_year: int = 252,
    turnover: torch.Tensor | None = None,
    transaction_costs: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    hac_lags: int | None = None,
) -> dict[str, float]:
    returns = _returns_vector(returns)
    if returns.numel() == 0:
        raise ValueError("Performance summary requires at least one return.")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be greater than zero.")

    mean = returns.mean()
    volatility = returns.std(unbiased=False)
    downside = torch.minimum(
        returns,
        torch.zeros((), dtype=returns.dtype, device=returns.device),
    )
    downside_vol = downside.square().mean().sqrt()
    annualized_return = float((mean * periods_per_year).item())
    annualized_volatility = float((volatility * math.sqrt(periods_per_year)).item())
    annualized_downside_volatility = float(
        (downside_vol * math.sqrt(periods_per_year)).item()
    )
    drawdown = max_drawdown(returns)
    skew, excess_kurtosis = _shape_metrics(returns, volatility)
    summary = {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "calmar": 0.0 if abs(drawdown) <= 1e-12 else annualized_return / abs(drawdown),
        "downside_volatility": annualized_downside_volatility,
        "excess_kurtosis": excess_kurtosis,
        "hac_t_stat": hac_t_stat(returns, max_lag=hac_lags),
        "hit_rate": float((returns > 0.0).to(dtype=torch.float32).mean().item()),
        "max_drawdown": drawdown,
        "sharpe": float(
            (mean / volatility.clamp_min(1e-12) * math.sqrt(periods_per_year)).item()
        ),
        "skew": skew,
        "sortino": float(
            (mean / downside_vol.clamp_min(1e-12) * math.sqrt(periods_per_year)).item()
        ),
        "t_stat": t_stat(returns),
    }
    if turnover is not None:
        if turnover.numel() != returns.numel():
            raise ValueError("turnover must align with returns.")
        summary["turnover"] = float(turnover.to(dtype=torch.float32).mean().item())
    if transaction_costs is not None:
        if transaction_costs.numel() != returns.numel():
            raise ValueError("transaction_costs must align with returns.")
        summary["transaction_cost_drag"] = float(
            transaction_costs.to(dtype=torch.float32).sum().item()
        )
    if positions is not None:
        if positions.ndim == 2 and positions.shape[0] != returns.numel():
            raise ValueError("positions must align with returns.")
        summary.update(_position_summary(positions))
    return summary


def factor_contribution(
    positions: torch.Tensor,
    realized_returns: torch.Tensor,
) -> torch.Tensor:
    if positions.shape != realized_returns.shape:
        raise ValueError("Positions and realized returns must share a shape.")
    return (
        positions.to(dtype=torch.float32) * realized_returns.to(dtype=torch.float32)
    ).sum(dim=0)


def named_factor_contribution(
    positions: torch.Tensor,
    realized_returns: torch.Tensor,
    action_entities: Sequence[str],
) -> dict[str, float]:
    contributions = factor_contribution(positions, realized_returns)
    if contributions.shape[0] != len(action_entities):
        raise ValueError("action_entities must align with the action dimension.")
    return {
        name: float(contributions[index].item())
        for index, name in enumerate(action_entities)
    }


def reward_terms_summary(terms: Sequence[RewardTerms]) -> dict[str, float]:
    if not terms:
        raise ValueError("reward_terms_summary requires at least one item.")
    fields = (
        "reward",
        "gross_pnl",
        "transaction_costs",
        "turnover_penalty",
        "risk_penalty",
        "concentration_penalty",
        "net_pnl",
        "ex_ante_portfolio_vol",
    )
    output: dict[str, float] = {}
    for name in fields:
        values = torch.tensor(
            [float(getattr(item, name)) for item in terms],
            dtype=torch.float32,
        )
        output[f"mean_{name}"] = float(values.mean().item())
        output[f"sum_{name}"] = float(values.sum().item())
    return output


def cost_sensitivity(
    gross_returns: torch.Tensor,
    turnover: torch.Tensor,
    cost_rates: Sequence[float],
    *,
    periods_per_year: int = 252,
    positions: torch.Tensor | None = None,
    hac_lags: int | None = None,
) -> dict[float, dict[str, float]]:
    gross_returns = _returns_vector(gross_returns)
    turnover = turnover.to(dtype=torch.float32).reshape(-1)
    if gross_returns.shape != turnover.shape:
        raise ValueError("gross_returns and turnover must align.")

    output: dict[float, dict[str, float]] = {}
    for rate in cost_rates:
        if rate < 0.0:
            raise ValueError("cost rates cannot be negative.")
        costs = turnover * rate
        output[float(rate)] = performance_summary(
            gross_returns - costs,
            periods_per_year=periods_per_year,
            turnover=turnover,
            transaction_costs=costs,
            positions=positions,
            hac_lags=hac_lags,
        )
    return output


def baseline_comparison(
    strategy_returns: torch.Tensor,
    baselines: Mapping[str, torch.Tensor],
    *,
    periods_per_year: int = 252,
    hac_lags: int | None = None,
) -> dict[str, dict[str, float]]:
    strategy_returns = _returns_vector(strategy_returns)
    output = {
        "strategy": performance_summary(
            strategy_returns,
            periods_per_year=periods_per_year,
            hac_lags=hac_lags,
        )
    }
    for name, baseline_returns in baselines.items():
        baseline_returns = _returns_vector(baseline_returns)
        if baseline_returns.shape != strategy_returns.shape:
            raise ValueError("Baseline returns must align with strategy returns.")
        output[name] = performance_summary(
            baseline_returns,
            periods_per_year=periods_per_year,
            hac_lags=hac_lags,
        )
        output[f"active_vs_{name}"] = performance_summary(
            strategy_returns - baseline_returns,
            periods_per_year=periods_per_year,
            hac_lags=hac_lags,
        )
    return output


def capacity_usage(
    positions: torch.Tensor,
    capacity: torch.Tensor,
    *,
    floor: float = 1e-12,
) -> dict[str, float]:
    if floor <= 0.0:
        raise ValueError("capacity floor must be greater than zero.")
    positions = positions.to(dtype=torch.float32)
    capacity = capacity.to(device=positions.device, dtype=positions.dtype).abs()
    try:
        capacity = torch.broadcast_to(capacity, positions.shape)
    except RuntimeError as error:
        raise ValueError("capacity must be broadcastable to positions.") from error
    usage = positions.abs() / capacity.clamp_min(floor)
    return {
        "average_capacity_usage": float(usage.mean().item()),
        "max_capacity_usage": float(usage.max().item()),
        "capacity_breach_rate": float(
            (usage > 1.0).to(dtype=torch.float32).mean().item()
        ),
    }


def parameter_count(model: object, *, trainable_only: bool = True) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise ValueError("model must expose a parameters() method.")
    total = 0
    for parameter in parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        total += parameter.numel()
    return int(total)


def walk_forward_sample_counts(
    windows: Sequence[WalkForwardWindow],
) -> list[dict[str, int]]:
    output: list[dict[str, int]] = []
    for index, window in enumerate(windows):
        train_count = (
            window.train_end - window.train_start
            if window.train_indices is None
            else int(window.train_indices.numel())
        )
        output.append(
            {
                "fold": index,
                "test_sample_count": window.test_end - window.test_start,
                "train_sample_count": train_count,
                "validation_sample_count": (
                    window.validation_end - window.validation_start
                ),
            }
        )
    return output


def subperiod_performance(
    returns: torch.Tensor,
    labels: Sequence[object],
    *,
    periods_per_year: int = 252,
    turnover: torch.Tensor | None = None,
    transaction_costs: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    hac_lags: int | None = None,
) -> dict[object, dict[str, float]]:
    if returns.shape[0] != len(labels):
        raise ValueError("Subperiod labels must align with returns.")
    if turnover is not None and turnover.shape[0] != len(labels):
        raise ValueError("Subperiod turnover must align with returns.")
    if transaction_costs is not None and transaction_costs.shape[0] != len(labels):
        raise ValueError("Subperiod transaction costs must align with returns.")
    if positions is not None and positions.shape[0] != len(labels):
        raise ValueError("Subperiod positions must align with returns.")

    output: dict[object, dict[str, float]] = {}
    for label in dict.fromkeys(labels):
        indices = torch.tensor(
            [index for index, item in enumerate(labels) if item == label],
            dtype=torch.long,
            device=returns.device,
        )
        output[label] = performance_summary(
            returns.index_select(0, indices),
            periods_per_year=periods_per_year,
            turnover=(
                None
                if turnover is None
                else turnover.to(device=returns.device).index_select(0, indices)
            ),
            transaction_costs=(
                None
                if transaction_costs is None
                else transaction_costs.to(device=returns.device).index_select(
                    0, indices
                )
            ),
            positions=(
                None
                if positions is None
                else positions.to(device=returns.device).index_select(0, indices)
            ),
            hac_lags=hac_lags,
        )
    return output
