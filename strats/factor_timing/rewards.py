import torch

from .schema import RewardConfig, RewardTerms


def compute_factor_timing_reward(
    *,
    positions: torch.Tensor,
    previous_positions: torch.Tensor,
    realized_returns: torch.Tensor,
    ex_ante_portfolio_vol: float | torch.Tensor,
    config: RewardConfig | None = None,
) -> RewardTerms:
    """
    Computes the clipped volatility-normalized factor timing reward.
    """

    config = RewardConfig() if config is None else config
    positions = positions.to(dtype=torch.float32)
    if positions.ndim != 1 or positions.numel() < 1:
        raise ValueError("positions must have shape [action_dim].")
    previous_positions = previous_positions.to(
        device=positions.device, dtype=torch.float32
    )
    realized_returns = realized_returns.to(device=positions.device, dtype=torch.float32)
    if previous_positions.shape != positions.shape:
        raise ValueError("previous_positions must have shape [action_dim].")
    if realized_returns.shape != positions.shape:
        raise ValueError("realized_returns must have shape [action_dim].")
    ex_ante_vol = torch.as_tensor(
        ex_ante_portfolio_vol,
        device=positions.device,
        dtype=torch.float32,
    ).clamp_min(0.0)
    if ex_ante_vol.numel() != 1:
        raise ValueError("ex_ante_portfolio_vol must be scalar.")
    vol = ex_ante_vol.clamp_min(config.volatility_floor)

    turnover = (positions - previous_positions).abs().sum()
    concentration = positions.square().sum()
    gross_pnl = (positions * realized_returns).sum()
    transaction_costs = config.transaction_cost_rate * turnover
    turnover_penalty = config.turnover_penalty_rate * turnover
    risk_penalty = config.risk_penalty_rate * ex_ante_vol
    concentration_penalty = config.concentration_penalty_rate * concentration
    net_pnl = (
        gross_pnl
        - transaction_costs
        - turnover_penalty
        - risk_penalty
        - concentration_penalty
    )
    reward = (net_pnl / vol).clamp(-config.clip_value, config.clip_value)
    return RewardTerms(
        reward=float(reward.item()),
        gross_pnl=float(gross_pnl.item()),
        transaction_costs=float(transaction_costs.item()),
        turnover_penalty=float(turnover_penalty.item()),
        risk_penalty=float(risk_penalty.item()),
        concentration_penalty=float(concentration_penalty.item()),
        net_pnl=float(net_pnl.item()),
        ex_ante_portfolio_vol=float(vol.item()),
    )
