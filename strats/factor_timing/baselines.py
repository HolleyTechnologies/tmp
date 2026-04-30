import torch


def zero_positions(
    action_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if action_dim < 1:
        raise ValueError("action_dim must be greater than zero.")
    return torch.zeros(action_dim, dtype=dtype, device=device)


def equal_weight_positions(
    action_dim: int,
    *,
    gross: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if action_dim < 1:
        raise ValueError("action_dim must be greater than zero.")
    if gross < 0.0:
        raise ValueError("gross cannot be negative.")
    return torch.full(
        (action_dim,),
        gross / action_dim,
        dtype=dtype,
        device=device,
    )


def inverse_volatility_positions(
    volatility: torch.Tensor,
    *,
    gross: float = 1.0,
    floor: float = 1e-6,
) -> torch.Tensor:
    if volatility.ndim != 1:
        raise ValueError("volatility must have shape [action_dim].")
    if gross < 0.0:
        raise ValueError("gross cannot be negative.")
    weights = 1.0 / volatility.to(dtype=torch.float32).abs().clamp_min(floor)
    scale = weights.sum().clamp_min(floor)
    return gross * weights / scale


def positions_from_scores(
    scores: torch.Tensor,
    *,
    gross: float = 1.0,
    floor: float = 1e-12,
) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must have shape [action_dim].")
    if gross < 0.0:
        raise ValueError("gross cannot be negative.")
    scores = scores.to(dtype=torch.float32)
    exposure = scores.abs().sum()
    if float(exposure.item()) <= floor:
        return torch.zeros_like(scores)
    return gross * scores / exposure


def momentum_positions(
    returns: torch.Tensor,
    *,
    lookback: int | None = None,
    gross: float = 1.0,
) -> torch.Tensor:
    if returns.ndim != 2:
        raise ValueError("returns must have shape [time, action_dim].")
    if returns.shape[0] < 1:
        raise ValueError("returns must contain at least one timestep.")
    window = returns if lookback is None else returns[-lookback:]
    if window.shape[0] < 1:
        raise ValueError("lookback selected no timesteps.")
    return positions_from_scores(
        window.to(dtype=torch.float32).mean(dim=0), gross=gross
    )


def reversal_positions(
    returns: torch.Tensor,
    *,
    lookback: int | None = None,
    gross: float = 1.0,
) -> torch.Tensor:
    return -momentum_positions(returns, lookback=lookback, gross=gross)
