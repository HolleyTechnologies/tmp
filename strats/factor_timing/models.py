from collections.abc import Mapping

import torch

from torch import nn

from ...contracts import SACActionDistribution
from .schema import FactorTimingModelConfig


_FROZEN_LOG_STD = -5.0


def _tensor(observation: Mapping[str, object], key: str) -> torch.Tensor:
    value = observation[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Factor timing observation {key!r} must be a tensor.")
    return value


def _batched_panel(panel: torch.Tensor) -> torch.Tensor:
    if panel.ndim == 3:
        return panel.unsqueeze(0)
    if panel.ndim == 4:
        return panel
    raise ValueError(
        "Factor timing panel must have shape [batch, entity, time, feature]."
    )


def _batched_context(context: torch.Tensor) -> torch.Tensor:
    if context.ndim == 1:
        return context.unsqueeze(0)
    if context.ndim == 2:
        return context
    raise ValueError("Factor timing context must have shape [batch, context_feature].")


def _batched_positions(positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim == 1:
        return positions.unsqueeze(0)
    if positions.ndim == 2:
        return positions
    raise ValueError("Factor timing positions must have shape [batch, action_dim].")


def _index_vector(
    indices: torch.Tensor,
    *,
    batch_size: int,
    config: FactorTimingModelConfig,
    device: torch.device,
) -> torch.Tensor:
    if indices.ndim == 2:
        if indices.shape[0] != batch_size:
            raise ValueError(
                "Batched tradable_entity_indices must align with the observation batch."
            )
        first = indices[0]
        if bool((indices != first.unsqueeze(0)).any().item()):
            raise ValueError(
                "Batched tradable_entity_indices must be identical across a replay batch."
            )
        indices = first
    if indices.ndim != 1:
        raise ValueError("tradable_entity_indices must have shape [action_dim].")
    indices = indices.to(device=device, dtype=torch.long)
    _validate_indices(indices, config)
    return indices


def _optional_tensor(
    observation: Mapping[str, object],
    key: str,
) -> torch.Tensor | None:
    value = observation.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Factor timing observation {key!r} must be a tensor.")
    return value


def _validate_panel(panel: torch.Tensor, config: FactorTimingModelConfig) -> None:
    if panel.shape[1] != config.observed_entity_count:
        raise ValueError("Observation panel entity count does not match model config.")
    if panel.shape[2] != config.lookback:
        raise ValueError("Observation panel lookback does not match model config.")
    if panel.shape[3] != config.input_dim:
        raise ValueError("Observation panel feature count does not match model config.")


def _validate_context(context: torch.Tensor, batch_size: int, context_dim: int) -> None:
    if context.shape != (batch_size, context_dim):
        raise ValueError("Observation context shape does not match model config.")


def _validate_positions(
    positions: torch.Tensor,
    batch_size: int,
    action_dim: int,
    *,
    name: str,
) -> None:
    if positions.shape != (batch_size, action_dim):
        raise ValueError(f"{name} shape does not match model config.")


def _validate_indices(indices: torch.Tensor, config: FactorTimingModelConfig) -> None:
    if indices.shape != (config.action_dim,):
        raise ValueError("tradable_entity_indices must have shape [action_dim].")
    if bool((indices < 0).any().item()) or bool(
        (indices >= config.observed_entity_count).any().item()
    ):
        raise ValueError("tradable_entity_indices are out of observed-entity range.")


def _tradable_mask(
    observation: Mapping[str, object],
    *,
    batch_size: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    mask = _optional_tensor(observation, "tradable_mask")
    if mask is None:
        return torch.ones(batch_size, action_dim, device=device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2:
        raise ValueError("tradable_mask must have shape [batch, action_dim].")
    if mask.shape != (batch_size, action_dim):
        raise ValueError("tradable_mask shape does not match model config.")
    return mask.to(device=device, dtype=torch.bool)


def _atanh_bounded(action: torch.Tensor) -> torch.Tensor:
    action = action.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(action) - torch.log1p(-action))


class FactorTimingActor(nn.Module):
    """
    SAC actor that maps a larger observed factor panel to tradable action slots.
    """

    def __init__(self, config: FactorTimingModelConfig) -> None:
        super().__init__()
        from indoraptor.machinelearning.models import SpaceTimeEncoder

        self.config = config
        self.backbone = SpaceTimeEncoder(
            input_dim=config.input_dim,
            model_dim=config.model_dim,
            num_temporal_layers=config.num_temporal_layers,
            num_cross_sectional_layers=config.num_cross_sectional_layers,
            num_heads=config.num_heads,
            patch_length=config.patch_length,
            patch_stride=config.patch_stride,
            dropout=config.dropout,
            num_entities=config.observed_entity_count,
            context_dim=config.context_dim if config.context_dim > 0 else None,
            pooling="last",
        )
        self.mean_head = nn.Sequential(
            nn.LayerNorm(config.model_dim + 1),
            nn.Linear(config.model_dim + 1, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((config.action_dim,), -1.0))

    def distribution(self, observation: Mapping[str, object]) -> SACActionDistribution:
        panel = _batched_panel(_tensor(observation, "panel"))
        _validate_panel(panel, self.config)
        batch_size = panel.shape[0]
        context = _batched_context(_tensor(observation, "context"))
        _validate_context(context, batch_size, self.config.context_dim)
        context = context.to(device=panel.device, dtype=panel.dtype)
        previous_positions = _batched_positions(
            _tensor(observation, "previous_positions")
        )
        _validate_positions(
            previous_positions,
            batch_size,
            self.config.action_dim,
            name="previous_positions",
        )
        previous_positions = previous_positions.to(
            device=panel.device, dtype=panel.dtype
        )
        entity_ids = _tensor(observation, "entity_ids")
        if entity_ids.ndim == 1:
            entity_ids = entity_ids.to(device=panel.device, dtype=torch.long)
        else:
            entity_ids = entity_ids.to(device=panel.device, dtype=torch.long)
        context_input = context if self.config.context_dim > 0 else None
        latent = self.backbone(
            panel,
            context=context_input,
            entity_ids=entity_ids,
        )
        indices = _index_vector(
            _tensor(observation, "tradable_entity_indices"),
            batch_size=batch_size,
            config=self.config,
            device=latent.device,
        )
        tradable_latent = latent.index_select(1, indices)
        features = torch.cat(
            (tradable_latent, previous_positions.unsqueeze(-1)), dim=-1
        )
        mean = self.mean_head(features).squeeze(-1)
        tradable_mask = _tradable_mask(
            observation,
            batch_size=batch_size,
            action_dim=self.config.action_dim,
            device=mean.device,
        )
        frozen_mean = _atanh_bounded(previous_positions.to(device=mean.device))
        mean = torch.where(tradable_mask, mean, frozen_mean)
        log_std = self.log_std.to(device=mean.device, dtype=mean.dtype).expand_as(mean)
        log_std = torch.where(
            tradable_mask,
            log_std,
            torch.full_like(log_std, _FROZEN_LOG_STD),
        )
        return SACActionDistribution(
            mean=mean,
            log_std=log_std,
        )


class FactorTimingTwinCritic(nn.Module):
    """
    Twin critic for factor timing SAC systems.
    """

    def __init__(self, config: FactorTimingModelConfig) -> None:
        super().__init__()
        from indoraptor.machinelearning.models import SpaceTimeEncoder

        self.config = config
        self.backbone = SpaceTimeEncoder(
            input_dim=config.input_dim,
            model_dim=config.model_dim,
            num_temporal_layers=config.num_temporal_layers,
            num_cross_sectional_layers=config.num_cross_sectional_layers,
            num_heads=config.num_heads,
            patch_length=config.patch_length,
            patch_stride=config.patch_stride,
            dropout=config.dropout,
            num_entities=config.observed_entity_count,
            context_dim=config.context_dim if config.context_dim > 0 else None,
            pooling="last",
        )
        slot_dim = config.model_dim + 4
        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, config.model_dim),
            nn.GELU(),
        )
        critic_dim = 2 * config.model_dim + 4
        self.q1 = nn.Sequential(
            nn.LayerNorm(critic_dim),
            nn.Linear(critic_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.LayerNorm(critic_dim),
            nn.Linear(critic_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )

    def q_values(
        self,
        observation: Mapping[str, object],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        panel = _batched_panel(_tensor(observation, "panel"))
        _validate_panel(panel, self.config)
        batch_size = panel.shape[0]
        context = _batched_context(_tensor(observation, "context"))
        _validate_context(context, batch_size, self.config.context_dim)
        context = context.to(device=panel.device, dtype=panel.dtype)
        entity_ids = _tensor(observation, "entity_ids").to(
            device=panel.device,
            dtype=torch.long,
        )
        action = _batched_positions(action).to(device=panel.device, dtype=panel.dtype)
        _validate_positions(
            action,
            batch_size,
            self.config.action_dim,
            name="action",
        )
        previous_positions = _batched_positions(
            _tensor(observation, "previous_positions")
        ).to(device=panel.device, dtype=panel.dtype)
        _validate_positions(
            previous_positions,
            batch_size,
            self.config.action_dim,
            name="previous_positions",
        )
        context_input = context if self.config.context_dim > 0 else None
        latent = self.backbone(
            panel,
            context=context_input,
            entity_ids=entity_ids,
        )
        indices = _index_vector(
            _tensor(observation, "tradable_entity_indices"),
            batch_size=batch_size,
            config=self.config,
            device=latent.device,
        )
        tradable_mask = _tradable_mask(
            observation,
            batch_size=batch_size,
            action_dim=self.config.action_dim,
            device=latent.device,
        )
        effective_action = torch.where(tradable_mask, action, previous_positions)
        action_delta = effective_action - previous_positions
        tradable_latent = latent.index_select(1, indices)
        slot_features = torch.cat(
            (
                tradable_latent,
                previous_positions.unsqueeze(-1),
                effective_action.unsqueeze(-1),
                action_delta.unsqueeze(-1),
                tradable_mask.to(dtype=panel.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        slot_summary = self.slot_encoder(slot_features).mean(dim=1)
        portfolio_summary = torch.stack(
            (
                previous_positions.abs().sum(dim=1),
                effective_action.abs().sum(dim=1),
                effective_action.sum(dim=1),
                action_delta.abs().sum(dim=1),
            ),
            dim=-1,
        )
        critic_input = torch.cat(
            (latent.mean(dim=1), slot_summary, portfolio_summary), dim=-1
        )
        return self.q1(critic_input), self.q2(critic_input)


class FactorTimingSACSystem(nn.Module):
    """
    Minimal actor/critic system compatible with the generic SAC update path.
    """

    def __init__(self, config: FactorTimingModelConfig) -> None:
        super().__init__()
        self.actor = FactorTimingActor(config)
        self.critic = FactorTimingTwinCritic(config)
        self.target_critic = FactorTimingTwinCritic(config)
        self.log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float32))
