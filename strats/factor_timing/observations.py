from collections.abc import Mapping

import torch

from .schema import (
    FactorTimingPanel,
    FactorTimingUniverse,
    validate_patch_aligned_lookback,
)


FactorTimingObservation = dict[str, object]


def build_factor_timing_observation(
    *,
    panel: torch.Tensor,
    context: torch.Tensor | None = None,
    previous_positions: torch.Tensor,
    universe: FactorTimingUniverse,
    entity_ids: torch.Tensor | None = None,
    tradable_mask: torch.Tensor | None = None,
    metadata: Mapping[str, object] | None = None,
    patch_length: int | None = None,
    patch_stride: int | None = None,
) -> FactorTimingObservation:
    """
    Builds a replay-collatable observation mapping for SAC/PPO model systems.
    """

    if panel.ndim != 3:
        raise ValueError("Factor timing panel must have shape [entity, time, feature].")
    if panel.shape[0] != universe.observed_entity_count:
        raise ValueError("Panel entity count must match the observed universe.")
    if previous_positions.shape != (universe.action_dim,):
        raise ValueError("previous_positions must have shape [action_dim].")
    if context is None:
        context = torch.zeros(0, dtype=panel.dtype, device=panel.device)
    if context.ndim != 1:
        raise ValueError("Factor timing context must have shape [context_feature].")
    previous_positions = previous_positions.to(device=panel.device, dtype=panel.dtype)
    if patch_length is not None:
        validate_patch_aligned_lookback(
            int(panel.shape[1]),
            patch_length=patch_length,
            patch_stride=patch_stride,
        )

    if entity_ids is None:
        entity_ids = torch.arange(
            universe.observed_entity_count,
            device=panel.device,
            dtype=torch.long,
        )
    if entity_ids.shape != (universe.observed_entity_count,):
        raise ValueError("entity_ids must have shape [observed_entity].")

    if tradable_mask is None:
        tradable_mask = torch.ones(
            universe.action_dim,
            device=panel.device,
            dtype=torch.bool,
        )
    if tradable_mask.shape != (universe.action_dim,):
        raise ValueError("tradable_mask must have shape [action_dim].")

    observation: FactorTimingObservation = {
        "context": context.to(device=panel.device, dtype=panel.dtype),
        "entity_ids": entity_ids.to(device=panel.device, dtype=torch.long),
        "panel": panel,
        "previous_positions": previous_positions,
        "tradable_entity_indices": torch.tensor(
            universe.tradable_observation_indices(),
            device=panel.device,
            dtype=torch.long,
        ),
        "tradable_mask": tradable_mask.to(device=panel.device, dtype=torch.bool),
    }
    if metadata is not None:
        observation["metadata"] = dict(metadata)
    return observation


def build_factor_timing_observation_from_panel(
    source: FactorTimingPanel,
    *,
    end: int,
    lookback: int,
    previous_positions: torch.Tensor,
    context: torch.Tensor | None = None,
    entity_ids: torch.Tensor | None = None,
    tradable_mask: torch.Tensor | None = None,
    metadata: Mapping[str, object] | None = None,
    patch_length: int | None = None,
    patch_stride: int | None = None,
) -> FactorTimingObservation:
    """
    Builds an observation from a canonical `[date, entity, feature]` panel.
    """

    panel = source.window(end=end, lookback=lookback)
    panel_tradable_mask = tradable_mask
    if panel_tradable_mask is None and source.action_availability_mask is not None:
        panel_tradable_mask = source.action_mask(end - 1)
    merged_metadata: dict[str, object] = {
        "action_entities": source.action_entities,
        "date": source.dates[end - 1],
        "end": end,
        "entity_names": source.entity_names,
        "feature_names": source.feature_names,
        "lookback": lookback,
        **dict(source.metadata),
        **dict(metadata or {}),
    }
    observation = build_factor_timing_observation(
        panel=panel,
        context=context,
        previous_positions=previous_positions,
        universe=source.universe,
        entity_ids=entity_ids,
        tradable_mask=panel_tradable_mask,
        metadata=merged_metadata,
        patch_length=patch_length,
        patch_stride=patch_stride,
    )
    if source.availability_mask is not None:
        observation["availability_mask"] = source.availability_mask[
            end - lookback : end
        ].movedim(0, 1)
    return observation


def tensor_observation(observation: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """
    Returns only tensor entries from an observation mapping.
    """

    return {
        key: value
        for key, value in observation.items()
        if isinstance(value, torch.Tensor)
    }
