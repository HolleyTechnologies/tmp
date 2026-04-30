from collections.abc import Mapping, Sequence

import torch

from ...contracts import SACTransition
from .schema import WalkForwardWindow


def expanding_walk_forward_windows(
    *,
    total_observations: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
    label_horizon: int = 1,
    embargo: int = 0,
) -> list[WalkForwardWindow]:
    """
    Builds simple expanding train windows with rolling validation/test tails.
    """

    if min(total_observations, train_size, validation_size, test_size) < 1:
        raise ValueError("Walk-forward sizes must be greater than zero.")
    step = test_size if step_size is None else step_size
    if step < 1:
        raise ValueError("Walk-forward step_size must be greater than zero.")
    _check_purge_inputs(label_horizon=label_horizon, embargo=embargo)

    windows: list[WalkForwardWindow] = []
    train_end = train_size
    while train_end + validation_size + test_size <= total_observations:
        validation_end = train_end + validation_size
        windows.append(
            WalkForwardWindow(
                train_start=0,
                train_end=train_end,
                validation_start=train_end,
                validation_end=validation_end,
                test_start=validation_end,
                test_end=validation_end + test_size,
                train_indices=purged_embargoed_train_indices(
                    total_observations=total_observations,
                    train_start=0,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=validation_end + test_size,
                    label_horizon=label_horizon,
                    embargo=embargo,
                ),
            )
        )
        train_end += step
    return windows


def rolling_walk_forward_windows(
    *,
    total_observations: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
    label_horizon: int = 1,
    embargo: int = 0,
) -> list[WalkForwardWindow]:
    """
    Builds rolling train windows with rolling validation/test tails.
    """

    if min(total_observations, train_size, validation_size, test_size) < 1:
        raise ValueError("Walk-forward sizes must be greater than zero.")
    step = test_size if step_size is None else step_size
    if step < 1:
        raise ValueError("Walk-forward step_size must be greater than zero.")
    _check_purge_inputs(label_horizon=label_horizon, embargo=embargo)

    windows: list[WalkForwardWindow] = []
    train_start = 0
    while train_start + train_size + validation_size + test_size <= total_observations:
        train_end = train_start + train_size
        validation_end = train_end + validation_size
        test_end = validation_end + test_size
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                validation_start=train_end,
                validation_end=validation_end,
                test_start=validation_end,
                test_end=test_end,
                train_indices=purged_embargoed_train_indices(
                    total_observations=total_observations,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                    label_horizon=label_horizon,
                    embargo=embargo,
                ),
            )
        )
        train_start += step
    return windows


def purged_embargoed_train_indices(
    *,
    total_observations: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    label_horizon: int = 1,
    embargo: int = 0,
) -> torch.Tensor:
    """
    Returns train indices excluding label overlap with an evaluation interval.
    """

    if total_observations < 1:
        raise ValueError("total_observations must be greater than zero.")
    if not 0 <= train_start < train_end <= total_observations:
        raise ValueError("Train bounds must be non-empty within total_observations.")
    if not 0 <= test_start < test_end <= total_observations:
        raise ValueError(
            "Evaluation bounds must be non-empty within total_observations."
        )
    _check_purge_inputs(label_horizon=label_horizon, embargo=embargo)

    indices = torch.arange(train_start, train_end, dtype=torch.long)
    label_start = indices
    label_end = indices + label_horizon
    overlaps_test = (label_start < test_end) & (label_end > test_start)
    embargo_end = min(total_observations, test_end + embargo)
    in_embargo = (indices >= test_end) & (indices < embargo_end)
    return indices[~(overlaps_test | in_embargo)]


def build_sac_transitions(
    *,
    observations: Sequence[Mapping[str, object]],
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: Sequence[bool | float | torch.Tensor] | torch.Tensor | None = None,
    discounts: torch.Tensor | None = None,
    metadata: Sequence[Mapping[str, object]] | None = None,
) -> list[SACTransition]:
    """
    Converts aligned observations into generic SAC transitions.
    """

    if len(observations) < 2:
        raise ValueError("At least two observations are required for transitions.")
    transition_count = len(observations) - 1
    if actions.shape[0] != transition_count or rewards.shape[0] != transition_count:
        raise ValueError("Actions and rewards must align with observation transitions.")
    if terminated is not None and len(terminated) != transition_count:
        raise ValueError("terminated flags must align with observation transitions.")
    if discounts is not None and discounts.shape[0] != transition_count:
        raise ValueError("Discounts must align with observation transitions.")
    if metadata is not None and len(metadata) != transition_count:
        raise ValueError("Metadata must align with observation transitions.")

    return [
        (
            SACTransition(
                observation=dict(observations[index]),
                action=actions[index],
                reward=rewards[index],
                next_observation=dict(observations[index + 1]),
                terminated=(
                    index == transition_count - 1
                    if terminated is None
                    else terminated[index]
                ),
                discount=None if discounts is None else discounts[index],
                metadata={} if metadata is None else dict(metadata[index]),
            )
        )
        for index in range(transition_count)
    ]


def _check_purge_inputs(*, label_horizon: int, embargo: int) -> None:
    if label_horizon < 1:
        raise ValueError("label_horizon must be greater than zero.")
    if embargo < 0:
        raise ValueError("embargo cannot be negative.")
