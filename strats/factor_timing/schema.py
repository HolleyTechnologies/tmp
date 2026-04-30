from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch


def validate_patch_aligned_lookback(
    lookback: int,
    *,
    patch_length: int,
    patch_stride: int | None = None,
) -> None:
    """
    Validates that the most recent timestep is retained by fixed patching.
    """

    if lookback < 1:
        raise ValueError("Factor timing lookback must be greater than zero.")
    if patch_length < 1:
        raise ValueError("Factor timing patch_length must be greater than zero.")
    stride = patch_length if patch_stride is None else patch_stride
    if stride < 1:
        raise ValueError("Factor timing patch_stride must be greater than zero.")
    if lookback < patch_length:
        raise ValueError("Factor timing lookback must be at least patch_length.")
    if (lookback - patch_length) % stride != 0:
        raise ValueError(
            "Factor timing lookback must satisfy "
            "(lookback - patch_length) % patch_stride == 0."
        )


@dataclass(frozen=True, slots=True)
class FactorEntity:
    """
    Entity observed by the strategy layer.
    """

    name: str
    region: str | None = None
    style: str | None = None
    tradable: bool = False
    group: str = "entity"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FactorEntity name must be non-empty.")
        if not self.group:
            raise ValueError("FactorEntity group must be non-empty.")


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    """
    Compact metadata for externally supplied or derived feature channels.
    """

    source: str | None = None
    transform: str | None = None
    inputs: tuple[str, ...] = ()
    lookback: int | None = None
    known_lag: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lookback is not None and self.lookback < 1:
            raise ValueError("FeatureProvenance lookback must be greater than zero.")
        if self.known_lag is not None and self.known_lag < 0:
            raise ValueError("FeatureProvenance known_lag cannot be negative.")


@dataclass(frozen=True, slots=True)
class FactorTimingUniverse:
    """
    Observed entities and the tradable action subset.
    """

    observed_entities: tuple[FactorEntity, ...]
    tradable_entities: tuple[FactorEntity, ...]

    def __post_init__(self) -> None:
        if not self.observed_entities:
            raise ValueError("Factor timing universe must contain observed entities.")
        if not self.tradable_entities:
            raise ValueError("Factor timing universe must contain action entities.")
        observed_names = self.observed_names
        if len(set(observed_names)) != len(observed_names):
            raise ValueError("Factor timing observed entity names must be unique.")
        tradable_names = self.tradable_names
        if len(set(tradable_names)) != len(tradable_names):
            raise ValueError("Factor timing tradable entity names must be unique.")
        missing = [name for name in tradable_names if name not in observed_names]
        if missing:
            raise ValueError(
                "Tradable entities must be present in the observed universe: "
                + ", ".join(missing)
            )

    @property
    def observed_names(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.observed_entities)

    @property
    def tradable_names(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.tradable_entities)

    @property
    def observed_entity_count(self) -> int:
        return len(self.observed_entities)

    @property
    def action_dim(self) -> int:
        return len(self.tradable_entities)

    def tradable_observation_indices(self) -> tuple[int, ...]:
        observed = {name: index for index, name in enumerate(self.observed_names)}
        return tuple(observed[name] for name in self.tradable_names)


@dataclass(frozen=True, slots=True)
class FactorTimingPanel:
    """
    Canonical strategy boundary for point-in-time-safe factor timing panels.
    """

    values: torch.Tensor
    dates: Sequence[object]
    entities: tuple[FactorEntity, ...]
    feature_names: tuple[str, ...]
    action_entities: tuple[str, ...]
    known_asof: torch.Tensor | None = None
    decision_times: torch.Tensor | None = None
    realized_forward_returns: torch.Tensor | None = None
    availability_mask: torch.Tensor | None = None
    action_availability_mask: torch.Tensor | None = None
    feature_provenance: Mapping[str, FeatureProvenance] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(
                "FactorTimingPanel values must have shape [date, entity, feature]."
            )
        date_count, entity_count, feature_count = self.values.shape
        if min(date_count, entity_count, feature_count) < 1:
            raise ValueError("FactorTimingPanel axes must be non-empty.")
        if len(self.dates) != date_count:
            raise ValueError("FactorTimingPanel dates must match the date axis.")
        if len(self.entities) != entity_count:
            raise ValueError("FactorTimingPanel entities must match the entity axis.")
        if len(self.feature_names) != feature_count:
            raise ValueError(
                "FactorTimingPanel feature_names must match the feature axis."
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("FactorTimingPanel feature names must be unique.")

        entity_names = self.entity_names
        if len(set(entity_names)) != len(entity_names):
            raise ValueError("FactorTimingPanel entity names must be unique.")
        if not self.action_entities:
            raise ValueError("FactorTimingPanel action_entities must be non-empty.")
        if len(set(self.action_entities)) != len(self.action_entities):
            raise ValueError("FactorTimingPanel action_entities must be unique.")
        missing = [name for name in self.action_entities if name not in entity_names]
        if missing:
            raise ValueError(
                "FactorTimingPanel action entities must be observed: "
                + ", ".join(missing)
            )

        self._validate_optional_tensors()
        self.validate_known_asof()

    @property
    def date_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def entity_count(self) -> int:
        return int(self.values.shape[1])

    @property
    def feature_count(self) -> int:
        return int(self.values.shape[2])

    @property
    def action_dim(self) -> int:
        return len(self.action_entities)

    @property
    def entity_names(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.entities)

    @property
    def universe(self) -> FactorTimingUniverse:
        by_name = {entity.name: entity for entity in self.entities}
        return FactorTimingUniverse(
            observed_entities=self.entities,
            tradable_entities=tuple(by_name[name] for name in self.action_entities),
        )

    def action_entity_indices(self) -> tuple[int, ...]:
        by_name = {name: index for index, name in enumerate(self.entity_names)}
        return tuple(by_name[name] for name in self.action_entities)

    def window(self, *, end: int, lookback: int) -> torch.Tensor:
        if lookback < 1:
            raise ValueError("FactorTimingPanel lookback must be greater than zero.")
        if end < lookback or end > self.date_count:
            raise ValueError("FactorTimingPanel window end is out of range.")
        return self.values[end - lookback : end].movedim(0, 1)

    def action_returns(self, date_index: int) -> torch.Tensor:
        if self.realized_forward_returns is None:
            raise ValueError("FactorTimingPanel has no realized_forward_returns.")
        if date_index < 0 or date_index >= self.date_count:
            raise ValueError("FactorTimingPanel return date_index is out of range.")
        return self.realized_forward_returns[date_index]

    def action_mask(self, date_index: int) -> torch.Tensor:
        if date_index < 0 or date_index >= self.date_count:
            raise ValueError("FactorTimingPanel mask date_index is out of range.")
        if self.action_availability_mask is None:
            return torch.ones(
                self.action_dim,
                dtype=torch.bool,
                device=self.values.device,
            )
        return self.action_availability_mask[date_index].to(
            device=self.values.device,
            dtype=torch.bool,
        )

    def validate_known_asof(self) -> None:
        if self.known_asof is None or self.decision_times is None:
            return
        known_asof = _expand_known_asof(self.known_asof, self.values.shape)
        decision_times = _decision_time_view(self.decision_times, self.date_count).to(
            device=known_asof.device,
            dtype=known_asof.dtype,
        )
        valid = known_asof <= decision_times
        if self.availability_mask is not None:
            mask = self.availability_mask.to(device=known_asof.device, dtype=torch.bool)
            valid = torch.where(mask, valid, torch.ones_like(valid, dtype=torch.bool))
        if not bool(valid.all().item()):
            raise ValueError(
                "FactorTimingPanel known_asof contains features unavailable at decision time."
            )

    def _validate_optional_tensors(self) -> None:
        if (
            self.availability_mask is not None
            and self.availability_mask.shape != self.values.shape
        ):
            raise ValueError(
                "FactorTimingPanel availability_mask must match values shape."
            )
        if (
            self.action_availability_mask is not None
            and self.action_availability_mask.shape
            != (self.date_count, self.action_dim)
        ):
            raise ValueError(
                "FactorTimingPanel action_availability_mask must have shape "
                "[date, action_dim]."
            )
        if (
            self.realized_forward_returns is not None
            and self.realized_forward_returns.shape
            != (
                self.date_count,
                self.action_dim,
            )
        ):
            raise ValueError(
                "FactorTimingPanel realized_forward_returns must have shape "
                "[date, action_dim]."
            )
        if self.known_asof is not None:
            _expand_known_asof(self.known_asof, self.values.shape)
        if self.decision_times is not None:
            _decision_time_view(self.decision_times, self.date_count)
        unknown = [
            name for name in self.feature_provenance if name not in self.feature_names
        ]
        if unknown:
            raise ValueError(
                "Feature provenance was supplied for unknown features: "
                + ", ".join(unknown)
            )


@dataclass(frozen=True, slots=True)
class FactorTimingModelConfig:
    """
    Model dimensions and patching policy for factor timing models.
    """

    input_dim: int
    observed_entity_count: int
    action_dim: int
    context_dim: int
    lookback: int
    patch_length: int
    model_dim: int = 32
    num_heads: int = 4
    num_temporal_layers: int = 2
    num_cross_sectional_layers: int = 1
    patch_stride: int | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("Factor timing lookback must be greater than zero.")
        if self.input_dim < 1:
            raise ValueError("Factor timing input_dim must be greater than zero.")
        if self.observed_entity_count < 1:
            raise ValueError(
                "Factor timing observed_entity_count must be greater than zero."
            )
        if self.action_dim < 1:
            raise ValueError("Factor timing action_dim must be greater than zero.")
        if self.action_dim > self.observed_entity_count:
            raise ValueError(
                "Factor timing action_dim cannot exceed observed_entity_count for "
                "entity-indexed actor heads."
            )
        if self.context_dim < 0:
            raise ValueError("Factor timing context_dim cannot be negative.")
        if self.model_dim < 1:
            raise ValueError("Factor timing model_dim must be greater than zero.")
        if self.num_heads < 1:
            raise ValueError("Factor timing num_heads must be greater than zero.")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("Factor timing model_dim must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Factor timing dropout must be in the interval [0, 1).")
        validate_patch_aligned_lookback(
            self.lookback,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
        )


@dataclass(slots=True)
class ForecastMoments:
    """
    Optional forecast moments used only by strategy-layer sizing.
    """

    action_mean: torch.Tensor | None = None
    action_std: torch.Tensor | None = None
    conviction: torch.Tensor | None = None
    expected_return: torch.Tensor | None = None
    forecast_uncertainty: torch.Tensor | None = None
    expected_volatility: torch.Tensor | None = None
    probability_positive: torch.Tensor | None = None
    downside_risk: torch.Tensor | None = None


@dataclass(slots=True)
class PositionSignal:
    """
    Raw bounded action plus optional strategy-local sizing information.
    """

    action: torch.Tensor
    moments: ForecastMoments | None = None


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """
    Simple strategy-layer portfolio projection limits.
    """

    gross_limit: float = 1.0
    net_limit: float | None = None
    leverage_limit: float | None = None
    max_position: float = 1.0
    max_turnover: float | None = None
    concentration_limit: float | None = None
    min_confidence_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.gross_limit < 0.0:
            raise ValueError("Portfolio gross_limit cannot be negative.")
        if self.net_limit is not None and self.net_limit < 0.0:
            raise ValueError("Portfolio net_limit cannot be negative.")
        if self.leverage_limit is not None and self.leverage_limit < 0.0:
            raise ValueError("Portfolio leverage_limit cannot be negative.")
        if self.max_position < 0.0:
            raise ValueError("Portfolio max_position cannot be negative.")
        if self.max_turnover is not None and self.max_turnover < 0.0:
            raise ValueError("Portfolio max_turnover cannot be negative.")
        if self.concentration_limit is not None and self.concentration_limit < 0.0:
            raise ValueError("Portfolio concentration_limit cannot be negative.")
        if not 0.0 <= self.min_confidence_scale <= 1.0:
            raise ValueError(
                "Portfolio min_confidence_scale must be in the interval [0, 1]."
            )


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """
    Daily reward penalty policy for factor timing.
    """

    transaction_cost_rate: float = 0.0
    turnover_penalty_rate: float = 0.0
    risk_penalty_rate: float = 0.0
    concentration_penalty_rate: float = 0.0
    volatility_floor: float = 1e-6
    clip_value: float = 5.0

    def __post_init__(self) -> None:
        if self.transaction_cost_rate < 0.0:
            raise ValueError("Reward transaction_cost_rate cannot be negative.")
        if self.turnover_penalty_rate < 0.0:
            raise ValueError("Reward turnover_penalty_rate cannot be negative.")
        if self.risk_penalty_rate < 0.0:
            raise ValueError("Reward risk_penalty_rate cannot be negative.")
        if self.concentration_penalty_rate < 0.0:
            raise ValueError("Reward concentration_penalty_rate cannot be negative.")
        if self.volatility_floor <= 0.0:
            raise ValueError("Reward volatility_floor must be greater than zero.")
        if self.clip_value <= 0.0:
            raise ValueError("Reward clip_value must be greater than zero.")


@dataclass(frozen=True, slots=True)
class RewardTerms:
    """
    Full scalar reward breakdown.
    """

    reward: float
    gross_pnl: float
    transaction_costs: float
    turnover_penalty: float
    risk_penalty: float
    concentration_penalty: float
    net_pnl: float
    ex_ante_portfolio_vol: float


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """
    Integer-indexed train, validation, and test window.
    """

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int
    train_indices: torch.Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (
            0
            <= self.train_start
            < self.train_end
            <= self.validation_start
            < self.validation_end
            <= self.test_start
            < self.test_end
        ):
            raise ValueError("WalkForwardWindow bounds must be ordered and non-empty.")
        if self.train_indices is not None and self.train_indices.ndim != 1:
            raise ValueError(
                "WalkForwardWindow train_indices must have shape [sample]."
            )


@dataclass(frozen=True, slots=True)
class ResearchTrialMetadata:
    """
    Compact metadata for research-trial auditability.
    """

    trial_id: str
    hypothesis: str | None = None
    researcher: str | None = None
    created_at: str | None = None
    feature_set: str | None = None
    model_name: str | None = None
    parameter_count: int | None = None
    train_sample_count: int | None = None
    validation_sample_count: int | None = None
    test_sample_count: int | None = None
    trial_count: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("ResearchTrialMetadata trial_id must be non-empty.")
        counts = (
            self.parameter_count,
            self.train_sample_count,
            self.validation_sample_count,
            self.test_sample_count,
            self.trial_count,
        )
        if any(value is not None and value < 0 for value in counts):
            raise ValueError("ResearchTrialMetadata counts cannot be negative.")


def _expand_known_asof(
    known_asof: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    if known_asof.ndim == 1:
        if known_asof.shape != (target_shape[0],):
            raise ValueError("known_asof [date] shape must match panel dates.")
        known_asof = known_asof.reshape(target_shape[0], 1, 1)
    elif known_asof.ndim == 2:
        if known_asof.shape == (target_shape[0], target_shape[2]):
            known_asof = known_asof.reshape(target_shape[0], 1, target_shape[2])
        elif known_asof.shape == (target_shape[0], target_shape[1]):
            known_asof = known_asof.reshape(target_shape[0], target_shape[1], 1)
        else:
            raise ValueError(
                "known_asof [date, *] shape must align with entity or feature axes."
            )
    elif known_asof.ndim != 3:
        raise ValueError("known_asof must have shape [date], [date, *], or values.")

    try:
        return torch.broadcast_to(known_asof, target_shape)
    except RuntimeError as error:
        raise ValueError("known_asof must be broadcastable to values shape.") from error


def _decision_time_view(
    decision_times: torch.Tensor,
    date_count: int,
) -> torch.Tensor:
    if decision_times.ndim != 1 or decision_times.shape[0] != date_count:
        raise ValueError("decision_times must have shape [date].")
    return decision_times.reshape(date_count, 1, 1)
