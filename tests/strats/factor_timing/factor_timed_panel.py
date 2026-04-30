"""
Synthetic canonical-panel SAC experiment for the factor timing strategy layer.
"""

import importlib.util
import math
import pathlib
import random
import sys

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
INDORAPTOR_SRC = ROOT.parent / "indoraptor" / "src"

for path in (SRC_ROOT, INDORAPTOR_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MANDATORY_RUNTIME_DEPENDENCIES = (
    "indoraptor",
    "pydantic",
    "torch",
)
MISSING_MANDATORY_RUNTIME_DEPENDENCIES = tuple(
    dependency
    for dependency in MANDATORY_RUNTIME_DEPENDENCIES
    if importlib.util.find_spec(dependency) is None
)
RUNTIME_READY = not MISSING_MANDATORY_RUNTIME_DEPENDENCIES

if RUNTIME_READY:
    import torch


FEATURE_NAMES = (
    "log_return",
    "delta_log_volume",
    "ewma_volatility_fast",
    "ewma_volatility_slow",
    "volatility_ratio",
    "vol_scaled_return",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

CONTEXT_DIM = 5
DATE_COUNT = 360
LOOKBACK = 24
PATCH_LENGTH = 6
PATCH_STRIDE = 6
MODEL_DIM = 24
VOLATILITY_LOOKBACK = 60

BASELINE_NAMES = (
    "zero",
    "equal_weight",
    "inverse_vol",
    "momentum",
    "reversal",
)
COST_SENSITIVITY_RATES = (0.0, 0.0005, 0.0010, 0.0020)
DIAGNOSTIC_LABEL = "synthetic diagnostic, not expected live performance"


@dataclass(frozen=True, slots=True)
class SyntheticPanelExperimentSummary:
    """
    Compact result from the canonical-panel synthetic factor timing experiment.
    """

    seed: int
    diagnostic_label: str
    factor_basket_assumption: str
    date_count: int
    train_sample_count: int
    validation_sample_count: int
    evaluation_sample_count: int
    replay_size: int
    update_steps: int
    action_dim: int
    observed_entity_count: int
    feature_count: int
    lookback: int
    parameter_count: int
    train_samples_per_parameter: float
    masked_action_rate: float
    initial_validation_net_pnl: float
    final_validation_net_pnl: float
    evaluation_net_pnl: float
    evaluation_sharpe: float
    evaluation_turnover: float
    last_actor_loss: float
    last_critic_loss: float
    last_temperature: float | None
    baseline_sharpes: Mapping[str, float]
    factor_contribution: Mapping[str, float]
    reward_terms: Mapping[str, float]
    cost_sensitivity_sharpes: Mapping[str, float]
    cost_sensitivity_net_pnls: Mapping[str, float]
    regime_counts: Mapping[str, int]
    regime_net_pnls: Mapping[str, float]
    capacity_usage: Mapping[str, float]


def _build_synthetic_factor_timing_panel(
    *,
    seed: int = 29,
    date_count: int = DATE_COUNT,
) -> object:
    """
    Builds the only synthetic component: a canonical point-in-time panel source.
    """

    from indosteg.strats.factor_timing import (
        FactorEntity,
        FactorTimingPanel,
        FeatureProvenance,
        delta_log,
        ewma_volatility,
        vol_scaled_returns,
        volatility_ratio,
    )

    if date_count < LOOKBACK + 120:
        raise ValueError("Synthetic panel date_count is too short for the experiment.")

    action_entity_names = (
        "us_beta",
        "us_residual_volatility",
        "us_momentum",
        "us_size",
        "us_transient",
    )
    # These action streams are synthetic returns for upstream-built neutral baskets.
    action_entity_metadata = {
        "basket_type": "long_short_factor_proxy",
        "neutrality_assumption": "approx_delta_beta_neutral",
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    entities = (
        FactorEntity(
            "market_risk_context",
            region="Global",
            group="context",
            metadata={"role": "context", "family": "market_risk"},
        ),
        FactorEntity(
            "us_momentum",
            region="US",
            style="momentum",
            tradable=True,
            group="factor",
            metadata={
                "role": "action",
                "family": "momentum",
                "region": "US",
                **action_entity_metadata,
            },
        ),
        FactorEntity(
            "apac_beta_context",
            region="APAC",
            style="beta",
            group="context",
            metadata={"role": "context", "family": "beta", "lead_lag": "lead"},
        ),
        FactorEntity(
            "us_beta",
            region="US",
            style="beta",
            tradable=True,
            group="factor",
            metadata={
                "role": "action",
                "family": "beta",
                "region": "US",
                **action_entity_metadata,
            },
        ),
        FactorEntity(
            "us_size",
            region="US",
            style="size",
            tradable=True,
            group="factor",
            metadata={
                "role": "action",
                "family": "size",
                "region": "US",
                **action_entity_metadata,
            },
        ),
        FactorEntity(
            "emea_momentum_context",
            region="EMEA",
            style="momentum",
            group="context",
            metadata={"role": "context", "family": "momentum", "lead_lag": "lead"},
        ),
        FactorEntity(
            "us_residual_volatility",
            region="US",
            style="residual_volatility",
            tradable=True,
            group="factor",
            metadata={
                "role": "action",
                "family": "residual_volatility",
                "region": "US",
                **action_entity_metadata,
            },
        ),
        FactorEntity(
            "apac_residual_volatility_context",
            region="APAC",
            style="residual_volatility",
            group="context",
            metadata={
                "role": "context",
                "family": "residual_volatility",
                "lead_lag": "lead",
            },
        ),
        FactorEntity(
            "us_transient",
            region="US",
            style="transient",
            tradable=True,
            group="factor",
            metadata={
                "role": "action",
                "family": "transient",
                "region": "US",
                **action_entity_metadata,
            },
        ),
    )
    entity_names = tuple(entity.name for entity in entities)
    action_indices = torch.tensor(
        [entity_names.index(name) for name in action_entity_names],
        dtype=torch.long,
    )
    action_entity_mask = torch.zeros(len(entities), dtype=torch.bool)
    action_entity_mask[action_indices] = True
    action_slot_by_name = {
        name: index for index, name in enumerate(action_entity_names)
    }
    context_index = {name: entity_names.index(name) for name in entity_names}

    entity_count = len(entities)
    action_dim = len(action_entity_names)
    base_volatility = torch.tensor(
        [0.0090, 0.0120, 0.0100, 0.0080, 0.0130],
        dtype=torch.float32,
    )
    returns = torch.zeros(entity_count, date_count, dtype=torch.float32)
    signed_flow = torch.zeros(action_dim, date_count, dtype=torch.float32)
    log_volume = torch.full((entity_count, date_count), 12.0, dtype=torch.float32)
    log_volatility = torch.zeros(action_dim, date_count, dtype=torch.float32)
    log_volatility[:, 0] = torch.log(base_volatility)
    regime_labels = ["warmup"] * date_count

    risk_state = torch.zeros(date_count, dtype=torch.float32)
    stress_state = torch.zeros(date_count, dtype=torch.float32)
    momentum_state = torch.zeros(date_count, dtype=torch.float32)
    size_state = torch.zeros(date_count, dtype=torch.float32)
    liquidity_state = torch.zeros(date_count, dtype=torch.float32)

    style_loading = torch.tensor(
        [0.75, -0.25, 0.15, 0.35, 0.10],
        dtype=torch.float32,
    )
    action_offsets = torch.tensor([0.12, 0.20, 0.08, -0.05, 0.24], dtype=torch.float32)

    for step in range(1, date_count):
        time_value = float(step)
        regime_shift = 0 if step < int(0.65 * date_count) else 7
        if step >= int(0.82 * date_count):
            regime_shift = 13
        regime_phase = (step + regime_shift) % 60
        scheduled_label = (
            "risk_on"
            if regime_phase < 10
            else "stress_rise"
            if regime_phase < 20
            else "trend"
            if regime_phase < 30
            else "liquidity"
            if regime_phase < 40
            else "transient"
            if regime_phase < 50
            else "reversal"
        )
        regime_boost = {
            "liquidity": 1.0 if scheduled_label == "liquidity" else 0.0,
            "reversal": 1.0 if scheduled_label == "reversal" else 0.0,
            "risk_on": 1.0 if scheduled_label == "risk_on" else 0.0,
            "stress_rise": 1.0 if scheduled_label == "stress_rise" else 0.0,
            "transient": 1.0 if scheduled_label == "transient" else 0.0,
            "trend": 1.0 if scheduled_label == "trend" else 0.0,
        }
        stress_pulse = 0.90 * math.exp(
            -(((time_value - 115.0) / 20.0) ** 2)
        ) + 0.75 * math.exp(-(((time_value - 255.0) / 24.0) ** 2))
        risk_noise = 0.030 * torch.randn((), generator=generator)
        momentum_noise = 0.035 * torch.randn((), generator=generator)
        size_noise = 0.035 * torch.randn((), generator=generator)
        liquidity_noise = 0.030 * torch.randn((), generator=generator)
        risk_state[step] = (
            0.985 * risk_state[step - 1]
            + 0.030 * math.sin(time_value / 31.0)
            - 0.018 * stress_pulse
            + risk_noise
        )
        stress_state[step] = (
            0.900 * stress_state[step - 1]
            + 0.080 * max(0.0, -float(risk_state[step].item()))
            + 0.070 * stress_pulse
            + 0.010 * torch.rand((), generator=generator)
        ).clamp(0.0, 2.0)
        momentum_state[step] = (
            0.860 * momentum_state[step - 1]
            + 0.045 * math.sin(time_value / 17.0)
            + 0.020 * math.sin(time_value / 43.0)
            + momentum_noise
        )
        size_state[step] = (
            0.965 * size_state[step - 1]
            + 0.050 * math.cos(time_value / 39.0)
            - 0.030 * float(stress_state[step].item())
            + size_noise
        )
        liquidity_state[step] = (
            0.940 * liquidity_state[step - 1]
            + 0.055 * math.sin((time_value + 18.0) / 29.0)
            + 0.035 * float(risk_state[step].item())
            - 0.055 * float(stress_state[step].item())
            + liquidity_noise
        )

        risk_on = math.tanh(
            float(risk_state[step - 1].item())
            - 0.55 * float(stress_state[step - 1].item())
        )
        stress = float(stress_state[step - 1].item())
        stress_change = float((stress_state[step - 1] - stress_state[step - 2]).item())
        momentum = math.tanh(float(momentum_state[step - 1].item()))
        size_cycle = math.tanh(float(size_state[step - 1].item()))
        liquidity = math.tanh(float(liquidity_state[step - 1].item()))
        liquidity_change = math.tanh(
            float((liquidity_state[step - 1] - liquidity_state[step - 4]).item())
            if step >= 4
            else 0.0
        )
        trend_gate = 0.5 + 0.5 * math.sin(time_value / 24.0 + 0.35)
        risk_on_gate = 0.5 + 0.5 * math.sin(time_value / 38.0 - 0.25)
        stress_gate = min(1.0, max(0.0, 2.2 * stress_change + 0.35 * stress))
        liquidity_gate = 0.5 + 0.5 * math.cos(time_value / 35.0 + 0.4)
        transient_gate = 0.5 + 0.5 * math.sin(time_value / 8.0)
        trend_gate = max(trend_gate, 0.85 * regime_boost["trend"])
        risk_on_gate = max(risk_on_gate, 0.85 * regime_boost["risk_on"])
        stress_gate = max(stress_gate, 0.80 * regime_boost["stress_rise"])
        liquidity_gate = max(liquidity_gate, 0.85 * regime_boost["liquidity"])
        transient_gate = max(transient_gate, 0.85 * regime_boost["transient"])

        common_volatility = 0.0065 * (1.0 + 0.70 * stress)
        common_shock = common_volatility * torch.randn((), generator=generator)
        apac_beta = 0.0032 * risk_on * risk_on_gate - 0.0013 * stress
        apac_residual = 0.0038 * stress_change + 0.0026 * stress
        emea_momentum = 0.0022 * momentum * trend_gate - 0.0018 * stress_gate
        returns[context_index["market_risk_context"], step] = (
            0.85 * common_shock
            - 0.0020 * stress
            + 0.0030 * torch.randn((), generator=generator)
        )
        returns[context_index["apac_beta_context"], step] = (
            apac_beta + 0.0050 * torch.randn((), generator=generator)
        )
        returns[context_index["apac_residual_volatility_context"], step] = (
            apac_residual + 0.0050 * torch.randn((), generator=generator)
        )
        returns[context_index["emea_momentum_context"], step] = (
            emea_momentum + 0.0050 * torch.randn((), generator=generator)
        )

        flow_previous = signed_flow[:, step - 1]
        crowding = flow_previous.abs().mean()
        stress_peak = max(
            0.0,
            float(stress_state[max(0, step - 4) : step].max().item()) - stress,
        )
        unwind_pressure = torch.sigmoid(4.0 * (crowding - 0.62)).item()
        reversal_pressure = (
            0.40 * max(0.0, -momentum)
            + 0.50 * unwind_pressure
            + 0.58 * stress_peak
            + 0.30 * max(0.0, -liquidity_change)
            + 0.48 * regime_boost["reversal"]
        )
        beta_lead = float(returns[context_index["apac_beta_context"], step - 1].item())
        residual_lead = float(
            returns[context_index["apac_residual_volatility_context"], step - 1].item()
        )
        momentum_lead = float(
            returns[context_index["emea_momentum_context"], step - 1].item()
        )

        alpha = torch.tensor(
            [
                0.0056 * risk_on * risk_on_gate
                + 0.0032 * regime_boost["risk_on"]
                - 0.0031 * stress
                + 0.19 * beta_lead,
                0.0062 * stress_gate
                + 0.0040 * stress_change
                + 0.0036 * regime_boost["stress_rise"]
                + 0.19 * residual_lead
                - 0.0010 * risk_on,
                0.0028 * momentum * trend_gate
                + 0.0016 * regime_boost["trend"]
                + 0.09 * momentum_lead
                - 0.0044 * reversal_pressure,
                0.0054
                * liquidity_gate
                * (0.25 * size_cycle + 0.75 * liquidity + 1.05 * liquidity_change)
                + 0.0035 * regime_boost["liquidity"]
                + 0.0015 * risk_on
                - 0.0030 * stress,
                0.0064
                * float(flow_previous[action_slot_by_name["us_transient"]].item())
                * transient_gate
                + 0.0032 * regime_boost["transient"]
                - 0.0048 * unwind_pressure,
            ],
            dtype=torch.float32,
        )
        alpha = (
            alpha
            + 0.0011 * flow_previous
            - 0.0037 * reversal_pressure * (flow_previous.sign() * flow_previous.abs())
        )

        target_log_volatility = torch.log(base_volatility) + (
            0.42 * stress
            + 0.12 * flow_previous.abs()
            + 0.070 * torch.randn(action_dim, generator=generator)
        )
        log_volatility[:, step] = (
            0.910 * log_volatility[:, step - 1] + 0.090 * target_log_volatility
        )
        volatility = log_volatility[:, step].exp().clamp(0.003, 0.045)
        common_loading = min(0.78, 0.25 + 0.34 * stress)
        idiosyncratic = 0.92 * torch.randn(action_dim, generator=generator) * volatility
        style_shock = 0.0020 * momentum * trend_gate * style_loading
        reversal_shock = -0.0025 * reversal_pressure * style_loading
        action_returns = (
            alpha
            + common_loading * common_shock
            + (1.0 - common_loading) * idiosyncratic
            + style_shock
            + reversal_shock
        ).clamp(-0.080, 0.080)
        returns[action_indices, step] = action_returns

        flow_target = torch.tanh(60.0 * alpha + 12.0 * action_returns)
        signed_flow[:, step] = (
            0.730 * flow_previous
            + 0.220 * flow_target
            + 0.180 * torch.randn(action_dim, generator=generator)
        ).clamp(-2.5, 2.5)

        log_volume[action_indices, step] = (
            0.880 * log_volume[action_indices, step - 1]
            + 0.120
            * (
                12.0
                + action_offsets
                + 0.24 * stress
                + 0.18 * signed_flow[:, step].abs()
                + 0.10 * liquidity
            )
            + 0.045 * torch.randn(action_dim, generator=generator)
        )
        context_flow = returns[:, step].abs() * 20.0 + 0.10 * stress
        log_volume[:, step] = torch.where(
            action_entity_mask,
            log_volume[:, step],
            0.900 * log_volume[:, step - 1]
            + 0.100 * (11.8 + context_flow)
            + 0.045 * torch.randn(entity_count, generator=generator),
        )
        regime_labels[step] = scheduled_label

    observed_returns = returns + torch.randn(
        entity_count,
        date_count,
        generator=generator,
        dtype=torch.float32,
    ) * (0.0012 + 0.06 * returns.abs())
    volume = (
        (
            log_volume
            + 0.025 * torch.randn(entity_count, date_count, generator=generator)
        )
        .clamp(4.0, 16.0)
        .exp()
    )
    delta_volume = torch.cat(
        (
            torch.zeros(entity_count, 1, dtype=torch.float32),
            delta_log(volume),
        ),
        dim=-1,
    )
    fast_volatility = (
        ewma_volatility(observed_returns, half_life=8.0, floor=0.0020)
        * (
            1.0
            + 0.034
            * torch.randn(entity_count, date_count, generator=generator).clamp(
                -2.0, 2.0
            )
        )
    ).clamp_min(0.0020)
    slow_volatility = (
        ewma_volatility(observed_returns, half_life=32.0, floor=0.0030)
        * (
            1.0
            + 0.026
            * torch.randn(entity_count, date_count, generator=generator).clamp(
                -2.0, 2.0
            )
        )
    ).clamp_min(0.0030)
    ratio = volatility_ratio(
        fast_volatility,
        slow_volatility,
        floor=0.0030,
        clip_value=5.0,
    )
    scaled = vol_scaled_returns(
        observed_returns,
        slow_volatility,
        floor=0.0030,
        clip_value=5.0,
    )
    values = torch.stack(
        (
            observed_returns,
            (delta_volume + 0.016 * torch.randn_like(delta_volume)).clamp(-5.0, 5.0),
            fast_volatility,
            slow_volatility,
            ratio,
            scaled,
        ),
        dim=-1,
    ).movedim(0, 1)

    forward_returns = torch.zeros(date_count, action_dim, dtype=torch.float32)
    forward_returns[:-1] = returns[action_indices, 1:].movedim(0, 1)
    action_mask = torch.ones(date_count, action_dim, dtype=torch.bool)
    for action_index in range(action_dim):
        action_mask[
            LOOKBACK + 11 + action_index :: 53 + 5 * action_index, action_index
        ] = False
    for burst_start, slots in (
        (88, (0, 2)),
        (122, (1, 4)),
        (238, (0, 1, 3)),
        (292, (2, 4)),
    ):
        burst_end = min(date_count, burst_start + 4)
        action_mask[burst_start:burst_end, torch.tensor(slots, dtype=torch.long)] = (
            False
        )
    for step in range(LOOKBACK, date_count):
        if regime_labels[step] == "stress_rise" and step % 11 in (0, 1):
            action_mask[step, 0] = False
        if regime_labels[step] == "liquidity" and step % 17 == 3:
            action_mask[step, 3] = False

    time_index = torch.arange(date_count, dtype=torch.float32)
    return FactorTimingPanel(
        values=values,
        dates=tuple(range(date_count)),
        entities=entities,
        feature_names=FEATURE_NAMES,
        action_entities=action_entity_names,
        known_asof=time_index,
        decision_times=time_index,
        realized_forward_returns=forward_returns,
        availability_mask=torch.ones_like(values, dtype=torch.bool),
        action_availability_mask=action_mask,
        feature_provenance={
            name: FeatureProvenance(
                source="synthetic_factor_timing_panel",
                transform="synthetic_phase1_feature_family",
                known_lag=0,
                metadata={"diagnostic_only": True},
            )
            for name in FEATURE_NAMES
        },
        metadata={
            "capacity": (0.65, 0.50, 0.60, 0.55, 0.40),
            "date_count": date_count,
            "diagnostic_label": DIAGNOSTIC_LABEL,
            "expected_action_dim": action_dim,
            "factor_basket_assumption": (
                "tradable streams are upstream long/short factor proxy baskets; "
                "the strategy times target exposure and does not construct baskets"
            ),
            "factor_basket_neutrality_assumption": "approx_delta_beta_neutral",
            "masked_action_rate": float(
                (~action_mask).to(dtype=torch.float32).mean().item()
            ),
            "regime_labels": tuple(regime_labels),
            "seed": seed,
            "source": "_build_synthetic_factor_timing_panel",
            "synthetic_assumptions": (
                "balanced conditional latent regimes",
                "gated momentum and crowding reversal phases",
                "noisy observed features with true forward returns",
                "volatility clustering",
                "time-varying cross-factor correlation",
                "compact regional lead-lag context",
                "flow pressure with continuation and unwind states",
            ),
        },
    )


def _split_decision_ranges(date_count: int) -> dict[str, range]:
    train_end = min(date_count - 90, max(LOOKBACK + 120, int(0.65 * date_count)))
    validation_end = min(
        date_count - 35,
        max(train_end + 40, int(0.82 * date_count)),
    )
    if not LOOKBACK < train_end < validation_end < date_count - 1:
        raise ValueError("Synthetic panel split bounds are not valid.")
    return {
        "train": range(LOOKBACK - 1, train_end - 1),
        "validation": range(train_end, validation_end - 1),
        "evaluation": range(validation_end, date_count - 1),
    }


def _normalized_panel(raw_panel: object, *, train_end: int) -> object:
    from indosteg.strats.factor_timing import (
        FactorTimingPanel,
        apply_standardizer,
        fit_robust_standardizer,
    )

    stats = fit_robust_standardizer(
        raw_panel.values[:train_end],
        sample_dims=(0, 1),
        clip_value=5.0,
        winsor_quantiles=(0.01, 0.99),
        missing_policy="raise",
    )
    return FactorTimingPanel(
        values=apply_standardizer(raw_panel.values, stats),
        dates=raw_panel.dates,
        entities=raw_panel.entities,
        feature_names=raw_panel.feature_names,
        action_entities=raw_panel.action_entities,
        known_asof=raw_panel.known_asof,
        decision_times=raw_panel.decision_times,
        realized_forward_returns=raw_panel.realized_forward_returns,
        availability_mask=raw_panel.availability_mask,
        action_availability_mask=raw_panel.action_availability_mask,
        feature_provenance=raw_panel.feature_provenance,
        metadata={
            **dict(raw_panel.metadata),
            "normalization": "train_fit_robust_standardizer",
        },
    )


def _portfolio_constraints() -> object:
    from indosteg.strats.factor_timing import PortfolioConstraints

    return PortfolioConstraints(
        gross_limit=1.0,
        net_limit=0.35,
        max_position=0.35,
        max_turnover=0.50,
        concentration_limit=0.36,
    )


def _reward_config() -> object:
    from indosteg.strats.factor_timing import RewardConfig

    return RewardConfig(
        transaction_cost_rate=0.0008,
        turnover_penalty_rate=0.0005,
        risk_penalty_rate=0.0010,
        concentration_penalty_rate=0.0008,
        volatility_floor=0.010,
        clip_value=5.0,
    )


def _make_system(panel: object) -> object:
    from indosteg.strats.factor_timing import (
        FactorTimingModelConfig,
        FactorTimingSACSystem,
    )

    return FactorTimingSACSystem(
        FactorTimingModelConfig(
            input_dim=panel.feature_count,
            observed_entity_count=panel.entity_count,
            action_dim=panel.action_dim,
            context_dim=CONTEXT_DIM,
            lookback=LOOKBACK,
            patch_length=PATCH_LENGTH,
            patch_stride=PATCH_STRIDE,
            model_dim=MODEL_DIM,
            num_heads=4,
            num_temporal_layers=1,
            num_cross_sectional_layers=1,
            dropout=0.0,
        )
    )


def _context(
    panel: object,
    *,
    decision_index: int,
    previous_positions: torch.Tensor,
    recent_turnover: float,
) -> torch.Tensor:
    values = panel.values[decision_index]
    return torch.tensor(
        [
            float(values[:, FEATURE_INDEX["log_return"]].std(unbiased=False).item()),
            float(values[:, FEATURE_INDEX["volatility_ratio"]].mean().item()),
            float(values[:, FEATURE_INDEX["delta_log_volume"]].mean().item()),
            float(previous_positions.abs().sum().item()),
            float(recent_turnover),
        ],
        dtype=torch.float32,
    )


def _observation(
    panel: object,
    *,
    decision_index: int,
    previous_positions: torch.Tensor,
    recent_turnover: float,
) -> Mapping[str, object]:
    from indosteg.strats.factor_timing import build_factor_timing_observation_from_panel

    return build_factor_timing_observation_from_panel(
        panel,
        end=decision_index + 1,
        lookback=LOOKBACK,
        context=_context(
            panel,
            decision_index=decision_index,
            previous_positions=previous_positions,
            recent_turnover=recent_turnover,
        ),
        previous_positions=previous_positions.clone(),
        patch_length=PATCH_LENGTH,
        patch_stride=PATCH_STRIDE,
    )


def _ex_ante_portfolio_volatility(
    raw_panel: object,
    *,
    decision_index: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    if float(positions.abs().sum().item()) <= 1e-12:
        return torch.zeros((), dtype=torch.float32)
    action_indices = torch.tensor(raw_panel.action_entity_indices(), dtype=torch.long)
    start = max(0, decision_index - VOLATILITY_LOOKBACK + 1)
    returns = raw_panel.values[
        start : decision_index + 1,
        action_indices,
        FEATURE_INDEX["log_return"],
    ]
    current_volatility = raw_panel.values[
        decision_index,
        action_indices,
        FEATURE_INDEX["ewma_volatility_slow"],
    ].clamp_min(0.003)
    if returns.shape[0] < 2:
        covariance = torch.diag(current_volatility.square())
    else:
        centered = returns - returns.mean(dim=0, keepdim=True)
        sample_covariance = centered.T @ centered / returns.shape[0]
        covariance = 0.65 * sample_covariance + 0.35 * torch.diag(
            current_volatility.square()
        )
    variance = positions @ covariance @ positions
    return variance.clamp_min(1e-8).sqrt()


def _trade_step(
    raw_panel: object,
    *,
    decision_index: int,
    action: torch.Tensor,
    previous_positions: torch.Tensor,
) -> tuple[torch.Tensor, object, dict[str, object]]:
    from indosteg.strats.factor_timing import (
        compute_factor_timing_reward,
        project_positions,
    )

    tradable_mask = raw_panel.action_mask(decision_index)
    positions = project_positions(
        action,
        previous_positions=previous_positions,
        constraints=_portfolio_constraints(),
        tradable_mask=tradable_mask,
    )
    reward_terms = compute_factor_timing_reward(
        positions=positions,
        previous_positions=previous_positions,
        realized_returns=raw_panel.action_returns(decision_index),
        ex_ante_portfolio_vol=_ex_ante_portfolio_volatility(
            raw_panel,
            decision_index=decision_index,
            positions=positions,
        ),
        config=_reward_config(),
    )
    turnover = float((positions - previous_positions).abs().sum().item())
    return (
        positions,
        reward_terms,
        {
            "decision_index": decision_index,
            "gross_pnl": reward_terms.gross_pnl,
            "positions": positions.clone(),
            "realized_returns": raw_panel.action_returns(decision_index).clone(),
            "reward_terms": asdict(reward_terms),
            "tradable_mask": tradable_mask.clone(),
            "transaction_costs": reward_terms.transaction_costs,
            "turnover": turnover,
        },
    )


def _policy_action(
    model_system: object,
    observation: Mapping[str, object],
    *,
    deterministic: bool,
) -> torch.Tensor:
    from indosteg.algorithms.sac.actions import (
        deterministic_tanh_action,
        sample_tanh_normal,
    )

    with torch.no_grad():
        distribution = model_system.actor.distribution(observation)
        if deterministic:
            return deterministic_tanh_action(distribution).squeeze(0).cpu()
        return sample_tanh_normal(distribution).action.squeeze(0).cpu()


def _baseline_action(
    name: str,
    raw_panel: object,
    *,
    decision_index: int,
) -> torch.Tensor:
    from indosteg.strats.factor_timing import (
        equal_weight_positions,
        inverse_volatility_positions,
        momentum_positions,
        reversal_positions,
        zero_positions,
    )

    action_dim = raw_panel.action_dim
    if name == "zero":
        return zero_positions(action_dim)
    if name == "equal_weight":
        return equal_weight_positions(action_dim, gross=0.70)

    action_indices = torch.tensor(raw_panel.action_entity_indices(), dtype=torch.long)
    start = max(0, decision_index - 20 + 1)
    trailing_returns = raw_panel.values[
        start : decision_index + 1,
        action_indices,
        FEATURE_INDEX["log_return"],
    ]
    if name == "inverse_vol":
        volatility = raw_panel.values[
            decision_index,
            action_indices,
            FEATURE_INDEX["ewma_volatility_slow"],
        ]
        return inverse_volatility_positions(volatility, gross=0.70)
    if name == "momentum":
        return momentum_positions(trailing_returns, lookback=20, gross=0.70)
    if name == "reversal":
        return reversal_positions(trailing_returns, lookback=20, gross=0.70)
    raise ValueError(f"Unknown baseline: {name}")


def _path_summary(
    raw_panel: object,
    *,
    reward_terms: Sequence[object],
    positions: Sequence[torch.Tensor],
    realized_returns: Sequence[torch.Tensor],
) -> dict[str, object]:
    from indosteg.strats.factor_timing import (
        capacity_usage,
        cost_sensitivity,
        named_factor_contribution,
        performance_summary,
        reward_terms_summary,
    )

    net_pnl = torch.tensor(
        [float(term.net_pnl) for term in reward_terms],
        dtype=torch.float32,
    )
    gross_pnl = torch.tensor(
        [float(term.gross_pnl) for term in reward_terms],
        dtype=torch.float32,
    )
    costs = torch.tensor(
        [float(term.transaction_costs) for term in reward_terms],
        dtype=torch.float32,
    )
    position_tensor = torch.stack(tuple(positions))
    realized_tensor = torch.stack(tuple(realized_returns))
    turnover = torch.cat(
        (
            position_tensor[:1].abs().sum(dim=1),
            (position_tensor[1:] - position_tensor[:-1]).abs().sum(dim=1),
        )
    )
    performance = performance_summary(
        net_pnl,
        turnover=turnover,
        transaction_costs=costs,
        positions=position_tensor,
        hac_lags=5,
    )
    performance["net_pnl"] = float(net_pnl.sum().item())
    performance["mean_reward"] = float(
        torch.tensor([float(term.reward) for term in reward_terms]).mean().item()
    )
    capacity = torch.tensor(raw_panel.metadata["capacity"], dtype=torch.float32)
    return {
        "capacity_usage": capacity_usage(position_tensor, capacity),
        "cost_sensitivity": cost_sensitivity(
            gross_pnl,
            turnover,
            COST_SENSITIVITY_RATES,
            positions=position_tensor,
            hac_lags=5,
        ),
        "cost_sensitivity_net_pnls": {
            float(rate): float((gross_pnl - turnover * rate).sum().item())
            for rate in COST_SENSITIVITY_RATES
        },
        "factor_contribution": named_factor_contribution(
            position_tensor,
            realized_tensor,
            raw_panel.action_entities,
        ),
        "net_pnl_series": net_pnl,
        "performance": performance,
        "reward_terms": reward_terms_summary(reward_terms),
    }


def _evaluate_policy(
    model_system: object,
    raw_panel: object,
    model_panel: object,
    decisions: range,
    *,
    deterministic: bool = True,
) -> dict[str, object]:
    positions = torch.zeros(raw_panel.action_dim, dtype=torch.float32)
    recent_turnover = 0.0
    path_positions: list[torch.Tensor] = []
    path_returns: list[torch.Tensor] = []
    path_terms: list[object] = []

    for decision_index in decisions:
        observation = _observation(
            model_panel,
            decision_index=decision_index,
            previous_positions=positions,
            recent_turnover=recent_turnover,
        )
        action = _policy_action(
            model_system,
            observation,
            deterministic=deterministic,
        )
        positions, terms, info = _trade_step(
            raw_panel,
            decision_index=decision_index,
            action=action,
            previous_positions=positions,
        )
        recent_turnover = float(info["turnover"])
        path_positions.append(positions)
        path_returns.append(info["realized_returns"])
        path_terms.append(terms)

    return _path_summary(
        raw_panel,
        reward_terms=path_terms,
        positions=path_positions,
        realized_returns=path_returns,
    )


def _regime_diagnostics(
    raw_panel: object,
    decisions: range,
    net_pnl: torch.Tensor,
) -> tuple[dict[str, int], dict[str, float]]:
    labels = tuple(raw_panel.metadata["regime_labels"])
    counts: dict[str, int] = {}
    net_pnls: dict[str, float] = {}
    for offset, decision_index in enumerate(decisions):
        label = str(labels[decision_index])
        counts[label] = counts.get(label, 0) + 1
        net_pnls[label] = net_pnls.get(label, 0.0) + float(net_pnl[offset].item())
    return dict(sorted(counts.items())), dict(sorted(net_pnls.items()))


def _evaluate_baseline(
    name: str,
    raw_panel: object,
    decisions: range,
) -> dict[str, object]:
    positions = torch.zeros(raw_panel.action_dim, dtype=torch.float32)
    path_positions: list[torch.Tensor] = []
    path_returns: list[torch.Tensor] = []
    path_terms: list[object] = []

    for decision_index in decisions:
        action = _baseline_action(name, raw_panel, decision_index=decision_index)
        positions, terms, info = _trade_step(
            raw_panel,
            decision_index=decision_index,
            action=action,
            previous_positions=positions,
        )
        path_positions.append(positions)
        path_returns.append(info["realized_returns"])
        path_terms.append(terms)

    return _path_summary(
        raw_panel,
        reward_terms=path_terms,
        positions=path_positions,
        realized_returns=path_returns,
    )


def _train_sac(
    model_system: object,
    raw_panel: object,
    model_panel: object,
    decisions: range,
    *,
    seed: int,
    batch_size: int,
    max_updates: int,
    warmup_steps: int,
) -> tuple[int, int, object]:
    from indosteg.algorithms.sac import (
        SACConfig,
        SACOptimizers,
        hard_update_modules,
        run_sac_update,
    )
    from indosteg.contracts import SACTransition
    from indosteg.replay import UniformReplayBuffer

    hard_update_modules(model_system.target_critic, model_system.critic)
    replay = UniformReplayBuffer(capacity=max(1024, len(decisions) + 64), seed=seed)
    config = SACConfig(
        batch_size=batch_size,
        discount=0.97,
        max_grad_norm=1.0,
        reward_scale=1.0,
        target={"tau": 0.02, "update_interval": 1},
        temperature={"initial_value": 0.04, "learnable": False},
    )
    optimizers = SACOptimizers(
        actor=torch.optim.Adam(model_system.actor.parameters(), lr=5e-4),
        critic=torch.optim.Adam(model_system.critic.parameters(), lr=5e-4),
    )

    positions = torch.zeros(raw_panel.action_dim, dtype=torch.float32)
    recent_turnover = 0.0
    update_steps = 0
    last_metrics = None
    decision_values = tuple(decisions)
    for step, decision_index in enumerate(decision_values):
        observation = _observation(
            model_panel,
            decision_index=decision_index,
            previous_positions=positions,
            recent_turnover=recent_turnover,
        )
        if step < warmup_steps:
            action = torch.empty(raw_panel.action_dim, dtype=torch.float32).uniform_(
                -1.0,
                1.0,
            )
        else:
            action = _policy_action(model_system, observation, deterministic=False)

        next_positions, reward_terms, info = _trade_step(
            raw_panel,
            decision_index=decision_index,
            action=action,
            previous_positions=positions,
        )
        next_observation = _observation(
            model_panel,
            decision_index=decision_index + 1,
            previous_positions=next_positions,
            recent_turnover=float(info["turnover"]),
        )
        replay.add(
            SACTransition(
                observation=observation,
                action=action,
                reward=reward_terms.reward,
                next_observation=next_observation,
                terminated=decision_index == decision_values[-1],
                metadata={
                    "decision_index": decision_index,
                    "projected_positions": info["positions"],
                    "reward_terms": info["reward_terms"],
                    "tradable_mask": info["tradable_mask"],
                    "turnover": info["turnover"],
                },
            )
        )
        positions = next_positions
        recent_turnover = float(info["turnover"])

        if (
            update_steps < max_updates
            and replay.can_sample(batch_size)
            and step >= warmup_steps
        ):
            update_steps += 1
            last_metrics = run_sac_update(
                model_system,
                replay.sample(batch_size),
                optimizers,
                config,
                update_step=update_steps,
            )

    if last_metrics is None:
        raise RuntimeError("Synthetic canonical-panel SAC run did not update.")
    return len(replay), update_steps, last_metrics


def _finite(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str):
        return all(_finite(item) for item in value)
    return True


def _assert_valid_experiment(
    raw_panel: object,
    model_panel: object,
    summary: SyntheticPanelExperimentSummary,
    evaluation: Mapping[str, object],
    baseline_results: Mapping[str, Mapping[str, object]],
) -> None:
    required_metrics = {
        "annualized_return",
        "annualized_volatility",
        "average_concentration",
        "average_effective_bets",
        "average_gross_exposure",
        "average_net_exposure",
        "calmar",
        "hac_t_stat",
        "hit_rate",
        "max_drawdown",
        "net_pnl",
        "sharpe",
        "sortino",
        "t_stat",
        "transaction_cost_drag",
        "turnover",
    }
    action_indices = raw_panel.action_entity_indices()

    if not bool(torch.isfinite(raw_panel.values).all().item()):
        raise AssertionError("Raw panel contains non-finite values.")
    if not bool(torch.isfinite(model_panel.values).all().item()):
        raise AssertionError("Model panel contains non-finite values.")
    if raw_panel.action_dim != int(raw_panel.metadata["expected_action_dim"]):
        raise AssertionError("Synthetic action dimension changed unexpectedly.")
    if raw_panel.entity_count <= raw_panel.action_dim:
        raise AssertionError("Observed entities should include context entities.")
    if action_indices == tuple(range(raw_panel.action_dim)):
        raise AssertionError("Experiment must not assume tradables are first.")
    if raw_panel.feature_count != len(FEATURE_NAMES):
        raise AssertionError("Synthetic feature count changed unexpectedly.")
    if summary.replay_size <= 0 or summary.update_steps <= 0:
        raise AssertionError("Replay and update counts must be positive.")
    if summary.masked_action_rate <= 0.0:
        raise AssertionError("Synthetic panel should exercise frozen action slots.")
    if summary.train_samples_per_parameter <= 0.0:
        raise AssertionError("Sample-to-parameter diagnostic must be positive.")
    if len(summary.regime_counts) < 3:
        raise AssertionError("Evaluation should cover multiple synthetic regimes.")
    if set(baseline_results) != set(BASELINE_NAMES):
        raise AssertionError("Not all compact baselines were evaluated.")
    if set(summary.factor_contribution) != set(raw_panel.action_entities):
        raise AssertionError("Factor contribution keys must match action entities.")
    if set(summary.regime_counts) != set(summary.regime_net_pnls):
        raise AssertionError("Regime counts and net PnL diagnostics must align.")
    if not required_metrics.issubset(evaluation["performance"]):
        raise AssertionError("Evaluation performance summary is missing metrics.")
    if not _finite(asdict(summary)):
        raise AssertionError("Summary contains non-finite values.")
    if not _finite(evaluation):
        raise AssertionError("Evaluation diagnostics contain non-finite values.")
    if not _finite(baseline_results):
        raise AssertionError("Baseline diagnostics contain non-finite values.")


def run_synthetic_canonical_panel_factor_timing_experiment(
    *,
    seed: int = 29,
    date_count: int = DATE_COUNT,
    batch_size: int = 32,
    max_updates: int = 128,
    warmup_steps: int = 32,
) -> SyntheticPanelExperimentSummary:
    """
    Runs the synthetic canonical-panel SAC experiment end to end.
    """

    from indosteg.strats.factor_timing import (
        baseline_comparison,
        parameter_count,
        walk_forward_sample_counts,
    )
    from indosteg.strats.factor_timing.schema import WalkForwardWindow

    random.seed(seed)
    torch.manual_seed(seed)

    raw_panel = _build_synthetic_factor_timing_panel(
        seed=seed,
        date_count=date_count,
    )
    ranges = _split_decision_ranges(raw_panel.date_count)
    train_end = ranges["train"].stop + 1
    model_panel = _normalized_panel(raw_panel, train_end=train_end)
    model_system = _make_system(model_panel)
    model_parameter_count = parameter_count(model_system)

    initial_validation = _evaluate_policy(
        model_system,
        raw_panel,
        model_panel,
        ranges["validation"],
    )
    replay_size, update_steps, last_metrics = _train_sac(
        model_system,
        raw_panel,
        model_panel,
        ranges["train"],
        seed=seed,
        batch_size=batch_size,
        max_updates=max_updates,
        warmup_steps=warmup_steps,
    )
    final_validation = _evaluate_policy(
        model_system,
        raw_panel,
        model_panel,
        ranges["validation"],
    )
    evaluation = _evaluate_policy(
        model_system,
        raw_panel,
        model_panel,
        ranges["evaluation"],
    )
    baseline_results = {
        name: _evaluate_baseline(name, raw_panel, ranges["evaluation"])
        for name in BASELINE_NAMES
    }
    baseline_diagnostics = baseline_comparison(
        evaluation["net_pnl_series"],
        {name: result["net_pnl_series"] for name, result in baseline_results.items()},
        hac_lags=5,
    )
    if not _finite(baseline_diagnostics):
        raise AssertionError(
            "Baseline comparison diagnostics contain non-finite values."
        )
    sample_counts = walk_forward_sample_counts(
        (
            WalkForwardWindow(
                train_start=ranges["train"].start,
                train_end=ranges["train"].stop,
                validation_start=ranges["validation"].start,
                validation_end=ranges["validation"].stop,
                test_start=ranges["evaluation"].start,
                test_end=ranges["evaluation"].stop,
            ),
        )
    )[0]

    cost_sensitivity_sharpes = {
        f"{rate:.4f}": float(metrics["sharpe"])
        for rate, metrics in evaluation["cost_sensitivity"].items()
    }
    cost_sensitivity_net_pnls = {
        f"{rate:.4f}": float(value)
        for rate, value in evaluation["cost_sensitivity_net_pnls"].items()
    }
    regime_counts, regime_net_pnls = _regime_diagnostics(
        raw_panel,
        ranges["evaluation"],
        evaluation["net_pnl_series"],
    )
    summary = SyntheticPanelExperimentSummary(
        seed=seed,
        diagnostic_label=DIAGNOSTIC_LABEL,
        factor_basket_assumption=str(raw_panel.metadata["factor_basket_assumption"]),
        date_count=raw_panel.date_count,
        train_sample_count=sample_counts["train_sample_count"],
        validation_sample_count=sample_counts["validation_sample_count"],
        evaluation_sample_count=sample_counts["test_sample_count"],
        replay_size=replay_size,
        update_steps=update_steps,
        action_dim=raw_panel.action_dim,
        observed_entity_count=raw_panel.entity_count,
        feature_count=raw_panel.feature_count,
        lookback=LOOKBACK,
        parameter_count=model_parameter_count,
        train_samples_per_parameter=(
            sample_counts["train_sample_count"] / model_parameter_count
        ),
        masked_action_rate=float(raw_panel.metadata["masked_action_rate"]),
        initial_validation_net_pnl=float(initial_validation["performance"]["net_pnl"]),
        final_validation_net_pnl=float(final_validation["performance"]["net_pnl"]),
        evaluation_net_pnl=float(evaluation["performance"]["net_pnl"]),
        evaluation_sharpe=float(evaluation["performance"]["sharpe"]),
        evaluation_turnover=float(evaluation["performance"]["turnover"]),
        last_actor_loss=last_metrics.actor_loss,
        last_critic_loss=last_metrics.critic_loss,
        last_temperature=last_metrics.temperature,
        baseline_sharpes={
            name: float(result["performance"]["sharpe"])
            for name, result in baseline_results.items()
        },
        factor_contribution={
            name: float(value)
            for name, value in evaluation["factor_contribution"].items()
        },
        reward_terms={
            name: float(value) for name, value in evaluation["reward_terms"].items()
        },
        cost_sensitivity_sharpes=cost_sensitivity_sharpes,
        cost_sensitivity_net_pnls=cost_sensitivity_net_pnls,
        regime_counts={name: int(value) for name, value in regime_counts.items()},
        regime_net_pnls={name: float(value) for name, value in regime_net_pnls.items()},
        capacity_usage={
            name: float(value) for name, value in evaluation["capacity_usage"].items()
        },
    )
    _assert_valid_experiment(
        raw_panel,
        model_panel,
        summary,
        evaluation,
        baseline_results,
    )
    return summary


def _format_mapping(values: Mapping[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in values.items())


def main() -> None:
    if not RUNTIME_READY:
        raise SystemExit(
            "Synthetic canonical-panel factor timing experiment requires: "
            + ", ".join(MISSING_MANDATORY_RUNTIME_DEPENDENCIES)
        )

    summary = run_synthetic_canonical_panel_factor_timing_experiment()
    scalar_keys = (
        "diagnostic_label",
        "factor_basket_assumption",
        "seed",
        "date_count",
        "observed_entity_count",
        "action_dim",
        "feature_count",
        "lookback",
        "parameter_count",
        "train_samples_per_parameter",
        "masked_action_rate",
        "train_sample_count",
        "validation_sample_count",
        "evaluation_sample_count",
        "replay_size",
        "update_steps",
        "initial_validation_net_pnl",
        "final_validation_net_pnl",
        "evaluation_net_pnl",
        "evaluation_sharpe",
        "evaluation_turnover",
        "last_actor_loss",
        "last_critic_loss",
        "last_temperature",
    )
    summary_dict = asdict(summary)
    for key in scalar_keys:
        value = summary_dict[key]
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"baseline_sharpes: {_format_mapping(summary.baseline_sharpes)}")
    print(f"factor_contribution: {_format_mapping(summary.factor_contribution)}")
    print(
        "reward_terms: "
        + _format_mapping(
            {
                "mean_reward": summary.reward_terms["mean_reward"],
                "sum_net_pnl": summary.reward_terms["sum_net_pnl"],
                "sum_transaction_costs": summary.reward_terms["sum_transaction_costs"],
            }
        )
    )
    print(
        f"cost_sensitivity_sharpes: {_format_mapping(summary.cost_sensitivity_sharpes)}"
    )
    print(
        f"cost_sensitivity_net_pnls: "
        f"{_format_mapping(summary.cost_sensitivity_net_pnls)}"
    )
    print(
        "regime_counts: "
        + ", ".join(f"{key}={value}" for key, value in summary.regime_counts.items())
    )
    print(f"regime_net_pnls: {_format_mapping(summary.regime_net_pnls)}")
    print(f"capacity_usage: {_format_mapping(summary.capacity_usage)}")


if __name__ == "__main__":
    main()
