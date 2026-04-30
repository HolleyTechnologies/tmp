"""
Synthetic SAC exemplar for the factor timing strategy layer.
"""

import importlib.util
import pathlib
import random
import sys

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


CONTEXT_DIM = 5
FEATURE_DIM = 5
LOOKBACK = 6
MODEL_DIM = 16
PATCH_LENGTH = 3
PATCH_STRIDE = 3
FEATURE_NAMES = (
    "realized_return",
    "ex_ante_volatility",
    "flow_signal",
    "liquidity_state",
    "spillover_state",
)


def _synthetic_universe():
    from indosteg.strats.factor_timing import FactorEntity, FactorTimingUniverse

    action_indices = (1, 3, 6, 8, 10, 12)
    entities = tuple(
        FactorEntity(
            f"synthetic_entity_{index}",
            tradable=index in action_indices,
            group="action" if index in action_indices else "context",
        )
        for index in range(13)
    )
    return FactorTimingUniverse(
        observed_entities=entities,
        tradable_entities=tuple(entities[index] for index in action_indices),
    )


@dataclass(frozen=True, slots=True)
class SyntheticFactorTimingSACSummary:
    """
    Compact summary from the synthetic factor timing SAC run.
    """

    seed: int
    total_steps: int
    replay_size: int
    update_steps: int
    action_dim: int
    observed_entity_count: int
    initial_eval_reward: float
    final_eval_reward: float
    initial_eval_net_pnl: float
    final_eval_net_pnl: float
    final_eval_sharpe: float
    final_eval_turnover: float
    last_actor_loss: float
    last_critic_loss: float
    last_temperature: float | None


class _SyntheticFactorTimingEnv:
    """
    Small synthetic daily factor timing path with tradables and spillovers.
    """

    def __init__(
        self,
        *,
        seed: int,
        horizon: int,
        phase: float,
    ) -> None:
        from indosteg.strats.factor_timing import (
            PortfolioConstraints,
            RewardConfig,
        )

        self.seed = seed
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.horizon = horizon
        self.phase = phase
        self.universe = _synthetic_universe()
        self.observed_count = self.universe.observed_entity_count
        self.action_dim = self.universe.action_dim
        self.tradable_indices = torch.tensor(
            self.universe.tradable_observation_indices(),
            dtype=torch.long,
        )
        tradable_mask = torch.zeros(self.observed_count, dtype=torch.bool)
        tradable_mask[self.tradable_indices] = True
        self.context_indices = (~tradable_mask).nonzero().squeeze(-1)
        self.constraints = PortfolioConstraints(
            gross_limit=1.0,
            net_limit=0.35,
            max_position=0.35,
            max_turnover=0.45,
            concentration_limit=0.35,
        )
        self.reward_config = RewardConfig(
            transaction_cost_rate=0.0008,
            turnover_penalty_rate=0.0015,
            risk_penalty_rate=0.002,
            concentration_penalty_rate=0.001,
            volatility_floor=0.06,
        )
        self.reset()

    def _build_paths(self):
        total_steps = self.horizon + LOOKBACK + 1
        time_index = torch.arange(total_steps, dtype=torch.float32) + self.phase
        entity_index = torch.arange(self.observed_count, dtype=torch.float32)
        tradable_index = torch.arange(self.action_dim, dtype=torch.float32)

        alpha = torch.sin(
            0.17 * time_index.unsqueeze(0) + 0.61 * tradable_index.unsqueeze(1)
        )
        alpha = alpha + 0.45 * torch.cos(
            0.07 * time_index.unsqueeze(0) - 0.43 * tradable_index.unsqueeze(1)
        )
        alpha = alpha / alpha.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)

        returns = torch.zeros(self.observed_count, total_steps, dtype=torch.float32)
        flow = torch.zeros_like(returns)
        liquidity = torch.zeros_like(returns)
        volatility = torch.zeros_like(returns)
        spillover = torch.zeros_like(returns)
        risk = 0.45 + 0.15 * torch.sin(0.05 * time_index + 0.2)

        for step in range(1, total_steps):
            deterministic_noise = 0.004 * torch.sin(
                0.37 * entity_index + 0.19 * time_index[step]
            )
            tradable_return = 0.045 * alpha[
                :, step - 1
            ] + deterministic_noise.index_select(0, self.tradable_indices)
            returns[self.tradable_indices, step] = tradable_return
            returns[self.context_indices, step] = (
                0.55 * tradable_return.mean()
                + 0.018
                * torch.sin(
                    0.11 * time_index[step]
                    + 0.47 * entity_index.index_select(0, self.context_indices)
                )
            )
            peer_mean = returns[self.tradable_indices, max(0, step - 3) : step].mean()
            spillover[:, step] = 0.60 * returns[:, step - 1] + 0.40 * peer_mean
            flow[self.tradable_indices, step] = alpha[:, step - 1] + 0.25 * peer_mean
            flow[self.context_indices, step] = spillover[self.context_indices, step]
            volatility[:, step] = (
                0.08 + 0.03 * risk[step] + 0.80 * returns[:, step].abs()
            )
            liquidity[:, step] = (
                1.0
                + 0.08 * torch.cos(0.03 * time_index[step] + 0.21 * entity_index)
                - 0.12 * flow[:, step].abs()
            ).clamp_min(0.35)

        return returns, flow, liquidity, volatility.clamp_min(0.04), spillover, risk

    def _build_panel(self):
        from indosteg.strats.factor_timing import (
            FactorTimingPanel,
            FeatureProvenance,
        )

        total_steps = self.returns.shape[1]
        values = torch.stack(
            (
                self.returns,
                self.volatility,
                self.flow,
                self.liquidity,
                self.spillover,
            ),
            dim=-1,
        ).movedim(0, 1)
        forward_returns = torch.zeros(
            total_steps,
            self.action_dim,
            dtype=torch.float32,
        )
        forward_returns[:-1] = self.returns[
            self.tradable_indices,
            1:,
        ].movedim(0, 1)
        action_mask = torch.ones(
            total_steps,
            self.action_dim,
            dtype=torch.bool,
        )
        for action_index in range(self.action_dim):
            action_mask[LOOKBACK - 1 + action_index :: 17 + action_index] = False

        time_index = torch.arange(total_steps, dtype=torch.float32)
        return FactorTimingPanel(
            values=values,
            dates=tuple(range(total_steps)),
            entities=self.universe.observed_entities,
            feature_names=FEATURE_NAMES,
            action_entities=self.universe.tradable_names,
            known_asof=time_index,
            decision_times=time_index,
            realized_forward_returns=forward_returns,
            availability_mask=torch.ones_like(values, dtype=torch.bool),
            action_availability_mask=action_mask,
            feature_provenance={
                name: FeatureProvenance(
                    source="synthetic",
                    transform="factor_timing_sac_fixture",
                )
                for name in FEATURE_NAMES
            },
            metadata={
                "phase": self.phase,
                "seed": self.seed,
                "source": "synthetic_factor_timing_sac",
            },
        )

    def _observation(self):
        from indosteg.strats.factor_timing import (
            build_factor_timing_observation_from_panel,
        )

        stop = self.step_index + LOOKBACK
        decision_index = stop - 1
        context = torch.tensor(
            [
                float(self.risk[decision_index].item()),
                float(
                    self.spillover[
                        self.tradable_indices,
                        decision_index,
                    ]
                    .mean()
                    .item()
                ),
                float(self.liquidity[:, decision_index].mean().item()),
                float(self.positions.abs().sum().item()),
                float(self.recent_turnover),
            ],
            dtype=torch.float32,
        )
        return build_factor_timing_observation_from_panel(
            self.panel,
            end=stop,
            lookback=LOOKBACK,
            context=context,
            previous_positions=self.positions.clone(),
            patch_length=PATCH_LENGTH,
            patch_stride=PATCH_STRIDE,
        )

    def reset(self):
        (
            self.returns,
            self.flow,
            self.liquidity,
            self.volatility,
            self.spillover,
            self.risk,
        ) = self._build_paths()
        self.panel = self._build_panel()
        self.step_index = 0
        self.recent_turnover = 0.0
        self.positions = torch.zeros(self.action_dim, dtype=torch.float32)
        return self._observation()

    def step(self, action: torch.Tensor):
        from indosteg.strats.factor_timing import (
            compute_factor_timing_reward,
            project_positions,
        )

        decision_index = self.step_index + LOOKBACK - 1
        tradable_mask = self.panel.action_mask(decision_index)
        target_positions = project_positions(
            action,
            previous_positions=self.positions,
            constraints=self.constraints,
            tradable_mask=tradable_mask,
        )
        realized_returns = self.panel.action_returns(decision_index)
        current_volatility = self.panel.values[
            decision_index,
            self.tradable_indices,
            FEATURE_NAMES.index("ex_ante_volatility"),
        ]
        ex_ante_vol = torch.sqrt(
            (target_positions.square() * current_volatility.square()).sum()
        )
        reward_terms = compute_factor_timing_reward(
            positions=target_positions,
            previous_positions=self.positions,
            realized_returns=realized_returns,
            ex_ante_portfolio_vol=ex_ante_vol,
            config=self.reward_config,
        )
        self.recent_turnover = float(
            (target_positions - self.positions).abs().sum().item()
        )
        self.positions = target_positions
        self.step_index += 1
        terminated = self.step_index >= self.horizon
        return (
            self._observation(),
            reward_terms.reward,
            terminated,
            {
                "net_pnl": reward_terms.net_pnl,
                "reward": reward_terms.reward,
                "reward_terms": asdict(reward_terms),
                "transaction_costs": reward_terms.transaction_costs,
                "tradable_mask": tradable_mask.clone(),
                "turnover": self.recent_turnover,
                "decision_index": decision_index,
                "positions": target_positions.clone(),
            },
        )


def _make_system(universe):
    from indosteg.strats.factor_timing import (
        FactorTimingModelConfig,
        FactorTimingSACSystem,
    )

    return FactorTimingSACSystem(
        FactorTimingModelConfig(
            input_dim=FEATURE_DIM,
            observed_entity_count=universe.observed_entity_count,
            action_dim=universe.action_dim,
            context_dim=CONTEXT_DIM,
            lookback=LOOKBACK,
            patch_length=PATCH_LENGTH,
            patch_stride=PATCH_STRIDE,
            model_dim=MODEL_DIM,
            num_heads=4,
            num_temporal_layers=1,
            num_cross_sectional_layers=1,
        )
    )


def _policy_action(model_system, observation, *, deterministic: bool) -> torch.Tensor:
    from indosteg.algorithms.sac.actions import (
        deterministic_tanh_action,
        sample_tanh_normal,
    )

    with torch.no_grad():
        distribution = model_system.actor.distribution(observation)
        if deterministic:
            return deterministic_tanh_action(distribution).squeeze(0).cpu()
        return sample_tanh_normal(distribution).action.squeeze(0).cpu()


def _evaluate_policy(
    model_system,
    *,
    seed: int,
    episodes: int,
    horizon: int,
    phase: float,
) -> dict[str, float]:
    from indosteg.strats.factor_timing import performance_summary

    rewards: list[float] = []
    net_pnls: list[float] = []
    costs: list[float] = []
    turnovers: list[float] = []
    positions = []
    for episode in range(episodes):
        environment = _SyntheticFactorTimingEnv(
            seed=seed + episode,
            horizon=horizon,
            phase=phase + 11.0 * episode,
        )
        observation = environment.reset()
        done = False
        while not done:
            action = _policy_action(model_system, observation, deterministic=True)
            observation, reward, done, info = environment.step(action)
            rewards.append(float(reward))
            net_pnls.append(float(info["net_pnl"]))
            costs.append(float(info["transaction_costs"]))
            turnovers.append(float(info["turnover"]))
            positions.append(info["positions"])

    pnl_tensor = torch.tensor(net_pnls, dtype=torch.float32)
    metrics = performance_summary(
        pnl_tensor,
        turnover=torch.tensor(turnovers, dtype=torch.float32),
        transaction_costs=torch.tensor(costs, dtype=torch.float32),
        positions=torch.stack(positions),
    )
    metrics["mean_reward"] = float(sum(rewards) / len(rewards))
    metrics["net_pnl"] = float(sum(net_pnls))
    return metrics


def run_synthetic_factor_timing_sac_exemplar(
    *,
    seed: int = 17,
    total_steps: int = 512,
    warmup_steps: int = 64,
    horizon: int = 64,
    batch_size: int = 32,
    replay_capacity: int = 4096,
    evaluation_episodes: int = 3,
) -> SyntheticFactorTimingSACSummary:
    """
    Runs a compact replay-based SAC loop through the factor timing stack.
    """

    from indosteg.algorithms.sac import (
        SACConfig,
        SACOptimizers,
        hard_update_modules,
        run_sac_update,
    )
    from indosteg.contracts import SACTransition
    from indosteg.replay import UniformReplayBuffer

    random.seed(seed)
    torch.manual_seed(seed)

    environment = _SyntheticFactorTimingEnv(seed=seed, horizon=horizon, phase=0.0)
    model_system = _make_system(environment.universe)
    hard_update_modules(model_system.target_critic, model_system.critic)

    replay = UniformReplayBuffer(capacity=replay_capacity, seed=seed)
    config = SACConfig(
        batch_size=batch_size,
        discount=0.97,
        max_grad_norm=1.0,
        target={
            "tau": 0.02,
            "update_interval": 1,
        },
        temperature={
            "initial_value": 0.03,
            "learnable": False,
        },
    )
    optimizers = SACOptimizers(
        actor=torch.optim.Adam(model_system.actor.parameters(), lr=8e-4),
        critic=torch.optim.Adam(model_system.critic.parameters(), lr=8e-4),
    )

    evaluation_seed = seed + 1000
    initial_metrics = _evaluate_policy(
        model_system,
        seed=evaluation_seed,
        episodes=evaluation_episodes,
        horizon=horizon,
        phase=200.0,
    )

    observation = environment.reset()
    last_metrics = None
    update_steps = 0
    for step in range(total_steps):
        if step < warmup_steps:
            action = torch.empty(environment.action_dim, dtype=torch.float32).uniform_(
                -1.0,
                1.0,
            )
        else:
            action = _policy_action(model_system, observation, deterministic=False)

        next_observation, reward, terminated, info = environment.step(action)
        replay.add(
            SACTransition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                metadata={
                    "decision_index": info["decision_index"],
                    "projected_positions": info["positions"],
                    "reward_terms": info["reward_terms"],
                    "tradable_mask": info["tradable_mask"],
                },
            )
        )
        observation = next_observation

        if replay.can_sample(batch_size):
            update_steps += 1
            last_metrics = run_sac_update(
                model_system,
                replay.sample(batch_size),
                optimizers,
                config,
                update_step=update_steps,
            )

        if terminated:
            observation = environment.reset()

    final_metrics = _evaluate_policy(
        model_system,
        seed=evaluation_seed,
        episodes=evaluation_episodes,
        horizon=horizon,
        phase=200.0,
    )
    if last_metrics is None:
        raise RuntimeError("Synthetic factor timing SAC run did not update.")

    return SyntheticFactorTimingSACSummary(
        seed=seed,
        total_steps=total_steps,
        replay_size=len(replay),
        update_steps=update_steps,
        action_dim=environment.action_dim,
        observed_entity_count=environment.observed_count,
        initial_eval_reward=float(initial_metrics["mean_reward"]),
        final_eval_reward=float(final_metrics["mean_reward"]),
        initial_eval_net_pnl=float(initial_metrics["net_pnl"]),
        final_eval_net_pnl=float(final_metrics["net_pnl"]),
        final_eval_sharpe=float(final_metrics["sharpe"]),
        final_eval_turnover=float(final_metrics["turnover"]),
        last_actor_loss=last_metrics.actor_loss,
        last_critic_loss=last_metrics.critic_loss,
        last_temperature=last_metrics.temperature,
    )


def main() -> None:
    if not RUNTIME_READY:
        raise SystemExit(
            "Synthetic factor timing SAC exemplar requires: "
            + ", ".join(MISSING_MANDATORY_RUNTIME_DEPENDENCIES)
        )

    summary = run_synthetic_factor_timing_sac_exemplar()
    for key, value in asdict(summary).items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
