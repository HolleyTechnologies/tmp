"""
Regression tests for the factor timing strategy layer.
"""

import importlib.util
import math
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
INDORAPTOR_SRC = ROOT.parent / "indoraptor" / "src"

for path in (SRC_ROOT, INDORAPTOR_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TORCH_READY = importlib.util.find_spec("torch") is not None
INDORAPTOR_READY = importlib.util.find_spec("indoraptor") is not None

if TORCH_READY:
    import torch


@unittest.skipUnless(
    TORCH_READY and INDORAPTOR_READY,
    "Factor timing tests require torch and indoraptor.",
)
class FactorTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)

    def _metadata_driven_universe(self):
        from indosteg.strats.factor_timing import FactorEntity, FactorTimingUniverse

        observed = (
            FactorEntity(
                "market_context",
                group="context",
                metadata={"role": "context", "family": "market"},
            ),
            FactorEntity(
                "momentum",
                tradable=True,
                group="factor",
                metadata={"role": "action", "family": "momentum"},
            ),
            FactorEntity(
                "flow_context",
                group="context",
                metadata={"role": "context", "family": "flow"},
            ),
            FactorEntity(
                "beta",
                tradable=True,
                group="factor",
                metadata={"role": "action", "family": "beta"},
            ),
            FactorEntity(
                "size",
                tradable=True,
                group="factor",
                metadata={"role": "action", "family": "size"},
            ),
            FactorEntity(
                "risk_context",
                group="context",
                metadata={"role": "context", "family": "risk"},
            ),
            FactorEntity(
                "residual_volatility",
                tradable=True,
                group="factor",
                metadata={"role": "action", "family": "residual_volatility"},
            ),
            FactorEntity(
                "liquidity_context",
                group="context",
                metadata={"role": "context", "family": "liquidity"},
            ),
            FactorEntity(
                "transient",
                tradable=True,
                group="factor",
                metadata={"role": "action", "family": "transient"},
            ),
        )
        return FactorTimingUniverse(
            observed_entities=observed,
            tradable_entities=(
                observed[3],
                observed[6],
                observed[1],
                observed[4],
                observed[8],
            ),
        )

    def _interleaved_action_universe(self):
        from indosteg.strats.factor_timing import FactorEntity, FactorTimingUniverse

        observed = (
            FactorEntity("macro_context", group="context"),
            FactorEntity(
                "beta",
                tradable=True,
                group="factor",
                metadata={"role": "action", "family": "beta"},
            ),
            FactorEntity("flow_context", group="context"),
            FactorEntity(
                "transient",
                tradable=True,
                group="factor",
                metadata={"role": "action", "family": "transient"},
            ),
            FactorEntity("liquidity_context", group="context"),
        )
        return FactorTimingUniverse(
            observed_entities=observed,
            tradable_entities=(observed[1], observed[3]),
        )

    def _observation(self):
        from indosteg.strats.factor_timing import build_factor_timing_observation

        universe = self._metadata_driven_universe()
        panel = torch.randn(universe.observed_entity_count, 6, 4)
        context = torch.randn(3)
        previous_positions = torch.zeros(universe.action_dim)
        return build_factor_timing_observation(
            panel=panel,
            context=context,
            previous_positions=previous_positions,
            universe=universe,
            patch_length=3,
            patch_stride=3,
        )

    def test_feature_and_action_universes_can_differ(self) -> None:
        from indosteg.strats.factor_timing import (
            build_factor_timing_observation,
            validate_patch_aligned_lookback,
        )

        universe = self._metadata_driven_universe()
        self.assertGreater(universe.observed_entity_count, universe.action_dim)
        self.assertEqual(
            len(universe.tradable_observation_indices()), universe.action_dim
        )

        observation = build_factor_timing_observation(
            panel=torch.randn(universe.observed_entity_count, 6, 5),
            context=torch.randn(4),
            previous_positions=torch.zeros(universe.action_dim),
            universe=universe,
            patch_length=3,
            patch_stride=3,
        )

        self.assertEqual(observation["panel"].shape[0], universe.observed_entity_count)
        self.assertEqual(
            observation["previous_positions"].shape[0],
            universe.action_dim,
        )
        self.assertGreater(
            observation["panel"].shape[0],
            observation["previous_positions"].shape[0],
        )
        with self.assertRaises(ValueError):
            validate_patch_aligned_lookback(7, patch_length=4, patch_stride=2)

    def test_custom_universe_supports_generic_context_and_action_subset(self) -> None:
        from indosteg.strats.factor_timing import build_factor_timing_observation

        universe = self._interleaved_action_universe()
        observation = build_factor_timing_observation(
            panel=torch.randn(universe.observed_entity_count, 6, 3),
            context=torch.randn(2),
            previous_positions=torch.zeros(universe.action_dim),
            universe=universe,
            patch_length=3,
            patch_stride=3,
        )

        self.assertEqual(universe.observed_entities[0].group, "context")
        self.assertIsNone(universe.observed_entities[0].region)
        self.assertEqual(universe.tradable_names, ("beta", "transient"))
        self.assertEqual(universe.tradable_observation_indices(), (1, 3))
        self.assertTrue(
            torch.equal(
                observation["tradable_entity_indices"],
                torch.tensor([1, 3], dtype=torch.long),
            )
        )
        self.assertEqual(observation["previous_positions"].shape, (2,))

    def test_residual_entity_is_tradable_only_when_supplied_as_action(self) -> None:
        from indosteg.strats.factor_timing import (
            FactorEntity,
            FactorTimingPanel,
            FactorTimingUniverse,
            build_factor_timing_observation,
        )

        observed = (
            FactorEntity("macro_context", group="macro"),
            FactorEntity(
                "transient_residual_proxy",
                group="residual",
                metadata={"signal_class": "transient"},
            ),
            FactorEntity("regional_peer", "Custom Region", "value", group="style"),
        )
        universe = FactorTimingUniverse(
            observed_entities=observed,
            tradable_entities=(observed[1],),
        )
        observation = build_factor_timing_observation(
            panel=torch.randn(3, 6, 2),
            previous_positions=torch.zeros(1),
            universe=universe,
            patch_length=3,
            patch_stride=3,
        )
        panel = FactorTimingPanel(
            values=torch.randn(4, 3, 2),
            dates=tuple(range(4)),
            entities=observed,
            feature_names=("return", "risk"),
            action_entities=("transient_residual_proxy",),
        )

        self.assertFalse(observed[1].tradable)
        self.assertEqual(universe.tradable_names, ("transient_residual_proxy",))
        self.assertEqual(universe.tradable_observation_indices(), (1,))
        self.assertTrue(
            torch.equal(
                observation["tradable_entity_indices"],
                torch.tensor([1], dtype=torch.long),
            )
        )
        self.assertEqual(panel.action_entity_indices(), (1,))
        self.assertEqual(panel.universe.action_dim, 1)

    def test_factor_timing_panel_validates_boundary_and_builds_observations(
        self,
    ) -> None:
        from indosteg.strats.factor_timing import (
            FactorEntity,
            FactorTimingPanel,
            FeatureProvenance,
            build_factor_timing_observation_from_panel,
        )

        entities = (
            FactorEntity("macro_context", group="macro"),
            FactorEntity("tradable_b", tradable=True),
            FactorEntity("flow_context", group="flow"),
            FactorEntity("tradable_a", tradable=True),
        )
        values = torch.arange(5 * 4 * 3, dtype=torch.float32).reshape(5, 4, 3)
        action_mask = torch.ones(5, 2, dtype=torch.bool)
        action_mask[3, 0] = False
        panel = FactorTimingPanel(
            values=values,
            dates=tuple(range(5)),
            entities=entities,
            feature_names=("raw_a", "derived_b", "context_c"),
            action_entities=("tradable_a", "tradable_b"),
            known_asof=torch.arange(5, dtype=torch.float32),
            decision_times=torch.arange(5, dtype=torch.float32),
            realized_forward_returns=torch.randn(5, 2),
            availability_mask=torch.ones_like(values, dtype=torch.bool),
            action_availability_mask=action_mask,
            feature_provenance={
                "derived_b": FeatureProvenance(
                    source="fixture",
                    transform="unit_test",
                    inputs=("raw_a",),
                    lookback=2,
                )
            },
            metadata={"fixture": "panel"},
        )

        observation = build_factor_timing_observation_from_panel(
            panel,
            end=4,
            lookback=3,
            previous_positions=torch.zeros(2),
            patch_length=3,
            patch_stride=3,
        )

        self.assertEqual(panel.action_entity_indices(), (3, 1))
        self.assertEqual(tuple(observation["panel"].shape), (4, 3, 3))
        self.assertEqual(tuple(observation["context"].shape), (0,))
        self.assertTrue(
            torch.equal(
                observation["tradable_entity_indices"],
                torch.tensor([3, 1], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                observation["tradable_mask"],
                torch.tensor([False, True], dtype=torch.bool),
            )
        )
        self.assertEqual(tuple(observation["availability_mask"].shape), (4, 3, 3))
        self.assertEqual(observation["metadata"]["feature_names"], panel.feature_names)
        self.assertEqual(tuple(panel.action_returns(0).shape), (2,))

    def test_factor_timing_panel_rejects_unavailable_known_asof(self) -> None:
        from indosteg.strats.factor_timing import FactorEntity, FactorTimingPanel

        entities = (
            FactorEntity("a", tradable=True),
            FactorEntity("b"),
        )
        with self.assertRaises(ValueError):
            FactorTimingPanel(
                values=torch.zeros(2, 2, 1),
                dates=(0, 1),
                entities=entities,
                feature_names=("x",),
                action_entities=("a",),
                known_asof=torch.tensor([0.0, 2.0]),
                decision_times=torch.tensor([0.0, 1.0]),
            )

    def test_contract_validation_guards_reject_invalid_inputs(self) -> None:
        from indosteg.strats.factor_timing import (
            FactorEntity,
            FactorTimingModelConfig,
            FactorTimingPanel,
            FactorTimingUniverse,
            PortfolioConstraints,
            ResearchTrialMetadata,
            RewardConfig,
        )

        entity = FactorEntity("a")
        with self.assertRaises(ValueError):
            FactorTimingUniverse(observed_entities=(entity,), tradable_entities=())
        with self.assertRaises(ValueError):
            FactorTimingPanel(
                values=torch.zeros(2, 1, 1),
                dates=(0, 1),
                entities=(entity,),
                feature_names=("x",),
                action_entities=("a",),
                action_availability_mask=torch.ones(2, 2, dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            FactorTimingModelConfig(
                input_dim=1,
                observed_entity_count=1,
                action_dim=1,
                context_dim=0,
                lookback=4,
                patch_length=2,
                dropout=1.0,
            )
        with self.assertRaises(ValueError):
            PortfolioConstraints(gross_limit=-1.0)
        with self.assertRaises(ValueError):
            RewardConfig(volatility_floor=0.0)
        with self.assertRaises(ValueError):
            ResearchTrialMetadata(trial_id="", trial_count=1)

    def test_normalization_uses_train_statistics_only(self) -> None:
        from indosteg.strats.factor_timing.normalization import (
            apply_standardizer,
            fit_standardizer,
        )

        train = torch.tensor([[1.0, 3.0], [3.0, 7.0]])
        validation = torch.tensor([[101.0, 107.0]])
        stats = fit_standardizer(train, sample_dims=(0,))
        normalized_validation = apply_standardizer(validation, stats)

        self.assertTrue(torch.allclose(stats.mean, torch.tensor([[2.0, 5.0]])))
        self.assertTrue(torch.allclose(stats.scale, torch.tensor([[1.0, 2.0]])))
        self.assertTrue(
            torch.allclose(normalized_validation, torch.tensor([[99.0, 51.0]]))
        )

    def test_robust_normalization_uses_train_iqr_and_missing_policy(self) -> None:
        from indosteg.strats.factor_timing import (
            apply_standardizer,
            fit_apply_train_standardizer,
            fit_robust_standardizer,
            fit_standardizer,
        )

        train = torch.tensor([[0.0], [2.0], [100.0]])
        validation = torch.tensor([[10000.0]])
        standard_stats = fit_standardizer(train, sample_dims=(0,))
        robust_stats = fit_robust_standardizer(
            train,
            sample_dims=(0,),
            clip_value=2.0,
        )
        normalized_validation = apply_standardizer(validation, robust_stats)

        self.assertEqual(robust_stats.method, "robust")
        self.assertAlmostEqual(float(robust_stats.mean.item()), 2.0, places=6)
        self.assertNotAlmostEqual(
            float(standard_stats.mean.item()),
            float(robust_stats.mean.item()),
        )
        self.assertAlmostEqual(float(normalized_validation.item()), 2.0, places=6)

        stats, normalized_train, normalized_validation, normalized_test = (
            fit_apply_train_standardizer(
                torch.tensor([[1.0, float("nan")], [3.0, 7.0], [5.0, 9.0]]),
                torch.tensor([[float("nan"), 11.0]]),
                torch.tensor([[7.0, 13.0]]),
                sample_dims=(0,),
                method="robust",
                missing_policy="omit",
            )
        )

        self.assertEqual(stats.missing_policy, "omit")
        self.assertTrue(torch.isfinite(normalized_train[:, 0]).all())
        self.assertAlmostEqual(float(normalized_validation[0, 0].item()), 0.0)
        self.assertTrue(torch.isfinite(normalized_test).all())

    def test_winsorization_bounds_are_fit_on_train_only(self) -> None:
        from indosteg.strats.factor_timing import (
            apply_standardizer,
            fit_standardizer,
        )

        train = torch.tensor([[0.0], [1.0], [100.0]])
        stats = fit_standardizer(
            train,
            sample_dims=(0,),
            winsor_quantiles=(0.0, 0.5),
        )
        transformed = apply_standardizer(torch.tensor([[10000.0]]), stats)

        self.assertIsNotNone(stats.winsor_upper)
        self.assertAlmostEqual(float(stats.winsor_upper.item()), 1.0, places=6)
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertLess(float(transformed.item()), 1.0)

    def test_optional_feature_helpers_are_name_agnostic(self) -> None:
        from indosteg.strats.factor_timing import (
            cross_sectional_dispersion,
            delta_log,
            downside_volatility,
            drawdown_state,
            ewma_volatility,
            ewma_volatility_ratio,
            group_spillover_summary,
            robust_zscore,
            rolling_robust_zscore,
            rolling_zscore,
            volatility_ratio,
        )

        values = torch.exp(torch.tensor([[1.0, 2.0, 4.0]]))
        returns = torch.tensor(
            [
                [0.01, 0.02, 0.03],
                [0.02, -0.01, 0.01],
                [-0.01, 0.00, 0.02],
            ]
        )

        self.assertTrue(torch.allclose(delta_log(values), torch.tensor([[1.0, 2.0]])))
        self.assertEqual(tuple(ewma_volatility(returns, half_life=20.0).shape), (3, 3))
        self.assertEqual(tuple(ewma_volatility_ratio(returns).shape), (3, 3))
        self.assertEqual(tuple(downside_volatility(returns, window=2).shape), (3, 2))
        self.assertEqual(tuple(drawdown_state(returns).shape), (3, 3))
        self.assertEqual(tuple(rolling_zscore(returns, window=2).shape), (3, 2))
        self.assertEqual(
            tuple(rolling_robust_zscore(returns, window=2).shape),
            (3, 2),
        )
        self.assertEqual(tuple(cross_sectional_dispersion(returns).shape), (3,))
        self.assertTrue(
            torch.isfinite(
                volatility_ratio(
                    torch.tensor([0.20, 0.40]),
                    torch.tensor([0.10, 0.20]),
                )
            ).all()
        )
        self.assertTrue(torch.isfinite(robust_zscore(returns, dim=1)).all())
        self.assertEqual(
            tuple(
                group_spillover_summary(
                    returns,
                    torch.tensor([2, 1, 2]),
                    window=2,
                ).shape
            ),
            (2, 2),
        )
        with self.assertRaises(ValueError):
            delta_log(torch.tensor([[-1.0, 2.0]]))

    def test_portfolio_projection_respects_constraints_and_confidence(self) -> None:
        from indosteg.strats.factor_timing import (
            ForecastMoments,
            PortfolioConstraints,
            PositionSignal,
            project_positions,
        )

        signal = PositionSignal(
            action=torch.tensor([1.0, 1.0, -1.0, 0.5]),
            moments=ForecastMoments(conviction=torch.tensor([1.0, 0.5, 1.0, 0.25])),
        )
        projected = project_positions(
            signal,
            previous_positions=torch.zeros(4),
            constraints=PortfolioConstraints(
                gross_limit=0.8,
                net_limit=0.2,
                max_position=0.4,
                max_turnover=0.7,
                concentration_limit=0.5,
            ),
            tradable_mask=torch.tensor([True, True, False, True]),
        )

        self.assertAlmostEqual(float(projected[2].item()), 0.0)
        self.assertLessEqual(float(projected.abs().sum().item()), 0.8 + 1e-6)
        self.assertLessEqual(float(projected.sum().abs().item()), 0.2 + 1e-6)
        self.assertLessEqual(float(projected.abs().max().item()), 0.4 + 1e-6)
        self.assertLessEqual(float(projected.abs().sum().item()), 0.7 + 1e-6)

    def test_portfolio_projection_freezes_masked_slots_without_turnover(self) -> None:
        from indosteg.strats.factor_timing import (
            PortfolioConstraints,
            RewardConfig,
            compute_factor_timing_reward,
            project_positions,
        )

        previous = torch.tensor([0.35, -0.20, 0.10])
        projected = project_positions(
            torch.tensor([-1.0, 0.80, -0.90]),
            previous_positions=previous,
            constraints=PortfolioConstraints(
                gross_limit=1.0,
                net_limit=1.0,
                max_position=0.8,
                max_turnover=10.0,
                concentration_limit=1.0,
            ),
            tradable_mask=torch.tensor([False, True, False]),
        )
        reward_terms = compute_factor_timing_reward(
            positions=projected,
            previous_positions=previous,
            realized_returns=torch.tensor([0.01, 0.02, -0.03]),
            ex_ante_portfolio_vol=0.5,
            config=RewardConfig(
                transaction_cost_rate=0.01,
                turnover_penalty_rate=0.02,
                clip_value=100.0,
            ),
        )

        self.assertAlmostEqual(float(projected[0].item()), 0.35, places=6)
        self.assertAlmostEqual(float(projected[2].item()), 0.10, places=6)
        self.assertNotAlmostEqual(float(projected[1].item()), float(previous[1].item()))
        expected_turnover = float((projected[1] - previous[1]).abs().item())
        self.assertAlmostEqual(
            reward_terms.transaction_costs,
            0.01 * expected_turnover,
            places=6,
        )

    def test_sac_distribution_signal_uses_uncertainty_as_confidence(self) -> None:
        from indosteg.contracts import SACActionDistribution
        from indosteg.strats.factor_timing import (
            PortfolioConstraints,
            PositionSignal,
            position_signal_from_sac_distribution,
            project_positions,
        )

        distribution = SACActionDistribution(
            mean=torch.zeros(2, 2),
            log_std=torch.tensor([[-3.0, -3.0], [1.0, 1.0]]),
        )
        action = torch.full((2, 2), 0.8)

        low_uncertainty = position_signal_from_sac_distribution(
            distribution,
            action=action,
            batch_index=0,
        )
        high_uncertainty = position_signal_from_sac_distribution(
            distribution,
            action=action,
            batch_index=1,
        )

        self.assertIsInstance(low_uncertainty, PositionSignal)
        self.assertTrue(
            torch.allclose(
                low_uncertainty.moments.action_mean,
                torch.zeros(2),
            )
        )
        self.assertLess(
            float(high_uncertainty.moments.conviction.mean().item()),
            float(low_uncertainty.moments.conviction.mean().item()),
        )
        self.assertLess(
            float(
                project_positions(
                    high_uncertainty,
                    constraints=PortfolioConstraints(gross_limit=10.0),
                )
                .abs()
                .sum()
                .item()
            ),
            float(
                project_positions(
                    low_uncertainty,
                    constraints=PortfolioConstraints(gross_limit=10.0),
                )
                .abs()
                .sum()
                .item()
            ),
        )
        with self.assertRaises(ValueError):
            position_signal_from_sac_distribution(distribution)

    def test_reward_records_pnl_and_penalty_terms(self) -> None:
        from indosteg.strats.factor_timing import (
            RewardConfig,
            compute_factor_timing_reward,
        )

        terms = compute_factor_timing_reward(
            positions=torch.tensor([0.5, -0.25]),
            previous_positions=torch.tensor([0.0, 0.25]),
            realized_returns=torch.tensor([0.02, -0.01]),
            ex_ante_portfolio_vol=0.5,
            config=RewardConfig(
                transaction_cost_rate=0.01,
                turnover_penalty_rate=0.02,
                risk_penalty_rate=0.03,
                concentration_penalty_rate=0.04,
            ),
        )

        self.assertAlmostEqual(terms.gross_pnl, 0.0125, places=6)
        self.assertGreater(terms.transaction_costs, 0.0)
        self.assertGreater(terms.turnover_penalty, 0.0)
        self.assertGreater(terms.risk_penalty, 0.0)
        self.assertGreater(terms.concentration_penalty, 0.0)
        self.assertAlmostEqual(terms.reward, terms.net_pnl / 0.5, places=6)

    def test_reward_risk_penalty_is_distinct_from_concentration(self) -> None:
        from indosteg.strats.factor_timing import (
            RewardConfig,
            compute_factor_timing_reward,
        )

        config = RewardConfig(
            risk_penalty_rate=0.10,
            concentration_penalty_rate=0.20,
            clip_value=100.0,
        )
        low_risk = compute_factor_timing_reward(
            positions=torch.tensor([0.50, -0.25]),
            previous_positions=torch.zeros(2),
            realized_returns=torch.zeros(2),
            ex_ante_portfolio_vol=0.20,
            config=config,
        )
        high_risk = compute_factor_timing_reward(
            positions=torch.tensor([0.50, -0.25]),
            previous_positions=torch.zeros(2),
            realized_returns=torch.zeros(2),
            ex_ante_portfolio_vol=0.60,
            config=config,
        )

        self.assertAlmostEqual(low_risk.risk_penalty, 0.02, places=6)
        self.assertAlmostEqual(high_risk.risk_penalty, 0.06, places=6)
        self.assertAlmostEqual(
            low_risk.concentration_penalty,
            high_risk.concentration_penalty,
            places=6,
        )

    def test_max_drawdown_includes_initial_wealth_baseline(self) -> None:
        from indosteg.strats.factor_timing import max_drawdown

        self.assertAlmostEqual(
            max_drawdown(torch.tensor([-0.10, 0.05])),
            -0.10,
            places=6,
        )

    def test_performance_summary_includes_trading_and_statistical_diagnostics(
        self,
    ) -> None:
        from indosteg.strats.factor_timing import performance_summary

        returns = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.02, 0.00])
        turnover = torch.tensor([0.10, 0.20, 0.15, 0.05, 0.10, 0.12])
        costs = torch.tensor([0.001, 0.002, 0.0015, 0.0005, 0.001, 0.0012])
        positions = torch.tensor(
            [
                [0.40, -0.10],
                [0.30, -0.20],
                [0.50, 0.00],
                [0.20, -0.10],
                [0.10, 0.10],
                [0.00, 0.20],
            ]
        )

        summary = performance_summary(
            returns,
            turnover=turnover,
            transaction_costs=costs,
            positions=positions,
        )

        for key in (
            "annualized_return",
            "annualized_volatility",
            "downside_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "hit_rate",
            "skew",
            "excess_kurtosis",
            "t_stat",
            "hac_t_stat",
            "turnover",
            "transaction_cost_drag",
            "gross_exposure",
            "net_exposure",
            "concentration",
            "effective_bets",
            "average_gross_exposure",
            "average_net_exposure",
        ):
            self.assertIn(key, summary)
            self.assertTrue(math.isfinite(summary[key]))

        self.assertAlmostEqual(summary["turnover"], float(turnover.mean().item()))
        self.assertAlmostEqual(
            summary["transaction_cost_drag"],
            float(costs.sum().item()),
        )
        self.assertAlmostEqual(
            summary["gross_exposure"],
            float(positions.abs().sum(dim=1).mean().item()),
        )

    def test_research_diagnostics_cover_baselines_costs_and_capacity(self) -> None:
        from indosteg.strats.factor_timing import (
            baseline_comparison,
            capacity_usage,
            cost_sensitivity,
            equal_weight_positions,
            inverse_volatility_positions,
            momentum_positions,
            named_factor_contribution,
            parameter_count,
            reversal_positions,
            reward_terms_summary,
            zero_positions,
        )
        from indosteg.strats.factor_timing.schema import RewardTerms

        returns = torch.tensor([0.01, -0.01, 0.02, 0.00])
        turnover = torch.tensor([0.10, 0.20, 0.10, 0.05])
        positions = torch.tensor([[0.5, -0.2], [0.4, -0.1]])
        realized = torch.tensor([[0.01, -0.02], [0.03, 0.01]])

        sensitivity = cost_sensitivity(returns, turnover, (0.0, 0.01))
        comparison = baseline_comparison(
            returns,
            {"zero": torch.zeros_like(returns)},
        )
        contributions = named_factor_contribution(
            positions,
            realized,
            ("factor_a", "factor_b"),
        )
        reward_summary = reward_terms_summary(
            (
                RewardTerms(1.0, 0.2, 0.01, 0.02, 0.03, 0.04, 0.10, 0.5),
                RewardTerms(2.0, 0.3, 0.02, 0.03, 0.04, 0.05, 0.16, 0.6),
            )
        )

        self.assertEqual(tuple(zero_positions(2).shape), (2,))
        self.assertAlmostEqual(float(equal_weight_positions(2).sum().item()), 1.0)
        self.assertAlmostEqual(
            float(inverse_volatility_positions(torch.tensor([0.2, 0.4])).sum().item()),
            1.0,
            places=6,
        )
        self.assertTrue(
            torch.allclose(momentum_positions(realized), -reversal_positions(realized))
        )
        self.assertLess(
            sensitivity[0.01]["annualized_return"],
            sensitivity[0.0]["annualized_return"],
        )
        self.assertIn("active_vs_zero", comparison)
        self.assertIn("factor_a", contributions)
        self.assertIn(
            "average_capacity_usage", capacity_usage(positions, torch.ones(2))
        )
        self.assertEqual(parameter_count(torch.nn.Linear(2, 1)), 3)
        self.assertAlmostEqual(reward_summary["mean_reward"], 1.5, places=6)

    def test_subperiod_performance_preserves_aligned_diagnostics(self) -> None:
        from indosteg.strats.factor_timing import subperiod_performance

        returns = torch.tensor([0.01, -0.02, 0.03, -0.01])
        turnover = torch.tensor([0.10, 0.20, 0.30, 0.40])
        costs = torch.tensor([0.001, 0.002, 0.003, 0.004])
        positions = torch.tensor(
            [
                [0.40, -0.10],
                [0.20, -0.20],
                [0.30, 0.10],
                [0.00, 0.20],
            ]
        )

        summary = subperiod_performance(
            returns,
            ("early", "early", "late", "late"),
            periods_per_year=12,
            turnover=turnover,
            transaction_costs=costs,
            positions=positions,
            hac_lags=1,
        )

        self.assertIn("turnover", summary["early"])
        self.assertIn("transaction_cost_drag", summary["early"])
        self.assertIn("gross_exposure", summary["late"])
        self.assertAlmostEqual(summary["early"]["turnover"], 0.15, places=6)
        self.assertAlmostEqual(
            summary["late"]["transaction_cost_drag"],
            float(costs[2:].sum().item()),
            places=6,
        )
        self.assertAlmostEqual(
            summary["late"]["gross_exposure"],
            float(positions[2:].abs().sum(dim=1).mean().item()),
            places=6,
        )

    def test_walk_forward_windows_support_purging_embargo_and_sample_counts(
        self,
    ) -> None:
        from indosteg.strats.factor_timing import (
            purged_embargoed_train_indices,
            rolling_walk_forward_windows,
            walk_forward_sample_counts,
        )

        indices = purged_embargoed_train_indices(
            total_observations=12,
            train_start=0,
            train_end=10,
            test_start=4,
            test_end=6,
            label_horizon=2,
            embargo=2,
        )
        windows = rolling_walk_forward_windows(
            total_observations=20,
            train_size=8,
            validation_size=2,
            test_size=3,
            label_horizon=2,
            embargo=1,
        )
        counts = walk_forward_sample_counts(windows)

        self.assertTrue(torch.equal(indices, torch.tensor([0, 1, 2, 8, 9])))
        self.assertGreater(len(windows), 0)
        self.assertTrue(torch.equal(windows[0].train_indices, torch.arange(0, 7)))
        self.assertLessEqual(counts[0]["train_sample_count"], 8)
        self.assertEqual(counts[0]["validation_sample_count"], 2)

    def test_build_sac_transitions_accepts_explicit_terminal_flags(self) -> None:
        from indosteg.strats.factor_timing import build_sac_transitions

        observations = tuple(
            {"panel": torch.tensor([float(index)])} for index in range(4)
        )
        transitions = build_sac_transitions(
            observations=observations,
            actions=torch.zeros(3, 2),
            rewards=torch.arange(3, dtype=torch.float32),
            terminated=torch.tensor([False, True, False]),
        )
        default_transitions = build_sac_transitions(
            observations=observations,
            actions=torch.zeros(3, 2),
            rewards=torch.arange(3, dtype=torch.float32),
        )

        self.assertEqual(
            [bool(transition.terminated) for transition in transitions],
            [False, True, False],
        )
        self.assertEqual(
            [bool(transition.terminated) for transition in default_transitions],
            [False, False, True],
        )

    def test_panel_and_direct_observation_paths_are_swappable(self) -> None:
        from indosteg.strats.factor_timing import (
            FactorEntity,
            FactorTimingPanel,
            build_factor_timing_observation,
            build_factor_timing_observation_from_panel,
        )

        entities = (
            FactorEntity("context_0", group="context"),
            FactorEntity("action_b", tradable=True),
            FactorEntity("context_1", group="context"),
            FactorEntity("action_a", tradable=True),
        )
        values = torch.arange(6 * 4 * 2, dtype=torch.float32).reshape(6, 4, 2)
        action_mask = torch.ones(6, 2, dtype=torch.bool)
        action_mask[4, 1] = False
        panel = FactorTimingPanel(
            values=values,
            dates=tuple(range(6)),
            entities=entities,
            feature_names=("feature_0", "feature_1"),
            action_entities=("action_a", "action_b"),
            realized_forward_returns=torch.randn(6, 2),
            action_availability_mask=action_mask,
            metadata={"source": "fixture"},
        )
        previous_positions = torch.tensor([0.25, -0.10])

        from_panel = build_factor_timing_observation_from_panel(
            panel,
            end=5,
            lookback=3,
            previous_positions=previous_positions,
            patch_length=3,
            patch_stride=3,
        )
        direct = build_factor_timing_observation(
            panel=panel.window(end=5, lookback=3),
            previous_positions=previous_positions,
            universe=panel.universe,
            tradable_mask=panel.action_mask(4),
            patch_length=3,
            patch_stride=3,
        )

        self.assertTrue(torch.equal(from_panel["panel"], direct["panel"]))
        self.assertTrue(
            torch.equal(
                from_panel["tradable_entity_indices"],
                torch.tensor([3, 1], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                from_panel["tradable_entity_indices"],
                direct["tradable_entity_indices"],
            )
        )
        self.assertTrue(
            torch.equal(from_panel["tradable_mask"], direct["tradable_mask"])
        )
        self.assertEqual(from_panel["metadata"]["source"], "fixture")

    def test_model_shapes_allow_action_dim_to_differ_from_observed_entities(
        self,
    ) -> None:
        from indosteg.contracts import SACActionDistribution
        from indosteg.strats.factor_timing import (
            FactorTimingModelConfig,
            FactorTimingSACSystem,
            build_factor_timing_observation,
        )

        universe = self._interleaved_action_universe()
        observation = build_factor_timing_observation(
            panel=torch.randn(universe.observed_entity_count, 6, 4),
            context=torch.randn(3),
            previous_positions=torch.zeros(universe.action_dim),
            universe=universe,
            patch_length=3,
            patch_stride=3,
        )
        config = FactorTimingModelConfig(
            input_dim=4,
            observed_entity_count=universe.observed_entity_count,
            action_dim=universe.action_dim,
            context_dim=3,
            lookback=6,
            patch_length=3,
            patch_stride=3,
            model_dim=16,
            num_heads=4,
            num_temporal_layers=1,
            num_cross_sectional_layers=1,
        )
        system = FactorTimingSACSystem(config)

        distribution = system.actor.distribution(observation)
        q_value_a, q_value_b = system.critic.q_values(
            observation,
            torch.zeros(1, universe.action_dim),
        )
        shifted_observation = {
            **observation,
            "previous_positions": torch.ones(universe.action_dim),
        }
        shifted_q_value_a, _ = system.critic.q_values(
            shifted_observation,
            torch.zeros(1, universe.action_dim),
        )

        self.assertIsInstance(distribution, SACActionDistribution)
        self.assertEqual(tuple(distribution.mean.shape), (1, universe.action_dim))
        self.assertEqual(tuple(distribution.log_std.shape), (1, universe.action_dim))
        self.assertEqual(tuple(q_value_a.shape), (1, 1))
        self.assertEqual(tuple(q_value_b.shape), (1, 1))
        self.assertFalse(torch.allclose(q_value_a, shifted_q_value_a))

    def test_model_consumes_masks_and_validates_batched_action_mapping(self) -> None:
        from indosteg.algorithms.sac.actions import deterministic_tanh_action
        from indosteg.strats.factor_timing import (
            FactorTimingModelConfig,
            FactorTimingSACSystem,
            build_factor_timing_observation,
        )

        universe = self._interleaved_action_universe()
        previous_positions = torch.tensor([0.35, -0.20])
        observation = build_factor_timing_observation(
            panel=torch.randn(universe.observed_entity_count, 6, 4),
            context=torch.randn(3),
            previous_positions=previous_positions,
            universe=universe,
            tradable_mask=torch.tensor([False, True]),
            patch_length=3,
            patch_stride=3,
        )
        config = FactorTimingModelConfig(
            input_dim=4,
            observed_entity_count=universe.observed_entity_count,
            action_dim=universe.action_dim,
            context_dim=3,
            lookback=6,
            patch_length=3,
            patch_stride=3,
            model_dim=16,
            num_heads=4,
            num_temporal_layers=1,
            num_cross_sectional_layers=1,
        )
        system = FactorTimingSACSystem(config)

        distribution = system.actor.distribution(observation)
        deterministic_action = deterministic_tanh_action(distribution).squeeze(0)
        q_value_a, q_value_b = system.critic.q_values(
            observation,
            torch.tensor([[-1.0, 0.40]]),
        )
        shifted_q_value_a, shifted_q_value_b = system.critic.q_values(
            observation,
            torch.tensor([[1.0, 0.40]]),
        )
        mismatched_mapping = {
            key: torch.stack((value, value), dim=0)
            for key, value in observation.items()
            if isinstance(value, torch.Tensor)
        }
        mismatched_mapping["tradable_entity_indices"] = torch.stack(
            (
                observation["tradable_entity_indices"],
                observation["tradable_entity_indices"].flip(0),
            ),
            dim=0,
        )

        self.assertAlmostEqual(
            float(deterministic_action[0].item()),
            float(previous_positions[0].item()),
            places=5,
        )
        self.assertLess(float(distribution.log_std[0, 0].item()), -4.0)
        self.assertTrue(torch.allclose(q_value_a, shifted_q_value_a))
        self.assertTrue(torch.allclose(q_value_b, shifted_q_value_b))
        with self.assertRaises(ValueError):
            system.actor.distribution(mismatched_mapping)

    def test_synthetic_factor_timing_sac_exemplar_runs_end_to_end(self) -> None:
        from test.strats.factor_timing.factor_timed_sac import (
            run_synthetic_factor_timing_sac_exemplar,
        )

        summary = run_synthetic_factor_timing_sac_exemplar(
            total_steps=192,
            warmup_steps=32,
            horizon=32,
            batch_size=24,
            evaluation_episodes=2,
        )

        self.assertGreater(summary.observed_entity_count, summary.action_dim)
        self.assertGreater(summary.update_steps, 0)
        self.assertGreaterEqual(summary.replay_size, 192)
        self.assertTrue(math.isfinite(summary.initial_eval_reward))
        self.assertTrue(math.isfinite(summary.final_eval_reward))
        self.assertTrue(math.isfinite(summary.final_eval_net_pnl))
        self.assertTrue(math.isfinite(summary.final_eval_sharpe))
        self.assertGreater(summary.last_critic_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
