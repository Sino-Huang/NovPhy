"""Contract tests for the Milestone 1a/1b JEPA backbone.

These pin the two structural guarantees the research plan depends on:

- the continuous latent ``z`` is the SOLE rollout state carrier, so the
  mode-head readouts must be graph-independent of the carrier and must never
  be constructed on the rollout path;
- the target encoder is a genuine EMA/stop-grad copy, so it must never
  receive a gradient and must never contain batch-dependent normalization.
"""
import unittest

import torch
from torch import nn

from world_model.data.types import ContractValueError
from world_model.model import (
    Abstraction,
    ContextEncoder,
    DualOutputPredictor,
    EmaTargetEncoder,
    EncoderConfig,
    JepaBackbone,
    JepaConfig,
    MACRO_TRANSITION_INPUTS,
    MICRO_TRANSITION_INPUTS,
    MacroReadout,
    PredictionPair,
    PredictorConfig,
    build_encoder,
    mode_weight,
)

_LATENT_DIM = 24
_ACTION_DIM = 5
_MICRO_PREDICATES = 7
_MACRO_PREDICATES = 5
_EVENT_TYPES = 10


def small_encoder_config(**overrides: object) -> EncoderConfig:
    """Return a deliberately tiny encoder configuration for fast unit tests."""
    fields: dict[str, object] = {
        "input_height": 32,
        "input_width": 40,
        "stem_channels": 8,
        "stage_channels": (8, 16, 24),
        "blocks_per_stage": 1,
        "group_norm_groups": 4,
        "latent_dim": _LATENT_DIM,
        "pool_heads": 2,
    }
    fields.update(overrides)
    return EncoderConfig(**fields)  # type: ignore[arg-type]


def small_predictor_config(**overrides: object) -> PredictorConfig:
    """Return a tiny predictor configuration matching ``small_encoder_config``."""
    fields: dict[str, object] = {
        "latent_dim": _LATENT_DIM,
        "action_dim": _ACTION_DIM,
        "hidden_dim": 32,
        "depth": 3,
        "pair_code_dim": 16,
        "delta_frequency_count": 4,
        "micro_predicate_count": _MICRO_PREDICATES,
        "macro_predicate_count": _MACRO_PREDICATES,
        "event_type_count": _EVENT_TYPES,
    }
    fields.update(overrides)
    return PredictorConfig(**fields)  # type: ignore[arg-type]


def small_jepa_config(**overrides: object) -> JepaConfig:
    """Return a tiny end-to-end backbone configuration."""
    fields: dict[str, object] = {
        "encoder": small_encoder_config(),
        "predictor": small_predictor_config(),
    }
    fields.update(overrides)
    return JepaConfig(**fields)  # type: ignore[arg-type]


class TinyModule(nn.Module):
    """A two-parameter module with a non-float buffer, for exact EMA arithmetic."""

    def __init__(self, weight: float, bias: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2, 2), weight))
        self.bias = nn.Parameter(torch.full((2,), bias))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.weight + self.bias


class SpyHead(nn.Module):
    """Wrap a readout head and count how many times it is invoked."""

    def __init__(self, wrapped: nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.call_count = 0

    def forward(self, *args: object, **kwargs: object) -> object:
        self.call_count += 1
        return self.wrapped(*args, **kwargs)


class SpyAdapter(SpyHead):
    """Count calls through one selected transition adapter."""


# ---------------------------------------------------------------------------
# Configuration contracts
# ---------------------------------------------------------------------------


class ModelConfigTests(unittest.TestCase):
    def test_transition_input_vocabulary_matches_the_cohort_v2_contract(self) -> None:
        self.assertEqual(MICRO_TRANSITION_INPUTS, ("contact", "supports"))
        self.assertEqual(
            MACRO_TRANSITION_INPUTS,
            ("steady-state", "structure-unstable"),
        )
        config = PredictorConfig()
        self.assertEqual(config.micro_predicate_count, len(MICRO_TRANSITION_INPUTS))
        self.assertEqual(config.macro_predicate_count, len(MACRO_TRANSITION_INPUTS))
        self.assertIn("jepa-predictor-config-v2", config.identity)

    def test_prediction_pair_accepts_the_declared_abstractions(self) -> None:
        for abstraction in Abstraction:
            pair = PredictionPair(delta=1, abstraction=abstraction)
            self.assertIs(pair.abstraction, abstraction)

    def test_prediction_pair_coerces_a_valid_abstraction_string(self) -> None:
        pair = PredictionPair(delta=2, abstraction="macro")
        self.assertIs(pair.abstraction, Abstraction.MACRO)

    def test_prediction_pair_rejects_an_unknown_abstraction(self) -> None:
        with self.assertRaises(ContractValueError):
            PredictionPair(delta=1, abstraction="relational")

    def test_prediction_pair_rejects_a_nonpositive_delta(self) -> None:
        with self.assertRaises(ContractValueError):
            PredictionPair(delta=0, abstraction=Abstraction.CONTINUOUS)

    def test_prediction_pair_rejects_a_non_integer_delta(self) -> None:
        with self.assertRaises(ContractValueError):
            PredictionPair(delta=1.0, abstraction=Abstraction.CONTINUOUS)

    def test_prediction_pair_is_frozen(self) -> None:
        pair = PredictionPair(delta=1, abstraction=Abstraction.MICRO)
        with self.assertRaises(AttributeError):
            pair.delta = 2  # type: ignore[misc]

    def test_encoder_config_rejects_channels_indivisible_by_group_count(self) -> None:
        with self.assertRaises(ContractValueError):
            small_encoder_config(stage_channels=(6, 16, 24), group_norm_groups=4)

    def test_encoder_config_rejects_an_empty_stage_tuple(self) -> None:
        with self.assertRaises(ContractValueError):
            small_encoder_config(stage_channels=())

    def test_predictor_config_rejects_a_nonpositive_depth(self) -> None:
        with self.assertRaises(ContractValueError):
            small_predictor_config(depth=0)

    def test_configuration_identity_is_stable_and_discriminating(self) -> None:
        left = small_jepa_config()
        right = small_jepa_config()
        self.assertEqual(left.identity, right.identity)
        widened = JepaConfig(
            encoder=small_encoder_config(),
            predictor=small_predictor_config(hidden_dim=64),
        )
        self.assertNotEqual(left.identity, widened.identity)


# ---------------------------------------------------------------------------
# 1a — context encoder
# ---------------------------------------------------------------------------


class ContextEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = small_encoder_config()
        self.encoder = ContextEncoder(self.config)
        self.images = torch.randn(3, 3, self.config.input_height, self.config.input_width)

    def test_forward_returns_a_pooled_carrier_and_a_token_grid(self) -> None:
        output = self.encoder(self.images)
        self.assertEqual(output.latent.shape, torch.Size([3, _LATENT_DIM]))
        self.assertEqual(output.tokens.ndim, 3)
        self.assertEqual(output.tokens.shape[0], 3)
        self.assertEqual(output.tokens.shape[2], self.config.stage_channels[-1])

    def test_the_carrier_is_the_pooled_vector_not_the_token_grid(self) -> None:
        output = self.encoder(self.images)
        # The token grid is a side output reserved for Milestone 1c (SPSG); it
        # must not be the thing the rollout carries.
        self.assertNotEqual(output.latent.shape, output.tokens.shape)
        self.assertGreater(output.tokens.shape[1], 1)

    def test_forward_is_deterministic_for_a_fixed_seed(self) -> None:
        torch.manual_seed(7)
        first = ContextEncoder(self.config)(self.images)
        torch.manual_seed(7)
        second = ContextEncoder(self.config)(self.images)
        torch.testing.assert_close(first.latent, second.latent)

    def test_forward_rejects_a_non_batched_image(self) -> None:
        with self.assertRaises(ContractValueError):
            self.encoder(self.images[0])

    def test_forward_rejects_a_wrong_channel_count(self) -> None:
        with self.assertRaises(ContractValueError):
            self.encoder(torch.randn(2, 1, self.config.input_height, self.config.input_width))

    def test_the_encoder_contains_no_batch_normalization(self) -> None:
        # Batch statistics would couple samples inside a batch and corrupt both
        # the stop-grad target branch and the EMA update.
        offenders = [
            type(module).__name__
            for module in self.encoder.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        self.assertEqual(offenders, [])

    def test_build_encoder_resolves_the_registered_backbone(self) -> None:
        encoder = build_encoder(self.config)
        self.assertIsInstance(encoder, ContextEncoder)

    def test_build_encoder_rejects_an_unregistered_backbone(self) -> None:
        with self.assertRaises(ContractValueError):
            build_encoder(small_encoder_config(name="vit_tiny_p16"))


# ---------------------------------------------------------------------------
# 1a — EMA / stop-grad target encoder
# ---------------------------------------------------------------------------


class EmaTargetEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.online = TinyModule(0.5, -0.25)
        self.target = EmaTargetEncoder(self.online, base_momentum=0.9)

    def test_construction_deep_copies_the_online_parameters(self) -> None:
        online_params = dict(self.online.named_parameters())
        for name, parameter in self.target.module.named_parameters():
            torch.testing.assert_close(parameter, online_params[name])
            self.assertIsNot(parameter, online_params[name])

    def test_update_applies_the_exact_ema_rule(self) -> None:
        with torch.no_grad():
            self.online.weight.fill_(1.5)
            self.online.bias.fill_(0.75)
        self.target.update(self.online, momentum=0.9)
        # target <- 0.9 * 0.5 + 0.1 * 1.5 = 0.6
        torch.testing.assert_close(
            self.target.module.weight, torch.full((2, 2), 0.6), atol=1e-7, rtol=0.0
        )
        # target <- 0.9 * -0.25 + 0.1 * 0.75 = -0.15
        torch.testing.assert_close(
            self.target.module.bias, torch.full((2,), -0.15), atol=1e-7, rtol=0.0
        )

    def test_update_before_any_optimizer_step_is_a_value_no_op(self) -> None:
        self.target.update(self.online, momentum=0.9)
        torch.testing.assert_close(self.target.module.weight, self.online.weight)
        torch.testing.assert_close(self.target.module.bias, self.online.bias)

    def test_non_float_buffers_are_copied_rather_than_interpolated(self) -> None:
        with torch.no_grad():
            self.online.step_count.fill_(11)
        self.target.update(self.online, momentum=0.9)
        self.assertEqual(self.target.module.step_count.item(), 11)
        self.assertEqual(self.target.module.step_count.dtype, torch.long)

    def test_every_target_parameter_is_detached_from_autograd(self) -> None:
        for parameter in self.target.parameters():
            self.assertFalse(parameter.requires_grad)

    def test_the_target_branch_is_a_stop_gradient(self) -> None:
        values = torch.ones(1, 2, requires_grad=True)
        prediction = self.target(values)
        self.assertFalse(prediction.requires_grad)

    def test_no_gradient_reaches_the_target_after_backward(self) -> None:
        values = torch.ones(1, 2, requires_grad=True)
        online_out = self.online(values)
        target_out = self.target(values)
        (online_out - target_out).pow(2).sum().backward()
        for parameter in self.target.parameters():
            self.assertIsNone(parameter.grad)
        self.assertIsNotNone(self.online.weight.grad)

    def test_the_momentum_schedule_ramps_from_base_to_final(self) -> None:
        total = 100
        schedule = [self.target.momentum_at(step, total) for step in range(total + 1)]
        self.assertAlmostEqual(schedule[0], 0.9, places=6)
        self.assertAlmostEqual(schedule[-1], 1.0, places=6)
        for earlier, later in zip(schedule, schedule[1:]):
            self.assertLessEqual(earlier, later + 1e-9)
        self.assertTrue(all(0.9 - 1e-9 <= value <= 1.0 + 1e-9 for value in schedule))

    def test_the_momentum_schedule_clamps_a_step_beyond_the_horizon(self) -> None:
        self.assertAlmostEqual(self.target.momentum_at(500, 100), 1.0, places=6)

    def test_update_rejects_a_momentum_outside_the_unit_interval(self) -> None:
        with self.assertRaises(ContractValueError):
            self.target.update(self.online, momentum=1.5)


# ---------------------------------------------------------------------------
# 1b — dual-output predictor
# ---------------------------------------------------------------------------


class DualOutputPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = small_predictor_config()
        self.predictor = DualOutputPredictor(self.config)
        self.latent = torch.randn(4, _LATENT_DIM)
        self.action = torch.randn(4, _ACTION_DIM)

    def mode_input(self, pair: PredictionPair, *, fill: float = 1.0) -> torch.Tensor | None:
        widths = {
            Abstraction.MICRO: _MICRO_PREDICATES,
            Abstraction.MACRO: _MACRO_PREDICATES,
        }
        width = widths.get(pair.abstraction)
        return None if width is None else torch.full((4, width), fill)

    def all_pairs(self) -> list[PredictionPair]:
        return [
            PredictionPair(delta=delta, abstraction=abstraction)
            for delta in (1, 2, 4)
            for abstraction in Abstraction
        ]

    def test_the_carrier_is_emitted_for_every_pair_in_the_grid(self) -> None:
        for pair in self.all_pairs():
            with self.subTest(pair=pair):
                output = self.predictor(
                    self.latent, self.action, pair, self.mode_input(pair)
                )
                self.assertIsNotNone(output.carrier)
                self.assertEqual(output.carrier.shape, torch.Size([4, _LATENT_DIM]))

    def test_the_mode_head_readout_is_masked_by_the_selected_abstraction(self) -> None:
        expectations = {
            Abstraction.CONTINUOUS: (False, False),
            Abstraction.MICRO: (True, False),
            Abstraction.MACRO: (False, True),
        }
        for pair in self.all_pairs():
            micro_expected, macro_expected = expectations[pair.abstraction]
            with self.subTest(pair=pair):
                output = self.predictor(
                    self.latent, self.action, pair, self.mode_input(pair)
                )
                self.assertEqual(output.micro_readout is not None, micro_expected)
                self.assertEqual(output.macro_readout is not None, macro_expected)

    def test_the_micro_readout_has_the_declared_predicate_width(self) -> None:
        output = self.predictor(
            self.latent,
            self.action,
            PredictionPair(delta=1, abstraction=Abstraction.MICRO),
            torch.ones(4, _MICRO_PREDICATES),
        )
        self.assertEqual(output.micro_readout.shape, torch.Size([4, _MICRO_PREDICATES]))

    def test_the_macro_readout_carries_state_duration_and_event_terms(self) -> None:
        output = self.predictor(
            self.latent,
            self.action,
            PredictionPair(delta=1, abstraction=Abstraction.MACRO),
            torch.ones(4, _MACRO_PREDICATES),
        )
        readout = output.macro_readout
        self.assertIsInstance(readout, MacroReadout)
        self.assertEqual(readout.macro_logits.shape, torch.Size([4, _MACRO_PREDICATES]))
        self.assertEqual(readout.delta_hat.shape, torch.Size([4, 1]))
        self.assertEqual(readout.event_logits.shape, torch.Size([4, _EVENT_TYPES]))

    def test_the_carrier_is_graph_independent_of_the_mode_heads(self) -> None:
        # The state-carrier principle: heads are readouts that enter only the
        # loss.  If any head parameter could influence the carrier, a symbolic
        # decode would sit inside the rollout path.
        head_parameters = [
            *self.predictor.micro_head.parameters(),
            *self.predictor.macro_head.parameters(),
        ]
        self.assertNotEqual(head_parameters, [])
        for pair in self.all_pairs():
            with self.subTest(pair=pair):
                output = self.predictor(
                    self.latent, self.action, pair, self.mode_input(pair)
                )
                gradients = torch.autograd.grad(
                    output.carrier.sum(),
                    head_parameters,
                    allow_unused=True,
                    retain_graph=True,
                )
                self.assertTrue(all(gradient is None for gradient in gradients))

    def test_the_head_loss_still_reaches_the_latent(self) -> None:
        latent = self.latent.clone().requires_grad_(True)
        output = self.predictor(
            latent,
            self.action,
            PredictionPair(delta=1, abstraction=Abstraction.MICRO),
            torch.ones(4, _MICRO_PREDICATES),
        )
        output.micro_readout.sum().backward()
        self.assertIsNotNone(latent.grad)
        self.assertGreater(latent.grad.abs().sum().item(), 0.0)

    def test_the_rollout_path_never_constructs_a_mode_head(self) -> None:
        micro_spy = SpyHead(self.predictor.micro_head)
        macro_spy = SpyHead(self.predictor.macro_head)
        self.predictor.micro_head = micro_spy
        self.predictor.macro_head = macro_spy
        pairs = (
            PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS),
            PredictionPair(delta=2, abstraction=Abstraction.MICRO),
            PredictionPair(delta=4, abstraction=Abstraction.MACRO),
            PredictionPair(delta=1, abstraction=Abstraction.MICRO),
        )
        mode_inputs = tuple(self.mode_input(pair) for pair in pairs)
        carriers = self.predictor.rollout(
            self.latent, self.action, pairs, mode_inputs=mode_inputs
        )
        self.assertEqual(len(carriers), len(pairs))
        for carrier in carriers:
            self.assertEqual(carrier.shape, torch.Size([4, _LATENT_DIM]))
        self.assertEqual(micro_spy.call_count, 0)
        self.assertEqual(macro_spy.call_count, 0)

    def test_a_decision_executes_only_its_selected_adapter_and_readout(self) -> None:
        for pair in self.all_pairs():
            with self.subTest(pair=pair):
                predictor = DualOutputPredictor(self.config)
                predictor.continuous_adapter = SpyAdapter(
                    predictor.continuous_adapter
                )
                predictor.micro_adapter = SpyAdapter(predictor.micro_adapter)
                predictor.macro_adapter = SpyAdapter(predictor.macro_adapter)
                predictor.micro_head = SpyHead(predictor.micro_head)
                predictor.macro_head = SpyHead(predictor.macro_head)

                predictor(self.latent, self.action, pair, self.mode_input(pair))

                adapter_counts = {
                    Abstraction.CONTINUOUS: predictor.continuous_adapter.call_count,
                    Abstraction.MICRO: predictor.micro_adapter.call_count,
                    Abstraction.MACRO: predictor.macro_adapter.call_count,
                }
                self.assertEqual(adapter_counts[pair.abstraction], 1)
                self.assertEqual(sum(adapter_counts.values()), 1)
                self.assertEqual(
                    predictor.micro_head.call_count,
                    int(pair.abstraction is Abstraction.MICRO),
                )
                self.assertEqual(
                    predictor.macro_head.call_count,
                    int(pair.abstraction is Abstraction.MACRO),
                )

    def test_the_rollout_chains_the_carrier_forward(self) -> None:
        pair = PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS)
        carriers = self.predictor.rollout(self.latent, self.action, (pair, pair))
        first = self.predictor(self.latent, self.action, pair).carrier
        second = self.predictor(first, self.action, pair).carrier
        torch.testing.assert_close(carriers[0], first)
        torch.testing.assert_close(carriers[1], second)

    def _make_conditioning_live(self) -> None:
        """Move the FiLM modulation off its zero-init identity start.

        The FiLM scale/shift begin at zero so the predictor starts as a plain
        residual MLP; conditioning must become live once the modulation weights
        are nonzero.  This helper perturbs them so the tests below verify the
        wiring (pair code -> FiLM -> carrier), not the initialization choice.
        """
        torch.manual_seed(0)
        with torch.no_grad():
            for block in self.predictor.blocks:
                block.modulation.weight.add_(
                    torch.randn_like(block.modulation.weight) * 0.1
                )
                block.modulation.bias.add_(torch.randn_like(block.modulation.bias) * 0.1)

    def test_the_horizon_changes_the_carrier(self) -> None:
        self._make_conditioning_live()
        pair_short = PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS)
        pair_long = PredictionPair(delta=4, abstraction=Abstraction.CONTINUOUS)
        short = self.predictor(self.latent, self.action, pair_short).carrier
        long = self.predictor(self.latent, self.action, pair_long).carrier
        self.assertFalse(torch.allclose(short, long))

    def test_the_abstraction_changes_the_carrier(self) -> None:
        self._make_conditioning_live()
        pair_continuous = PredictionPair(delta=2, abstraction=Abstraction.CONTINUOUS)
        pair_macro = PredictionPair(delta=2, abstraction=Abstraction.MACRO)
        continuous = self.predictor(self.latent, self.action, pair_continuous).carrier
        macro = self.predictor(
            self.latent,
            self.action,
            pair_macro,
            self.mode_input(pair_macro, fill=0.0),
        ).carrier
        self.assertFalse(torch.allclose(continuous, macro))

    def test_symbolic_content_changes_the_carrier_separately_from_mode_identity(self) -> None:
        self._make_conditioning_live()
        pair = PredictionPair(delta=2, abstraction=Abstraction.MICRO)
        without_content = self.predictor(
            self.latent, self.action, pair, self.mode_input(pair, fill=0.0)
        ).carrier
        with_content = self.predictor(
            self.latent, self.action, pair, self.mode_input(pair, fill=1.0)
        ).carrier
        continuous = self.predictor(
            self.latent,
            self.action,
            PredictionPair(delta=2, abstraction=Abstraction.CONTINUOUS),
        ).carrier

        self.assertFalse(torch.allclose(without_content, with_content))
        self.assertFalse(torch.allclose(without_content, continuous))

    def test_symbolic_modes_fail_closed_without_their_selected_content(self) -> None:
        for abstraction in (Abstraction.MICRO, Abstraction.MACRO):
            with self.subTest(abstraction=abstraction):
                with self.assertRaises(ContractValueError):
                    self.predictor(
                        self.latent,
                        self.action,
                        PredictionPair(delta=1, abstraction=abstraction),
                    )

    def test_continuous_mode_rejects_symbolic_content_instead_of_ignoring_it(self) -> None:
        with self.assertRaises(ContractValueError):
            self.predictor(
                self.latent,
                self.action,
                PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS),
                torch.ones(4, _MICRO_PREDICATES),
            )

    def test_the_pair_conditioning_is_joint_rather_than_additive(self) -> None:
        # An additively factorized conditioner satisfies the parallelogram
        # identity  c(d1,a1) + c(d2,a2) == c(d1,a2) + c(d2,a1).  A joint,
        # nonlinearly fused code must break it (proposal section 4.4).
        conditioner = self.predictor.conditioner
        code = lambda delta, abstraction: conditioner.code(  # noqa: E731
            PredictionPair(delta=delta, abstraction=abstraction), 1, torch.device("cpu")
        )
        diagonal = code(2, Abstraction.MICRO) + code(4, Abstraction.MACRO)
        antidiagonal = code(2, Abstraction.MACRO) + code(4, Abstraction.MICRO)
        self.assertFalse(torch.allclose(diagonal, antidiagonal, atol=1e-5))

    def test_the_action_conditions_the_carrier(self) -> None:
        pair = PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS)
        first = self.predictor(self.latent, self.action, pair).carrier
        second = self.predictor(self.latent, self.action + 1.0, pair).carrier
        self.assertFalse(torch.allclose(first, second))

    def test_forward_rejects_a_non_batched_latent(self) -> None:
        pair = PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS)
        with self.assertRaises(ContractValueError):
            self.predictor(self.latent[0], self.action, pair)

    def test_forward_rejects_a_wrong_latent_width(self) -> None:
        pair = PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS)
        with self.assertRaises(ContractValueError):
            self.predictor(torch.randn(4, _LATENT_DIM + 1), self.action, pair)

    def test_forward_rejects_a_wrong_action_width(self) -> None:
        pair = PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS)
        with self.assertRaises(ContractValueError):
            self.predictor(self.latent, torch.randn(4, _ACTION_DIM + 1), pair)

    def test_forward_rejects_a_mismatched_batch_size(self) -> None:
        pair = PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS)
        with self.assertRaises(ContractValueError):
            self.predictor(self.latent, torch.randn(3, _ACTION_DIM), pair)

    def test_forward_rejects_a_non_pair_selection(self) -> None:
        with self.assertRaises(ContractValueError):
            self.predictor(self.latent, self.action, (1, "continuous"))

    def test_rollout_rejects_an_empty_pair_sequence(self) -> None:
        with self.assertRaises(ContractValueError):
            self.predictor.rollout(self.latent, self.action, ())


# ---------------------------------------------------------------------------
# 1b — reliability weighting of the symbolic term
# ---------------------------------------------------------------------------


class ModeWeightTests(unittest.TestCase):
    def test_the_continuous_mode_masks_the_symbolic_term(self) -> None:
        self.assertEqual(mode_weight(Abstraction.CONTINUOUS, 0.8), 0.0)

    def test_the_micro_mode_uses_the_reliability_estimate(self) -> None:
        self.assertAlmostEqual(mode_weight(Abstraction.MICRO, 0.8), 0.8)

    def test_the_macro_mode_is_always_supervised(self) -> None:
        self.assertEqual(mode_weight(Abstraction.MACRO, 0.0), 1.0)

    def test_a_masked_term_contributes_exactly_zero_gradient(self) -> None:
        torch.manual_seed(0)
        predictor = DualOutputPredictor(small_predictor_config())
        latent = torch.randn(2, _LATENT_DIM)
        action = torch.randn(2, _ACTION_DIM)
        output = predictor(
            latent,
            action,
            PredictionPair(delta=1, abstraction=Abstraction.MICRO),
            torch.ones(2, _MICRO_PREDICATES),
        )
        weight = mode_weight(Abstraction.CONTINUOUS, 1.0)
        (weight * output.micro_readout.pow(2).sum()).backward()
        for parameter in predictor.micro_head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.abs().sum().item(), 0.0)

    def test_the_reliability_estimate_must_lie_in_the_unit_interval(self) -> None:
        with self.assertRaises(ContractValueError):
            mode_weight(Abstraction.MICRO, 1.4)


# ---------------------------------------------------------------------------
# 1a + 1b — assembled backbone
# ---------------------------------------------------------------------------


class JepaBackboneTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = small_jepa_config()
        self.backbone = JepaBackbone(self.config)
        encoder_config = self.config.encoder
        self.images = torch.randn(
            2, 3, encoder_config.input_height, encoder_config.input_width
        )
        self.action = torch.randn(2, _ACTION_DIM)

    def test_the_target_encoder_starts_equal_to_the_online_encoder(self) -> None:
        online = self.backbone.encode(self.images).latent
        target = self.backbone.encode_target(self.images).latent
        torch.testing.assert_close(online, target)

    def test_the_target_encoding_is_a_stop_gradient(self) -> None:
        self.assertFalse(self.backbone.encode_target(self.images).latent.requires_grad)

    def test_trainable_parameters_exclude_the_target_encoder(self) -> None:
        trainable = {id(parameter) for parameter in self.backbone.trainable_parameters()}
        for parameter in self.backbone.target.parameters():
            self.assertNotIn(id(parameter), trainable)
        for parameter in self.backbone.encoder.parameters():
            self.assertIn(id(parameter), trainable)
        for parameter in self.backbone.predictor.parameters():
            self.assertIn(id(parameter), trainable)

    def test_every_trainable_parameter_requires_a_gradient(self) -> None:
        for parameter in self.backbone.trainable_parameters():
            self.assertTrue(parameter.requires_grad)

    def test_update_target_moves_the_target_toward_the_online_encoder(self) -> None:
        with torch.no_grad():
            for parameter in self.backbone.encoder.parameters():
                parameter.add_(1.0)
        before = self.backbone.encode_target(self.images).latent.clone()
        self.backbone.update_target(step=0, total_steps=100)
        after = self.backbone.encode_target(self.images).latent
        self.assertFalse(torch.allclose(before, after))

    def test_predict_emits_a_carrier_for_the_selected_pair(self) -> None:
        latent = self.backbone.encode(self.images).latent
        pair = PredictionPair(delta=1, abstraction=Abstraction.CONTINUOUS)
        output = self.backbone.predict(latent, self.action, pair)
        self.assertEqual(output.carrier.shape, torch.Size([2, _LATENT_DIM]))
        self.assertIsNone(output.micro_readout)
        self.assertIsNone(output.macro_readout)

    def test_predict_routes_selected_symbolic_content_to_the_carrier(self) -> None:
        latent = self.backbone.encode(self.images).latent
        pair = PredictionPair(delta=1, abstraction=Abstraction.MICRO)

        without_content = self.backbone.predict(
            latent,
            self.action,
            pair,
            torch.zeros(2, _MICRO_PREDICATES),
        ).carrier
        with_content = self.backbone.predict(
            latent,
            self.action,
            pair,
            torch.ones(2, _MICRO_PREDICATES),
        ).carrier

        self.assertFalse(torch.allclose(without_content, with_content))

    def test_the_backbone_contains_no_batch_normalization(self) -> None:
        offenders = [
            type(module).__name__
            for module in self.backbone.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
