from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.run_issue_67_short_unroll import main as issue_67_main
from world_model.data.deployment_temporal import TemporalVisualCarrierAdapter
from world_model.model import PredictorConfig
from world_model.training.lineage_scaling import (
    CarrierKind,
    CarrierLineage,
    ContinuousTransitionExample,
    LineageScalingError,
)
from world_model.training.short_unroll import (
    ShortUnrollTrainingSpec,
    ShortUnrollWindow,
    build_short_unroll_windows,
    evaluate_recursive_carrier,
    load_short_unroll_checkpoint,
    save_short_unroll_checkpoint,
    short_unroll_losses,
    train_short_unroll_predictor,
)


def _transition(
    position: int,
    horizon: int,
    segment_end: int,
) -> ContinuousTransitionExample:
    target_position = min(position + horizon, segment_end)
    return ContinuousTransitionExample(
        identity=f"transition-d{position}-h{horizon}",
        context=torch.full((4,), position / 100.0),
        action=torch.tensor((-0.2, 0.1, 0.6, 0.0, 1.0)),
        target=torch.full((4,), target_position / 100.0),
        physical_diagnostics={},
        decision_index=position,
        horizon=horizon,
        target_decision_index=target_position,
    )


def _lineage(
    name: str,
    *,
    role: str = "training",
    segment_ends: tuple[int, ...] = (60,),
    include_h1: bool = False,
) -> CarrierLineage:
    starts = (0, *segment_ends[:-1])
    transitions = []
    for start, end in zip(starts, segment_ends, strict=True):
        horizons = (1, 15) if include_h1 else (15,)
        for horizon in horizons:
            transitions.extend(
                _transition(position, horizon, end)
                for position in range(start, end, horizon)
            )
    return CarrierLineage(
        trajectory_identity=f"trajectory-{name}",
        scenario_lineage_identity=f"lineage-{name}",
        exposure_role=role,
        source_release_identity="release-fixture",
        carrier=CarrierKind.DEPLOYMENT,
        carrier_identity=TemporalVisualCarrierAdapter.identity,
        transitions=tuple(transitions),
        complete=True,
        decision_count=segment_ends[-1],
        segment_end_positions=segment_ends,
    )


def _spec(
    *,
    name: str = "self-conditioned-h15-u2",
    unroll_steps: int = 2,
    seed: int = 67,
) -> ShortUnrollTrainingSpec:
    baseline = unroll_steps == 1
    return ShortUnrollTrainingSpec(
        name=name,
        unroll_steps=unroll_steps,
        local_loss_weight=1.0,
        unrolled_loss_weight=0.0 if baseline else 1.0,
        carrier_bound=2.0,
        carrier_bound_loss_weight=0.0 if baseline else 0.01,
        optimizer_example_budget=4,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        seed=seed,
        carrier_identity=TemporalVisualCarrierAdapter.identity,
        lineage_manifest_reference="issue-67-training-manifest-v1",
        predictor_config=PredictorConfig(
            latent_dim=4,
            action_dim=5,
            hidden_dim=8,
            depth=1,
            pair_code_dim=4,
            delta_frequency_count=2,
        ),
    )


class _RecordingAddOne(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs: list[torch.Tensor] = []

    def carrier(
        self,
        latent: torch.Tensor,
        _action: torch.Tensor,
        _pair: object,
    ) -> torch.Tensor:
        self.inputs.append(latent.detach().clone())
        return latent + 1.0 + self.anchor * 0.0


class Issue67ShortUnrollTests(unittest.TestCase):
    def test_windows_never_cross_shot_segments(self) -> None:
        lineage = _lineage("two-shots", segment_ends=(60, 120))

        two_step = build_short_unroll_windows((lineage,), unroll_steps=2)
        four_step = build_short_unroll_windows((lineage,), unroll_steps=4)

        self.assertEqual(
            [(item.segment_ordinal, item.start_decision_index) for item in two_step],
            [(1, 0), (1, 15), (1, 30), (2, 60), (2, 75), (2, 90)],
        )
        self.assertEqual(
            [(item.segment_ordinal, item.start_decision_index) for item in four_step],
            [(1, 0), (2, 60)],
        )
        self.assertTrue(all(
            window.transitions[-1].target_decision_index
            <= lineage.segment_ends[window.segment_ordinal - 1]
            for window in (*two_step, *four_step)
        ))

    def test_training_rejects_exposure_role_leakage(self) -> None:
        with self.assertRaisesRegex(LineageScalingError, "deployment training"):
            build_short_unroll_windows(
                (_lineage("train"), _lineage("cal", role="calibration")),
                unroll_steps=2,
            )

    def test_second_step_uses_prediction_instead_of_observed_context(self) -> None:
        first = ContinuousTransitionExample(
            "first",
            torch.zeros(4),
            torch.zeros(5),
            torch.ones(4),
            {},
            decision_index=0,
            horizon=15,
            target_decision_index=15,
        )
        second = ContinuousTransitionExample(
            "second",
            torch.full((4,), 100.0),
            torch.zeros(5),
            torch.full((4,), 2.0),
            {},
            decision_index=15,
            horizon=15,
            target_decision_index=30,
        )
        model = _RecordingAddOne()

        losses = short_unroll_losses(
            model,
            (ShortUnrollWindow(1, 1, 0, (first, second)),),
            carrier_bound=2.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(torch.equal(model.inputs[1], torch.ones((1, 4))))
        self.assertEqual(float(losses.unrolled.detach()), 0.0)

    def test_seeded_training_is_deterministic_and_reports_all_losses(self) -> None:
        lineages = (_lineage("a"), _lineage("b"))
        spec = _spec()

        first_model, first = train_short_unroll_predictor(
            spec, lineages, device="cpu"
        )
        second_model, second = train_short_unroll_predictor(
            spec, lineages, device="cpu"
        )

        for name, value in first_model.state_dict().items():
            self.assertTrue(torch.equal(value, second_model.state_dict()[name]))
        self.assertEqual(first.optimizer_examples, 4)
        self.assertEqual(first.optimizer_steps, 2)
        self.assertEqual(first.effective_epochs, 4 / 6)
        self.assertEqual(first.mean_local_loss, second.mean_local_loss)
        self.assertEqual(first.mean_unrolled_loss, second.mean_unrolled_loss)
        self.assertEqual(
            first.mean_carrier_bound_penalty,
            second.mean_carrier_bound_penalty,
        )
        self.assertEqual(first.failures, ())
        self.assertEqual(first.device, "cpu")

    def test_evaluation_reports_absolute_bound_excess_by_recursive_step(self) -> None:
        model = _RecordingAddOne()
        lineage = _lineage("cal", role="calibration")

        result = evaluate_recursive_carrier(
            model,
            (lineage,),
            horizon=15,
            carrier_bound=0.5,
        )

        self.assertEqual(len(result.step_curve), 4)
        self.assertEqual(
            [metric.evaluated_predictions for metric in result.step_curve],
            [1, 1, 1, 1],
        )
        self.assertAlmostEqual(
            result.step_curve[0].mean_absolute_carrier_bound_excess,
            0.5,
        )
        self.assertGreater(
            result.step_curve[-1].mean_absolute_carrier_bound_excess,
            result.step_curve[0].mean_absolute_carrier_bound_excess,
        )
        self.assertIsNotNone(result.error_auc)
        self.assertEqual(result.nonfinite_failures, 0)

    def test_checkpoint_binds_the_full_training_spec_without_hashes(self) -> None:
        spec = _spec()
        model, report = train_short_unroll_predictor(
            spec, (_lineage("a"),), device="cpu"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_short_unroll_checkpoint(path, model, report)

            loaded, loaded_report = load_short_unroll_checkpoint(
                path, spec, device="cpu"
            )
            self.assertEqual(loaded.config, spec.predictor_config)
            self.assertEqual(loaded_report.spec, spec)
            self.assertNotIn("sha256", loaded_report.checkpoint_identity)
            with self.assertRaisesRegex(LineageScalingError, "frozen training cell"):
                load_short_unroll_checkpoint(
                    path,
                    replace(spec, lineage_manifest_reference="another-manifest"),
                    device="cpu",
                )

    def test_public_dry_run_is_no_write_and_reports_foreground_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = issue_67_main(["--dry-run", "--output", str(output)])

            rendered = stdout.getvalue()
            self.assertEqual(status, 0)
            self.assertFalse(output.exists())
            self.assertIn("dry-run validate configuration=1/3", rendered)
            self.assertIn("[train teacher-forced-h15/seed-67]", rendered)
            self.assertIn('"files_written": false', rendered)
            self.assertIn('"final_evaluation_opened": false', rendered)

    def test_prepare_freezes_nine_compact_cells_and_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            arguments = [
                "--prepare",
                "--output",
                str(output),
                "--optimizer-example-budget",
                "16",
                "--batch-size",
                "4",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(issue_67_main(arguments), 0)
                first = (output / "plan.json").read_bytes()
                self.assertEqual(issue_67_main(arguments), 0)

            self.assertEqual((output / "plan.json").read_bytes(), first)
            plan = json.loads(first)
            self.assertEqual(plan["configuration_count"], 9)
            self.assertEqual(
                {item["unroll_steps"] for item in plan["configurations"]},
                {1, 2, 4},
            )
            self.assertNotIn("sha256", plan["identity"])


if __name__ == "__main__":
    unittest.main()
