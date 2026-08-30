from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from world_model.data import CohortV2Rollout
from world_model.training import (
    CohortV2ControllerConfig,
    CohortV2ControllerFeatureCodec,
    CohortV2ExecutionProfile,
    CohortV2ExhaustiveEvaluator,
    CohortV2TrajectoryCostSpec,
    build_cohort_v2_controller_examples,
    evaluate_cohort_v2_controllers,
    generate_cohort_v2_trajectory_labels,
    measure_cohort_v2_evaluation,
    train_cohort_v2_controllers,
    validate_cohort_v2_controllers,
    write_cohort_v2_controllers,
)
from world_model.training.cohort_v2_controller import _mean
from world_model.training.grid_artifacts import canonical_json_bytes

from tests.test_cohort_v2_policy_baselines import _BaselineScorer, _calibration
from tests.test_cohort_v2_exhaustive_evaluation import _reader


def _png(red: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (20, 12), (red, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _controller_reader(role: str, red: int):
    source = _reader(role)
    original = source.rollouts[0]
    rollout = CohortV2Rollout(
        attempt_id=original.attempt_id,
        exposure_role=original.exposure_role,
        coverage_stratum=original.coverage_stratum,
        scenario_lineage_identity=original.scenario_lineage_identity,
        intervention={
            "interface_action": {
                "drag_release": (10, -5),
                "frame_height": 100,
                "releaseTime": 1000,
                "tapTime": 0,
            }
        },
        agent_observation_identity=original.agent_observation_identity,
        agent_observation_fixed_step=original.agent_observation_fixed_step,
        frame_records=original.frame_records,
    )

    class Reader:
        release_identity = source.release_identity
        capability_declaration_identity = source.capability_declaration_identity
        partition_identity = source.partition_identity
        derivation_identity = "derivations:fixture"
        rollouts = (rollout,)

        @staticmethod
        def load_observation(item, *, observation_role: str) -> bytes:
            assert item is rollout
            assert observation_role == "agent"
            return _png(red)

    return Reader()


def _inputs():
    readers = tuple(
        _controller_reader(role, red)
        for role, red in zip(
            ("training", "calibration", "model_selection"),
            (40, 80, 120),
            strict=True,
        )
    )
    evaluation = CohortV2ExhaustiveEvaluator(_BaselineScorer()).evaluate(readers)
    measurement = measure_cohort_v2_evaluation(
        evaluation,
        readers,
        _calibration(),
        CohortV2ExecutionProfile(
            controller_executed=False,
            shared_perception_executed=False,
        ),
    )
    spec = CohortV2TrajectoryCostSpec(0.0, 0.0, 1.0)
    labels = generate_cohort_v2_trajectory_labels(evaluation, measurement, spec)
    return readers, evaluation, measurement, labels, spec


def _validation_kwargs() -> dict[str, str]:
    return {
        "trajectory_label_artifact_identity": "labels:fixture",
        "baseline_artifact_identity": "baselines:fixture",
        "derivation_index_identity": "derivations:fixture",
        "implementation_revision": "implementation:fixture",
    }


def _write_fixture(root: Path):
    readers, evaluation, measurement, labels, spec = _inputs()
    config = CohortV2ControllerConfig(epochs=2, batch_size=4, hidden_dim=8)
    receipt = write_cohort_v2_controllers(
        root,
        readers,
        evaluation,
        measurement,
        labels,
        spec,
        config,
        **_validation_kwargs(),
    )
    return receipt, readers, evaluation, measurement, labels, spec


def _artifact_digests(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("controller_decisions.jsonl", "scores.json", "manifest.json")
    }


class CohortV2ControllerTests(unittest.TestCase):
    def test_float_mean_is_stable_across_python_sum_implementations(self) -> None:
        self.assertEqual(_mean((1e16, 1.0, -1e16)), 0.0)

    def test_features_use_agent_observation_intervention_and_elapsed_position(self):
        config = CohortV2ControllerConfig(epochs=1)
        codec = CohortV2ControllerFeatureCodec(config)
        intervention = {
            "interface_action": {
                "drag_release": (10, -5),
                "frame_height": 100,
                "releaseTime": 1000,
                "tapTime": 0,
            }
        }

        first = codec.encode(_png(40), elapsed_fixed_steps=0, intervention=intervention)
        later = codec.encode(_png(40), elapsed_fixed_steps=5, intervention=intervention)

        self.assertEqual(first.shape, (config.feature_dim,))
        self.assertFalse(first.equal(later))

    def test_joint_and_two_head_controllers_share_inputs_capacity_and_held_out_scope(self):
        readers, evaluation, measurement, labels, spec = _inputs()
        config = CohortV2ControllerConfig(epochs=2, batch_size=4, hidden_dim=8)
        examples = build_cohort_v2_controller_examples(readers, labels, config)
        models = train_cohort_v2_controllers(examples, evaluation.grid.pairs, config)
        result = evaluate_cohort_v2_controllers(
            models, examples, evaluation, measurement, spec
        )

        counts = tuple(sum(parameter.numel() for parameter in model.parameters()) for model in models)
        self.assertEqual(counts[0], counts[1])
        self.assertEqual({score.exposure_role for score in result.scores}, {"model_selection"})
        self.assertEqual({score.controller_id for score in result.scores}, {"joint_pair", "matched_capacity_two_head"})
        self.assertEqual({score.state_count for score in result.scores}, {3})
        self.assertTrue(all(score.utility_available_count == 3 for score in result.scores))

    def test_preconfirmatory_analysis_can_include_calibration_without_training_on_it(self):
        readers, evaluation, measurement, labels, spec = _inputs()
        config = CohortV2ControllerConfig(epochs=2, batch_size=4, hidden_dim=8)
        examples = build_cohort_v2_controller_examples(
            readers,
            labels,
            config,
            included_roles=("training", "calibration", "model_selection"),
        )
        models = train_cohort_v2_controllers(examples, evaluation.grid.pairs, config)
        result = evaluate_cohort_v2_controllers(
            models,
            examples,
            evaluation,
            measurement,
            spec,
            evaluation_roles=("calibration", "model_selection"),
        )

        self.assertEqual(
            {score.exposure_role for score in result.scores},
            {"calibration", "model_selection"},
        )
        self.assertEqual({score.state_count for score in result.scores}, {3})

    def test_artifacts_reload_models_and_recompute_held_out_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            written, readers, evaluation, measurement, labels, spec = _write_fixture(
                root
            )
            validated = validate_cohort_v2_controllers(
                root,
                readers,
                evaluation,
                measurement,
                labels,
                spec,
                **_validation_kwargs(),
            )
            manifest = json.loads((root / "manifest.json").read_bytes())

            self.assertEqual(written, validated)
            self.assertFalse(manifest["oracle_engine_state_is_controller_input"])
            self.assertFalse(manifest["final_evaluation_consumed"])
            self.assertEqual(manifest["matched_parameter_count"], written.parameter_count)

            scores = root / "scores.json"
            scores.write_bytes(scores.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                ValueError,
                (
                    r"stored_artifact_identities.*scores_identity.*python_version=.*"
                    r"threads=.*deterministic_algorithms=.*checkpoint\.pt\(size="
                ),
            ):
                validate_cohort_v2_controllers(
                    root,
                    readers,
                    evaluation,
                    measurement,
                    labels,
                    spec,
                    **_validation_kwargs(),
                )

    def test_validation_reports_noncanonical_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, readers, evaluation, measurement, labels, spec = _write_fixture(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="ascii")

            with self.assertRaisesRegex(
                ValueError, "canonical_manifest.*field=canonical_json_encoding"
            ):
                validate_cohort_v2_controllers(
                    root, readers, evaluation, measurement, labels, spec,
                    **_validation_kwargs(),
                )

    def test_validation_reports_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, readers, evaluation, measurement, labels, spec = _write_fixture(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["checkpoint_identity"] = "checkpoint:changed"
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(
                ValueError, "stored_artifact_identities.*checkpoint_identity"
            ):
                validate_cohort_v2_controllers(
                    root, readers, evaluation, measurement, labels, spec,
                    **_validation_kwargs(),
                )

    def test_validation_reports_first_recomputed_decision_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, readers, evaluation, measurement, labels, spec = _write_fixture(root)
            decisions_path = root / "controller_decisions.jsonl"
            records = [json.loads(line) for line in decisions_path.read_bytes().splitlines()]
            records[0]["selected_pair"]["requested_horizon"] = 99
            decisions = b"".join(canonical_json_bytes(record) for record in records)
            decisions_path.write_bytes(decisions)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["decisions_identity"] = f"sha256:{hashlib.sha256(decisions).hexdigest()}"
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(
                ValueError,
                r"recomputed_decisions.*record=0.*field=\$\.selected_pair\.requested_horizon",
            ):
                validate_cohort_v2_controllers(
                    root, readers, evaluation, measurement, labels, spec,
                    **_validation_kwargs(),
                )

    def test_validation_reports_first_recomputed_score_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, readers, evaluation, measurement, labels, spec = _write_fixture(root)
            scores_path = root / "scores.json"
            scores = json.loads(scores_path.read_bytes())
            scores["scores"][0]["pair_accuracy"] = 0.125
            score_bytes = canonical_json_bytes(scores)
            scores_path.write_bytes(score_bytes)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["scores_identity"] = f"sha256:{hashlib.sha256(score_bytes).hexdigest()}"
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(
                ValueError, r"recomputed_scores.*field=\$\.scores\[0\]\.pair_accuracy"
            ):
                validate_cohort_v2_controllers(
                    root, readers, evaluation, measurement, labels, spec,
                    **_validation_kwargs(),
                )

    def test_validation_reports_first_source_provenance_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, readers, evaluation, measurement, labels, spec = _write_fixture(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["baseline_artifact_identity"] = "baselines:changed"
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(
                ValueError,
                r"recomputed_manifest_provenance.*field=\$\.baseline_artifact_identity",
            ):
                validate_cohort_v2_controllers(
                    root, readers, evaluation, measurement, labels, spec,
                    **_validation_kwargs(),
                )

    def test_subprocess_controller_results_are_byte_identical_across_environment_matrix(self):
        worker_output = os.environ.get("NOVPHY_CONTROLLER_DETERMINISM_OUTPUT")
        if worker_output is not None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_fixture(root)
                Path(worker_output).write_bytes(canonical_json_bytes({
                    "artifacts": _artifact_digests(root),
                    "torch_threads": torch.get_num_threads(),
                }))
            return

        results = []
        with tempfile.TemporaryDirectory() as directory:
            for threads, hash_seed in ((1, "0"), (1, "17"), (4, "0"), (4, "17")):
                output = Path(directory) / f"{threads}-{hash_seed}.json"
                environment = os.environ.copy()
                environment.update({
                    "MKL_NUM_THREADS": str(threads),
                    "NOVPHY_CONTROLLER_DETERMINISM_OUTPUT": str(output),
                    "OMP_NUM_THREADS": str(threads),
                    "PYTHONHASHSEED": hash_seed,
                })
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "unittest",
                        (
                            "tests.test_cohort_v2_controller.CohortV2ControllerTests."
                            "test_subprocess_controller_results_are_byte_identical_across_environment_matrix"
                        ),
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                result = json.loads(output.read_bytes())
                self.assertEqual(result["torch_threads"], threads)
                results.append(result["artifacts"])

        self.assertTrue(all(result == results[0] for result in results[1:]))


if __name__ == "__main__":
    unittest.main()
