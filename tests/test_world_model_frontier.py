import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.grid_data import MotionRegime
from world_model.training.scoring import (
    ExhaustiveScorer,
    Partition,
    ScoringExample,
    write_score_artifacts,
)
from world_model.training.frontier import FrontierError, analyze_frontier, pareto_frontier


class ZeroPredictor:
    def latent_mse(self, examples, requested_delta, effective_delta):
        return tuple(0.0 for _example in examples)


def frontier_rows(per_regime=100, include_selection=True):
    rows = []
    for regime in ("active", "calm"):
        for state_index in range(per_regime):
            selected_delta = 1 if state_index % 2 == 0 else 5
            errors = {1: 0.01 if selected_delta == 1 else 0.1, 5: 0.01 if selected_delta == 5 else 0.1, 15: 0.02}
            for delta in (1, 5, 15):
                row = {
                    "state_id": f"{regime}-{state_index:03d}",
                    "regime": regime,
                    "delta": delta,
                    "weighted_prediction_error": errors[delta],
                    "compute_cost": 1.0 / delta,
                    "error_scale": 0.01,
                }
                if include_selection:
                    row["selected_delta"] = selected_delta
                rows.append(row)
    return rows


def write_frontier_input(root):
    examples = tuple(
        ScoringExample(
            state_id=f"{partition}-{regime}-{index:03d}",
            partition=partition,
            motion_regime=regime,
            frame_count=16,
            context_position=0,
        )
        for partition in Partition
        for regime in (MotionRegime.QUIESCENT, MotionRegime.TRANSITIONAL)
        for index in range(100)
    )
    score_root = root / "scores"
    receipt = write_score_artifacts(
        score_root,
        ExhaustiveScorer(ZeroPredictor()).score(examples),
        checkpoint_digest="a" * 64,
        shard_size=128,
    )
    manifest = json.loads((score_root / "manifest.json").read_text(encoding="ascii"))
    payload = {
        "checkpoint_digest": manifest["checkpoint_digest"],
        "partition": "evaluation",
        "schema_version": "temporal_frontier_input_v1",
        "score_artifact_root": "scores",
        "score_manifest_digest": receipt.manifest_digest,
        "score_spec_digest": manifest["score_spec_digest"],
        "state_digest": manifest["state_digest"],
    }
    path = root / "frontier-input.json"
    path.write_bytes(canonical_json_bytes(payload))
    return path, payload, score_root


class FrontierTests(unittest.TestCase):
    def test_dominance_and_ties(self):
        self.assertEqual(pareto_frontier([
            {"delta": 1, "weighted_prediction_error": 1.0, "compute_cost": 1.0},
            {"delta": 5, "weighted_prediction_error": 2.0, "compute_cost": 2.0},
        ]), [1])
        self.assertEqual(pareto_frontier([
            {"delta": 1, "weighted_prediction_error": 1.0, "compute_cost": 1.0},
            {"delta": 5, "weighted_prediction_error": 1.0, "compute_cost": 1.0},
        ]), [1, 5])

    def test_bootstrap_is_deterministic_and_requires_states(self):
        rows = frontier_rows()
        self.assertEqual(analyze_frontier(rows, seed=4), analyze_frontier(rows, seed=4))
        self.assertEqual(analyze_frontier(rows, seed=4)["bootstrap"]["replicates"], 1000)
        with self.assertRaises(FrontierError):
            analyze_frontier(frontier_rows(per_regime=99))

    def test_nonfinite_metric_is_rejected(self):
        rows = frontier_rows()
        rows[0]["weighted_prediction_error"] = float("nan")
        with self.assertRaises(FrontierError):
            analyze_frontier(rows)

    def test_oracle_labels_honor_selected_delta_and_recompute_with_error_scale(self):
        selected = analyze_frontier(frontier_rows(), seed=4)
        recomputed = analyze_frontier(frontier_rows(include_selection=False), seed=4)

        for result in (selected, recomputed):
            self.assertEqual(len(result["oracle_labels"]), 200)
            self.assertEqual(result["oracle_labels"].count(1), 100)
            self.assertEqual(result["oracle_labels"].count(5), 100)
            self.assertNotIn(15, result["oracle_labels"])

    def test_rejects_inconsistent_selected_delta_duplicate_pair_and_inadequate_regime(self):
        inconsistent = frontier_rows()
        inconsistent[1]["selected_delta"] = 15
        duplicate = frontier_rows()
        duplicate.append(dict(duplicate[0]))

        for rows in (inconsistent, duplicate, frontier_rows(per_regime=50)):
            with self.subTest(rows=len(rows)):
                with self.assertRaises(FrontierError):
                    analyze_frontier(rows)

    def test_cli_validates_canonical_score_artifacts_and_binds_every_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, payload, score_root = write_frontier_input(root)
            output_path = root / "out"
            source_bytes = source_path.read_bytes()
            completed = subprocess.run(
                [sys.executable, "scripts/plot_jepa_pair_frontier.py", "--input", str(source_path), "--output-dir", str(output_path)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            digest = hashlib.sha256(source_bytes).hexdigest()
            for name in ("frontier.json", "frontier.md", "frontier.svg", "frontier.pdf"):
                artifact = (output_path / name).read_bytes()
                self.assertIn(digest.encode(), artifact, name)
                self.assertIn(b"alpha", artifact.lower(), name)
                self.assertIn(b"physical", artifact.lower(), name)
            result = json.loads((output_path / "frontier.json").read_text(encoding="ascii"))
            self.assertEqual(result["verdict"], "not_supported")
            reasons = {item["metric"]: item["reason"] for item in result["unavailable_metrics"]}
            self.assertEqual(reasons["micro"], "symbolic_supervision_unavailable")
            self.assertEqual(reasons["macro"], "symbolic_supervision_unavailable")

            arbitrary_path = root / "arbitrary.json"
            arbitrary_path.write_bytes(canonical_json_bytes({"states": []}))
            arbitrary_result = subprocess.run(
                [sys.executable, "scripts/plot_jepa_pair_frontier.py", "--input", str(arbitrary_path), "--output-dir", str(root / "arbitrary")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(arbitrary_result.returncode, 2)

            stale = dict(payload)
            stale["checkpoint_digest"] = "b" * 64
            source_path.write_bytes(canonical_json_bytes(stale))
            stale_result = subprocess.run(
                [sys.executable, "scripts/plot_jepa_pair_frontier.py", "--input", str(source_path), "--output-dir", str(root / "stale")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(stale_result.returncode, 2)

            source_path.write_bytes(canonical_json_bytes(payload))
            shard = next((score_root / "label_shards" / "evaluation").glob("*.jsonl"))
            shard.write_bytes(shard.read_bytes() + b"\n")
            tampered_result = subprocess.run(
                [sys.executable, "scripts/plot_jepa_pair_frontier.py", "--input", str(source_path), "--output-dir", str(root / "tampered")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(tampered_result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
