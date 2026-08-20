import json
import tempfile
import unittest
from pathlib import Path

from world_model.training import grid_artifacts as artifacts


def _metric(delta: int, error: float) -> artifacts.PairMetricArtifact:
    return artifacts.PairMetricArtifact(
        delta=delta,
        alpha="continuous",
        requested_delta=delta,
        effective_delta=delta,
        weighted_error=error,
        compute_cost=1.0 / delta,
        availability="available",
    )


def _state(index: int) -> artifacts.BestPairState:
    return artifacts.BestPairState(
        state_index=index,
        temporal_ceiling=15,
        metrics=(_metric(1, 1.0), _metric(5, 0.5), _metric(15, 0.25)),
        pareto_points=((1, 1.0, 1.0), (5, 0.5, 0.2)),
        selected_delta=15,
        selected_alpha="continuous",
    )


def _manifest(count: int = 4096) -> artifacts.SweepManifest:
    return artifacts.SweepManifest(
        source_identity="best-pair-source-v1:fixture",
        checkpoint_path="checkpoint.pt",
        catalog_identity="episode-catalog-v1:dev:legacy_rgb_v1:1:collector_v1",
        grid_identity="pair-grid-v1:continuous",
        score_identity="pair-score-spec-v1:fixture",
        partition_identity="partition-v1:fixture",
        state_count=count,
        shard_size=4096,
    )


class GridArtifactTests(unittest.TestCase):
    def test_canonical_json_is_sorted_ascii_and_newline_terminated(self) -> None:
        encoded = artifacts.canonical_json_bytes({"z": "é", "a": 1})
        self.assertEqual(encoded, b'{"a":1,"z":"\\u00e9"}\n')

    def test_two_shards_round_trip_and_resume_are_byte_identical(self) -> None:
        states = tuple(_state(i) for i in range(8192))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            first = artifacts.write_best_pair_artifacts(root, _manifest(8192), states)
            first_bytes = {path.name: path.read_bytes() for path in root.iterdir()}
            self.assertEqual(len(first.shards), 2)
            second = artifacts.write_best_pair_artifacts(root, _manifest(8192), states, resume=True)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, {path.name: path.read_bytes() for path in root.iterdir()})
            self.assertEqual(artifacts.validate_best_pair_artifacts(root), second)

    def test_changed_source_identity_and_order_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            artifacts.write_best_pair_artifacts(root, _manifest(), tuple(_state(i) for i in range(4096)))
            payload = json.loads((root / "manifest.json").read_text())
            payload["source_identity"] = "best-pair-source-v1:changed"
            (root / "manifest.json").write_bytes(artifacts.canonical_json_bytes(payload))
            with self.assertRaises(artifacts.ArtifactValidationError):
                artifacts.validate_best_pair_artifacts(root, _manifest())

    def test_alpha_exclusions_are_explicit(self) -> None:
        self.assertEqual(
            artifacts.ALPHA_EXCLUSIONS,
            ({"alpha": "micro", "reason": "symbolic_supervision_unavailable"},
             {"alpha": "macro", "reason": "symbolic_supervision_unavailable"}),
        )

    def test_nonfinite_and_protected_output_are_rejected(self) -> None:
        with self.assertRaises(artifacts.ArtifactContractError):
            artifacts.PairMetricArtifact(1, "continuous", 1, 1, float("nan"), 1.0, "available")
        with tempfile.TemporaryDirectory() as directory:
            protected = Path(directory) / "frames" / "run"
            with self.assertRaises(artifacts.ArtifactContractError):
                artifacts.write_best_pair_artifacts(protected, _manifest(), tuple(_state(i) for i in range(4096)))


if __name__ == "__main__":
    unittest.main()
