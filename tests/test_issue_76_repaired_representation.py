from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import run_issue_76_repaired_representation as repaired
from tests.test_native_history_fit import synthetic_shard


class RepairedRepresentationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_metadata_derivation_retains_every_nonimage_target_and_parent(self):
        original = synthetic_shard(5)
        original["plan_identity"] = "parent"
        value = repaired.metadata_shard(original, dict(identity="new"))
        self.assertEqual(value["source_plan_identity"], "parent")
        self.assertEqual(value["plan_identity"], "new")
        self.assertEqual(value["segment_ranges"], original["segment_ranges"])
        self.assertNotIn("images", value["tensors"])
        self.assertIn("images", original["tensors"])
        for key, tensor in value["tensors"].items():
            self.assertTrue(torch.equal(tensor, original["tensors"][key]))

    def test_parser_reuse_preserves_weights_and_following_history_initialization(self):
        torch.manual_seed(123)
        parser = repaired.fit.NativeVisualParser()
        following = repaired.fit.CommonHistoryFit().state_dict()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "visual.pt"
            torch.save(dict(plan_identity="balanced", complete=True, updates_requested=1500,
                updates_applied=1488, updates_skipped=12, model=parser.state_dict()), path)
            plan = dict(parser_plan=dict(identity="balanced"), parser_checkpoints={"123": str(path)})
            torch.manual_seed(123)
            actual = repaired.load_parser(plan, 123, "cpu")
            after = repaired.fit.CommonHistoryFit().state_dict()
            for key, value in following.items():
                torch.testing.assert_close(value, after[key], rtol=0, atol=0)
            for key, value in parser.state_dict().items():
                torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(actual.training)
            self.assertTrue(all(not p.requires_grad for p in actual.parameters()))

    def test_physical_contrast_preserves_unavailable_and_failed_assignments(self):
        common = dict(seed=1, pure=True, member_identity="one", role="model_selection", family="f", available=True)
        old = {**common, "policies": {p: dict(failure=None, **{k: 2. for k in repaired.PHYSICAL_METRICS})
                                     for p in repaired.CONTRAST_POLICIES}}
        new = {**common, "policies": {p: dict(failure=None, **{k: 1. for k in repaired.PHYSICAL_METRICS})
                                     for p in repaired.CONTRAST_POLICIES}}
        new["policies"]["adaptive"] = dict(failure="nonfinite")
        unavailable = {**common, "member_identity": "two", "available": False, "policies": {}}
        rows = repaired.physical_contrasts(dict(original_reference=[old, unavailable]), [new, unavailable])
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["repaired_minus_original"]["adaptive"]["deltas"])
        self.assertEqual(rows[0]["repaired_minus_original"]["fixed-750-continuous"]["deltas"]["block_count_absolute_error"], -1.)
        self.assertEqual(rows[1]["repaired_minus_original"], {})

    def test_synthetic_end_to_end_common_paired_fits_and_diagnostics(self):
        fit = repaired.fit
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source", Path(folder) / "target"
            entries = []
            for role in ("training", "calibration", "model_selection"):
                shard = synthetic_shard(230)
                shard.update(plan_identity="synthetic-parent", member_identity=role, exposure_role=role)
                path = f"prepared/{role}.pt"
                fit.atomic_torch(source / path, shard)
                entries.append(dict(member_identity=role, base_cluster=role, exposure_role=role,
                    path=path, frames=230, usable=True, family="synthetic", gameplay_success=False))
            parent = dict(identity="synthetic-parent", synthetic=True, members=[dict(identity=e["member_identity"]) for e in entries])
            fit.files.write(source / "data-index.json", dict(plan_identity=parent["identity"],
                entries=entries, fit_data_gate_passed=True, coverage={}))
            parser_path = source / "parser.pt"
            fit.atomic_torch(parser_path, dict(plan_identity="synthetic-parser", complete=True,
                updates_requested=1500, updates_applied=1488, updates_skipped=12,
                model=fit.NativeVisualParser().state_dict()))
            plan = {**parent, "identity": "synthetic-new", "parent_plan": parent,
                "source_root": str(source), "collection_root": str(source), "seeds": [123],
                "parser_plan": dict(identity="synthetic-parser"), "parser_checkpoints": {"123": str(parser_path)},
                "new_artifact_bytes": 2**30, "preparation_seconds": 60,
                "capacity": fit.capacity_contract(), "updates": dict(history=1, predictor=3, controller=1),
                "limits": dict(common_seconds_per_seed=60, predictor_seconds_per_arm_seed=60,
                    controller_diagnostic_seconds_per_arm_seed=60, working_bytes=2**30)}
            with patch.object(repaired, "ROOT", target):
                repaired.prepare_data(plan)
                repaired.train_common(plan, "cpu")
                fit.train_predictors(target, plan, "cpu")
                fit.train_controllers(target, plan, "cpu")
                fit.diagnose(target, plan, "cpu")
                report = fit.publish_diagnostic(target, plan, validate=True)
            self.assertEqual(len(report["rows"]), 4)
            self.assertTrue(all(r["available"] for r in report["rows"]))
            self.assertFalse(report["fresh_access_allowed"])
            common = torch.load(fit.common_path(target, 123), weights_only=False)
            self.assertTrue(common["synthetic"])
            for pure in (False, True):
                checkpoint = torch.load(fit.model_path(target, 123, pure, "predictor"), weights_only=False)
                self.assertTrue(checkpoint["complete"])


if __name__ == "__main__":
    unittest.main()
