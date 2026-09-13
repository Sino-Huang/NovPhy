from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from scripts import run_issue_76_parallel_prepare as parallel


class ParallelPreparationTests(unittest.TestCase):
    def test_success_and_failure_shards_and_final_index_match_original_routine(self):
        fit = parallel.fit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequential, workers, collection = root / "sequential", root / "workers", root / "collection"
            members = [dict(identity=f"member-{i}", base_cluster=f"member-{i}",
                            exposure_role="training", generator_family="family") for i in range(2)]
            plan = dict(identity="fixture-plan", members=members, minimum_usable_fraction_per_role_family=.9)
            for member in members:
                fit.files.write(fit.collection.c2.result_path(collection, member),
                                dict(member_identity=member["identity"], complete=True, gameplay_success=False))

            def derive(collection_root, member, result):
                if member["identity"] == "member-1":
                    raise ValueError("fixture invalid source")
                return dict(member_identity=member["identity"], source_result=result,
                            references=[{"step": 0}, {"step": 50}],
                            tensors={"images": torch.arange(8).reshape(2, 4)}, segment_ranges=[])

            with patch.object(fit, "prepare_episode", side_effect=derive), patch.object(fit, "check_disk", return_value=0):
                fit._prepare_data(sequential, plan, collection, Mock())
                for member in members:
                    parallel.prepare_one(workers, collection, plan["identity"], member)
                fit._prepare_data(workers, plan, collection, Mock())
            self.assertEqual(fit.files.read(sequential / "data-index.json"), fit.files.read(workers / "data-index.json"))
            self.assertFalse(fit.files.read(workers / "data-index.json")["fit_data_gate_passed"])
            for member in members:
                a = torch.load(sequential / "prepared" / (member["identity"] + ".pt"), weights_only=False)
                b = torch.load(workers / "prepared" / (member["identity"] + ".pt"), weights_only=False)
                torch.testing.assert_close(a.pop("tensors"), b.pop("tensors"), rtol=0, atol=0)
                self.assertEqual(a, b)

    def test_worker_never_overwrites_a_completed_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "prepared/member.pt"
            target.parent.mkdir()
            target.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "existing shard"):
                parallel.prepare_one(root, root, "plan", {"identity": "member"})
            self.assertEqual(target.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
