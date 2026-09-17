from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_issue_76_event_development as runner


class FakeProcess:
    """Control-flow fixture only; no engine or scientific capture validation."""
    started_rows = []
    fail_identity = None

    def __init__(self, *, target, args):
        self.args = args
        self.pid = None
        self.exitcode = None
        self.checks = 0

    def start(self):
        root, source, row, limits, slot = self.args
        self.pid = 100000 + len(self.started_rows)
        self.started_rows.append(row["identity"])
        attempt = root / "attempts" / row["identity"]
        frame = attempt / "aligned/capture-v2:fixture/frame_000001.json"
        frame.parent.mkdir(parents=True)
        frame.write_text("{}")
        failed = row["identity"] == self.fail_identity
        runner.files.write(root / "results" / (row["identity"] + ".json"),
                           {"member_identity": row["identity"], "complete": not failed,
                            "failure": "fixture_failure" if failed else None})

    def is_alive(self):
        self.checks += 1
        if self.checks > 2:
            self.exitcode = 0 if self.exitcode is None else self.exitcode
        return self.exitcode is None

    def join(self, timeout=None):
        pass


class EventSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root, self.output = Path(self.directory.name) / "root", Path(self.directory.name) / "public"
        (self.root / "player").mkdir(parents=True)
        (self.root / "player/9001.x86_64").write_bytes(b"fixture")
        self.output.mkdir()
        action = {"drag_x": -80, "drag_y": 10, "release_time_ms": 600, "tap_time_ms": 0}
        rows = [{"identity": f"case-{i}", "ordinal": i + 1, "source_member_identity": "source",
                 "candidate_ordinal": i, "original_action_reference": i == 0, "action": action}
                for i in range(10)]
        limits = {"workers": 8, "collection_wall_seconds": 10000, "smoke_wall_seconds": 1000,
                  "attempt_seconds": 420, "shot_seconds": 180, "aggregate_cpu_rss_mib": 32768,
                  "worker_cpu_rss_mib": 4096, "artifact_bytes": 2**30,
                  "minimum_free_bytes_before_attempt": 0, "rgb_frames_per_shot_max": 601}
        self.plan = {"identity": "fixture-execution", "readiness_action": action,
                     "inventory": {"limits": limits, "assignments": rows,
                                   "smoke_member_identities": [row["identity"] for row in rows],
                                   "source_members": [{"identity": "source", "exposure_role": "calibration",
                                                       "base_cluster": "base", "engine_seed": 2}]}}
        FakeProcess.started_rows = []
        FakeProcess.fail_identity = None

    def run_smoke(self):
        def terminate(process):
            if process.exitcode is None:
                process.exitcode = -15
        with patch.object(runner.multiprocessing, "get_context", return_value=SimpleNamespace(Process=FakeProcess)), patch.object(
                runner, "process_rss", return_value=1.), patch.object(runner.resources, "check_ports_available"), patch.object(
                runner.time, "sleep"), patch.object(runner, "terminate_worker", side_effect=terminate):
            return runner.run(self.plan, self.root, self.output, smoke=True)

    def test_eight_worker_waves_seal_all_assignments_and_clean_resume_never_repeats(self):
        budget = self.run_smoke()
        self.assertFalse(budget["running"])
        self.assertFalse(budget["stopped"])
        self.assertEqual(len(budget["completed_attempt_bytes"]), 10)
        self.assertEqual(budget["maximum_overlapping_native_workers"], 8)
        self.assertEqual(len(FakeProcess.started_rows), 10)
        again = self.run_smoke()
        self.assertEqual(len(FakeProcess.started_rows), 10)
        self.assertEqual(again["completed_attempt_bytes"], budget["completed_attempt_bytes"])

    def test_smoke_failure_retains_wave_and_stops_unattempted_members(self):
        FakeProcess.fail_identity = "case-1"
        budget = self.run_smoke()
        self.assertTrue(budget["stopped"])
        self.assertEqual(budget["stop_reason"], "smoke_capture_failure")
        self.assertEqual(FakeProcess.started_rows, [f"case-{i}" for i in range(8)])
        self.assertEqual(set(budget["completed_attempt_bytes"]), set(FakeProcess.started_rows))
        self.assertEqual(budget["unattempted_member_identities"], ["case-8", "case-9"])
        failed = runner.files.read(self.root / "results/case-1.json")
        self.assertEqual(failed["failure"], "fixture_failure")
        with self.assertRaises(ValueError):
            self.run_smoke()

    def test_unsealed_attempt_is_not_replayed(self):
        (self.root / "attempts/case-0").mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.run_smoke()
        self.assertEqual(FakeProcess.started_rows, [])

    def test_disk_reserve_stop_records_every_assignment_as_unattempted(self):
        self.plan["inventory"]["limits"]["minimum_free_bytes_before_attempt"] = 1
        with patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            budget = self.run_smoke()
        self.assertEqual(budget["stop_reason"], "disk_reserve_limit")
        self.assertTrue(budget["stopped"])
        self.assertEqual(len(budget["unattempted_member_identities"]), 10)
        self.assertEqual(FakeProcess.started_rows, [])

    def test_execution_freeze_rejects_source_and_inventory_changes(self):
        source = self.root / "scripts/source.py"
        source.parent.mkdir()
        source.write_text("original")
        plan = {**self.plan, "identity": "issue-76-event-development-execution-v1",
                "capture_execution_authorized": True, "model_training_authorized": False,
                "new_optimizer_updates": 0, "fresh_access": False,
                "source_text": {"scripts/source.py": "original"}}
        runner.files.write(self.root / "inventory.json", plan["inventory"])
        runner.files.write(self.root / "execution-plan.json", plan)
        with patch.object(runner.files, "ROOT", self.root):
            self.assertEqual(runner.load_plan(self.root), plan)
            source.write_text("changed")
            with self.assertRaises(ValueError):
                runner.load_plan(self.root)
            source.write_text("original")
            runner.files.write(self.root / "inventory.json", {})
            with self.assertRaises(ValueError):
                runner.load_plan(self.root)

    def test_prepare_reads_serialized_source_plan_path(self):
        source_plan = self.root / "source-plan.json"
        runner.files.write(source_plan, {"members": self.plan["inventory"]["source_members"]})
        inventory = {**self.plan["inventory"], "source_text": {},
                     "source_collection_plan_path": str(source_plan),
                     "player_source": str(self.root / "player")}
        inventory["source_members"][0]["actions"] = [self.plan["readiness_action"]]
        runner.files.write(self.root / "inventory.json", inventory)
        with patch.object(runner.metadata, "select_sources", return_value=inventory["source_members"]), patch.object(
                runner.metadata, "assignments", return_value=inventory["assignments"]), patch.object(runner, "SOURCES", ()):
            runner.prepare(self.root, self.output)
        self.assertTrue((self.root / "execution-plan.json").is_file())

    def test_full_run_requires_exact_smoke_integrity_and_concurrency(self):
        expected = self.plan["inventory"]["smoke_member_identities"]
        report = {"validated": True, "individual_integrity_passed": True,
                  "execution_plan_identity": self.plan["identity"], "member_identities": expected,
                  "maximum_overlapping_native_workers": 8}
        for change in ({"validated": False}, {"individual_integrity_passed": False},
                       {"member_identities": expected[:-1]}, {"maximum_overlapping_native_workers": 7}):
            runner.files.write(self.root / "smoke-validation.json", {**report, **change})
            with self.assertRaises(ValueError):
                runner.selected_assignments(self.plan, self.root, False)
        runner.files.write(self.root / "smoke-validation.json", report)
        with self.assertRaises(ValueError):
            runner.run(self.plan, self.root, self.output, smoke=False)
