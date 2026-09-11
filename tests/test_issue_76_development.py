from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import issue_76_censored_episode as episode
from scripts import run_issue_76_development as run
from scripts import issue_76_development_report as publication


class DevelopmentCollectionTests(unittest.TestCase):
    def test_exact_role_and_family_counts_and_seed_disjointness(self):
        assigned = run.assignments("development")
        self.assertEqual(len(assigned), 350)
        self.assertEqual(Counter(row[3] for row in assigned), {"training": 250, "calibration": 50, "model_selection": 50})
        self.assertEqual(Counter(row[0] for row in assigned), {family: 70 for family in run.FAMILIES})
        self.assertEqual(len({row[1] for row in assigned}), 350)
        self.assertFalse({r[1] for r in assigned} & {r[1] for r in run.assignments("smoke")})
        self.assertEqual(sum(r[4] for r in run.assignments("smoke")), 3)

    def test_censored_segment_stops_episode_and_cannot_become_a_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "player").mkdir()
            member = {"identity": "fixture", "base_cluster": "fixture", "exposure_role": "training",
                      "engine_seed": 1, "template": {}, "scenario": {}, "generated_slots": [], "actions": [{}, {}, {}]}
            bridge = Mock()
            bridge.get_current_level.return_value = 1
            bridge.get_game_state.side_effect = [SimpleNamespace(name="PLAYING"), SimpleNamespace(name="WON")]
            segment = {"summary": {"censored": True, "first_fixed_step": 0, "last_fixed_step": 30000,
                                   "sample_count": 30001, "physical_seconds": 12},
                       "frame_count": 601, "initial_entity_ids": []}
            capture = episode.old.capture
            with patch.dict(episode.os.environ), patch.object(episode.files, "install_level"), \
                    patch.object(episode.files, "materialize", return_value=(None, SimpleNamespace(to_dict=lambda: {}))), \
                    patch.object(capture, "start_display", return_value=(":fixture", None)), \
                    patch.object(capture, "free_port", side_effect=[1, 2, 3]), \
                    patch.object(capture, "start_engine", return_value=SimpleNamespace(novphy_log_file=SimpleNamespace(name="fixture.log"))), \
                    patch.object(capture, "connect_with_retry", return_value=bridge), \
                    patch.object(capture, "prepare_for_play"), patch.object(capture, "stop_started_engine"), \
                    patch.object(episode, "capture_segment", return_value=segment) as shoot:
                episode.capture_episode(root, member, {"shot_seconds": 180})
            result = episode.files.read(root / "results/fixture.json")
            self.assertTrue(result["complete"])
            self.assertFalse(result["gameplay_success"])
            self.assertEqual(result["gameplay_failure_penalty"], 1)
            self.assertEqual(result["stopping_reason"], "native_time_window_limit")
            self.assertEqual(result["attempted_shots"], [1])
            self.assertEqual(result["unattempted_shots"], [2, 3])
            self.assertEqual(shoot.call_count, 1)

    def test_resume_never_replays_completed_or_interrupted_assignments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members = [{"identity": n, "actions": [{}, {}]} for n in ("done", "interrupted")]
            episode.files.write(run.c2.result_path(root, members[0]), {"complete": True})
            (root / "attempts/interrupted").mkdir(parents=True)
            episode.files.write(root / "budget.json", {"stopped": False, "active_seconds": 0})
            with patch.object(run.multiprocessing, "get_context", side_effect=AssertionError("unexpected replay")):
                run.run(root, {"limits": {}, "members": members})
            value = episode.files.read(run.c2.result_path(root, members[1]))
            self.assertFalse(value["complete"])
            self.assertEqual(value["failure"], "interrupted_attempt_no_retry")

    def test_progress_publication_preserves_the_frozen_report_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members = [{"identity": name} for name in ("won", "failed", "unattempted")]
            for member, complete, won in ((members[0], True, True), (members[1], False, False)):
                episode.files.write(run.c2.result_path(root, member),
                                    {"complete": complete, "gameplay_success": won, "segments": []})
            episode.files.write(root / "budget.json", {"active_seconds": 1})
            plan = {"identity": "fixture", "members": members}
            self.assertEqual(publication.report(root, plan), run.report(root, plan))


if __name__ == "__main__":
    unittest.main()
