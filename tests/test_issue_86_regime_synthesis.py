"""Focused tests for issue-86 cross-family regime-modulation synthesis (fast, CPU, synthetic)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from scripts import run_cross_family_regime_synthesis as runner
from world_model.training import cross_family_regime_synthesis as regime

ROOT = Path(__file__).resolve().parents[1]


def clevrer_shard(collision_frames, clip=None):
    frames = 128
    if clip is None:
        clip = torch.zeros((frames, regime.CARRIER_DIM))
    windows = {"z": torch.stack([clip[start:start + 61] for start in range(40)]),
               "action": torch.zeros((40, 5)), "length": torch.full((40,), 60, dtype=torch.long),
               "relations": torch.zeros((40, 61, 18, 18, 2)),
               "relations_mask": torch.zeros((40, 61, 18, 18, 2), dtype=torch.bool)}
    return {"schema": "issue_79_clevrer_scene_v1", "scene_index": 10000,
            "windows": windows, "clip": clip,
            "collision_frames": torch.tensor(collision_frames, dtype=torch.long)}


def moving_clip():
    """128-frame clip whose active slot decays from speed 0.5 to 0.05 within 61 frames."""
    clip = torch.zeros((128, regime.CARRIER_DIM))
    for frame in range(128):
        carrier = torch.zeros(regime.CARRIER_DIM)
        carrier[2 + 0 * 13 + 0] = 1.0                      # slot 0 present
        speed = max(0.5 - 0.0075 * frame, 0.05)
        carrier[2 + 0 * 13 + 8], carrier[2 + 0 * 13 + 9] = speed, 0.0
        clip[frame] = carrier
    return clip


def novphy_target(state="issue-77-n1-007", *, with_events=((5167, 8148),), moving=True):
    """A #77-shaped target document with two candidate branches."""

    def carriers(motion):
        values = {}
        for offset in (0, 60, 600):
            carrier = [0.0] * regime.CARRIER_DIM
            for slot in range(18):
                base = 2 + slot * 13
                carrier[base + regime.PRESENCE_COLUMN] = 1.0
                if slot == 0 and motion:
                    carrier[base + 8], carrier[base + 9] = motion[offset], 0.0
            values[str(offset)] = carrier
        return values

    fast = {0: 0.0, 60: 0.02, 600: 0.0} if moving else {0: 0.0, 60: 0.0, 600: 0.0}
    document = {"state": state, "plan_identity": "issue-77-n1-diagnostic-v1",
                "dropped_candidates": [], "candidates": []}
    for ordinal in range(13):
        events = {"collision": list(with_events[ordinal % len(with_events)]),
                  "bird_launched": [2502], "entity_death": [], "entity_destroyed": [],
                  "stable_entered": [16639], "stable_exited": [2602], "level_clear": []}
        document["candidates"].append({
            "source": {"ordinal": ordinal}, "status": "available",
            "carriers": carriers(fast), "absorbed_offsets": [585, 595, 599, 600],
            "timing": {"events": events, "censored": True, "last_offset": 333}})
    return document


class StatisticFormulaTests(unittest.TestCase):
    def test_event_statistics_follow_the_frozen_formulas(self):
        # window frames [39, 99], settlement at the last collision of the unit timeline
        stats = regime.window_event_statistics([34, 84], 39, 61, 84)
        self.assertAlmostEqual(stats["contact_active_fraction"], 1 / 61)
        self.assertAlmostEqual(stats["post_settlement_fraction"], 15 / 61)
        self.assertAlmostEqual(stats["collision_event_rate"], 1 / 61)
        self.assertFalse(stats["no_collision_events"])
        # duplicate records on one frame count once for contact, twice for rate
        stats = regime.window_event_statistics([10, 10, 11], 0, 61, 11)
        self.assertAlmostEqual(stats["contact_active_fraction"], 2 / 61)
        self.assertAlmostEqual(stats["collision_event_rate"], 3 / 61)
        self.assertAlmostEqual(stats["post_settlement_fraction"], 49 / 61)

    def test_no_collision_timeline_is_typed_not_imputed(self):
        stats = regime.window_event_statistics([], 0, 61, -1)
        self.assertEqual(stats["contact_active_fraction"], 0.0)
        self.assertEqual(stats["post_settlement_fraction"], 1.0)
        self.assertEqual(stats["collision_event_rate"], 0.0)
        self.assertTrue(stats["no_collision_events"])

    def test_velocity_decay_ratio_is_last_over_peak_and_typed_when_motionless(self):
        profile = regime.window_velocity_profile([0.5, 0.4, 0.1])
        self.assertAlmostEqual(profile["velocity_decay_ratio"], 0.2)
        self.assertIsNone(profile["typed_reason"])
        profile = regime.window_velocity_profile([0.0, 0.0, 0.0])
        self.assertIsNone(profile["velocity_decay_ratio"])
        self.assertEqual(profile["typed_reason"], "no_motion_in_window")
        self.assertEqual((profile["speed_first"], profile["speed_peak"]), (0.0, 0.0))

    def test_carrier_speed_averages_present_slots_only(self):
        carrier = [0.0] * regime.CARRIER_DIM
        carrier[2] = 1.0                       # slot 0 present
        carrier[2 + 8], carrier[2 + 9] = 3.0, 4.0
        carrier[2 + 13] = 1.0                  # slot 1 present but motionless
        self.assertEqual(regime.carrier_speed(carrier), 2.5)

    def test_clevrer_windows_enumerate_the_frozen_starts(self):
        rows = regime.clevrer_window_rows(clevrer_shard([34, 84], moving_clip()))
        self.assertEqual(len(rows), 40)
        first = rows[0]
        self.assertEqual(first["cell_id"], "scene-10000:window-00")
        self.assertAlmostEqual(first["contact_active_fraction"], 1 / 61)
        self.assertEqual(first["post_settlement_fraction"], 0.0)      # window ends at 60 < 84
        self.assertAlmostEqual(rows[39]["post_settlement_fraction"], 15 / 61)
        # the clip decays 0.5 -> 0.05 over the first 61 frames: ratio 0.1 at both ends
        self.assertAlmostEqual(first["velocity_decay_ratio"], 0.1)
        self.assertAlmostEqual(first["speed_peak"], 0.5)

    def test_novphy_branches_map_native_steps_to_observed_frames(self):
        document = novphy_target()
        rows = regime.novphy_window_rows(document)
        self.assertEqual(len(rows), 13)
        row = rows[0]
        self.assertEqual(row["cell_id"], "state-007:candidate-00")
        # native steps 5167 and 8148 map to observed frames 103 and 162
        self.assertEqual(row["event_records_in_window"], 2)
        self.assertTrue(row["absorbed_endpoint"])
        self.assertTrue(row["censored"])
        # motion on slot 0 only, averaged over all 18 present slots (frozen formula)
        self.assertAlmostEqual(row["speed_peak"], 0.02 / 18)
        self.assertEqual(row["velocity_decay_ratio"], 0.0)   # absorbed endpoint is settled
        self.assertAlmostEqual(row["post_settlement_fraction"],
                               (601 - (162 + 1)) / 601)
        broken = novphy_target(with_events=((-50,),))
        row = regime.novphy_window_rows(broken)[0]
        self.assertEqual(row["event_records_in_window"], 0)

    def test_family_intervals_skip_typed_cells_and_report_unavailable(self):
        rows = regime.novphy_window_rows(novphy_target())
        interval, unavailable = regime.family_intervals(rows, "contact_active_fraction", 200, 7)
        self.assertEqual(interval["n"], 13)
        self.assertEqual(unavailable, 0)
        motionless = novphy_target(with_events=((10,),), moving=False)
        rows = regime.novphy_window_rows(motionless)
        interval, unavailable = regime.family_intervals(rows, "velocity_decay_ratio", 200, 7)
        self.assertIsNone(interval)
        self.assertEqual(unavailable, 13)


class RuleTests(unittest.TestCase):
    def test_rule_outcome_respects_direction_and_margin(self):
        self.assertEqual(regime.rule_outcome(0.0, 0.5, 0.7, "decreasing", 0.1), "consistent")
        self.assertEqual(regime.rule_outcome(0.45, 0.5, 0.7, "decreasing", 0.1), "unresolved")
        self.assertEqual(regime.rule_outcome(0.9, 0.5, 0.7, "decreasing", 0.1), "inverted")
        self.assertEqual(regime.rule_outcome(0.9, 0.5, 0.7, "increasing", 0.1), "consistent")
        self.assertEqual(regime.rule_outcome(0.0, 0.5, 0.7, "increasing", 0.1), "inverted")

    def test_disposition_mappings_use_only_frozen_tokens(self):
        self.assertEqual(regime.question_1_disposition("consistent", True), "supported")
        self.assertEqual(regime.question_1_disposition("unresolved", True),
                         "not_supported_by_this_experiment")
        self.assertEqual(regime.question_1_disposition(None, False),
                         "readiness_or_precision_insufficient")
        self.assertEqual(regime.question_2_disposition(["unresolved", "consistent"]),
                         "supported")
        self.assertEqual(regime.question_2_disposition(["unresolved", "inverted"]),
                         "not_supported_by_this_experiment")
        self.assertEqual(regime.question_2_disposition([None, None]),
                         "readiness_or_precision_insufficient")
        for token in (regime.question_1_disposition("consistent", True),
                      regime.question_2_disposition(["consistent"])):
            self.assertIn(token, regime.DISPOSITION_TOKENS)

    def test_falsification_regions_constrain_by_presence_count(self):
        record = regime.falsification_record(0.0, 0.5, 0.7, "decreasing", 0.1)
        self.assertEqual(record["count_3"], (float("-inf"), 0.4))
        self.assertEqual(record["count_2"], (0.1, float("inf")))
        self.assertEqual(record["count_le_1"], (0.7 + 0.1, float("inf")))
        record = regime.falsification_record(0.9, 0.5, 0.7, "increasing", 0.1)
        self.assertEqual(record["count_3"], (0.7 + 0.1, float("inf")))
        self.assertEqual(record["count_2"], (float("-inf"), 0.9 - 0.1))
        self.assertEqual(record["count_le_1"], (float("-inf"), 0.5 - 0.1))

    def test_descriptive_interval_is_deterministic_and_brackets_the_mean(self):
        first = regime.descriptive_interval(list(range(100)), 4000, 11)
        second = regime.descriptive_interval(list(range(100)), 4000, 11)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 49.5)
        self.assertLess(first["ci_low"], first["mean"])
        self.assertGreater(first["ci_high"], first["mean"])


class RunnerContractTests(unittest.TestCase):
    def test_cli_help_and_mutually_exclusive_modes(self):
        helped = subprocess.run([sys.executable, "-m",
                                 "scripts.run_cross_family_regime_synthesis", "--help"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(helped.returncode, 0)
        self.assertIn("--dry-run", helped.stdout)
        both = subprocess.run([sys.executable, "-m", "scripts.run_cross_family_regime_synthesis",
                               "--dry-run", "--prepare"], capture_output=True, text=True,
                              cwd=ROOT)
        self.assertNotEqual(both.returncode, 0)
        self.assertIn("not allowed with argument", both.stderr)

    def test_frozen_plan_rejects_placeholders_and_incomplete_sections(self):
        plan = runner.make_plan()
        self.assertEqual(runner._validate_plan(plan), plan)
        for key in ("component_table", "rules", "bootstrap"):
            broken = json.loads(json.dumps(plan))
            del broken[key]
            with self.assertRaises(ValueError):
                runner._validate_plan(broken)
        broken = json.loads(json.dumps(plan))
        broken["claim_boundary"] += " ... or something ~"
        with self.assertRaises(ValueError):
            runner._validate_plan(broken)

    def test_frozen_rules_cover_the_four_statistics_with_one_primary(self):
        plan = runner.make_plan()
        self.assertEqual({rule["statistic"] for rule in plan["rules"]},
                         set(regime.STATISTIC_NAMES))
        self.assertEqual(len(plan["rules"]), 4)
        self.assertEqual([rule["role"] for rule in plan["rules"] if rule["role"] == "primary"],
                         ["primary"])
        self.assertEqual([rule["statistic"] for rule in plan["rules"]
                          if rule["role"] == "primary"], ["post_settlement_fraction"])
        self.assertTrue(all(isinstance(rule["margin"], float) and rule["margin"] > 0
                            for rule in plan["rules"]))
        self.assertTrue(all(rule["direction"] in regime.RULE_DIRECTIONS
                            for rule in plan["rules"]))

    def test_component_table_matches_published_dispositions(self):
        table = runner.COMPONENT_TABLE
        self.assertEqual(table["novphy_normal_mechanics"]["presence_count"], 3)
        self.assertEqual(table["physion_dominoes"],
                         {**table["physion_dominoes"], "S1": "absent",
                          "S2": "present", "S3": "present", "presence_count": 2})
        self.assertEqual(table["clevrer"],
                         {**table["clevrer"], "S1": "present", "S2": "present",
                          "S3": "absent", "presence_count": 2})
        clevrer_summary = json.loads(
            (ROOT / ".local-artifacts/issue-79-clevrer-boundary-v1/summary.json").read_text())
        physion_summary = json.loads(
            (ROOT / ".local-artifacts/issue-84-third-family-v1/summary.json").read_text())
        self.assertFalse(clevrer_summary["dispositions"]["question_1_detail"]["components"]["S3"]["holds"])
        self.assertTrue(clevrer_summary["dispositions"]["question_1_detail"]["components"]["S1"]["holds"])
        self.assertFalse(physion_summary["dispositions"]["question_1_detail"]["components"]["S1"]["holds"])
        self.assertTrue(physion_summary["dispositions"]["question_1_detail"]["components"]["S3"]["holds"])

    def test_scheduled_cells_match_the_frozen_contracts(self):
        scheduled = runner.scheduled_cell_ids()
        self.assertEqual(len(scheduled["clevrer"]), 480)
        self.assertEqual(len(scheduled["physion_dominoes"]), 480)
        self.assertEqual(len(scheduled["novphy_normal_mechanics"]), 52)
        self.assertEqual(runner.FAMILIES["clevrer"]["scheduled_cells"], 480)
        self.assertEqual(runner.FAMILIES["physion_dominoes"]["scheduled_cells"], 480)
        self.assertEqual(runner.FAMILIES["novphy_normal_mechanics"]["scheduled_cells"], 52)

    def test_missing_scheduled_cells_become_typed_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "targets").mkdir()
            target = novphy_target()
            target["candidates"] = target["candidates"][:2]
            (root / "targets/state-007.json").write_text(json.dumps(target))
            for state in ("008", "014", "016"):
                (root / f"targets/state-{state}.json").write_text(
                    json.dumps({"plan_identity": "issue-77-n1-diagnostic-v1",
                                "state": f"issue-77-n1-{state}", "candidates": []}))
            rows = runner.compute_family_cells("novphy_normal_mechanics",
                                               {"novphy_normal_mechanics": root})
            self.assertEqual(len(rows), 52)
            typed = [row for row in rows if row["status"] == "typed_terminal_failure"]
            self.assertEqual(len(typed), 50)

    def test_publication_roundtrip_on_a_synthetic_report(self):
        plan = runner.make_plan()
        family_cells = {
            "clevrer": regime.clevrer_window_rows(clevrer_shard([34, 84], moving_clip())),
            "physion_dominoes": [
                dict(row, cell_id=f"trial-0:window-{index:02d}", window_start_frame=index)
                for index, row in enumerate(
                    regime.clevrer_window_rows(clevrer_shard([10], moving_clip())))],
            "novphy_normal_mechanics": regime.novphy_window_rows(novphy_target()),
        }
        # mini plan scales the scheduled counts to the synthetic cells
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            mini_plan = json.loads(json.dumps(plan))
            mini_plan["families"]["clevrer"]["scheduled_cells"] = 40
            mini_plan["families"]["physion_dominoes"]["scheduled_cells"] = 40
            mini_plan["families"]["novphy_normal_mechanics"]["scheduled_cells"] = 13
            report = runner.build_report(mini_plan, family_cells)
            runner.write_json(output / "summary.json", report)
            runner.write_text(output / "comparisons.csv", runner.comparisons_csv(report))
            runner.write_text(output / "regime_modulation.csv",
                              runner.regime_modulation_csv(report))
            runner.write_text(output / "findings.md", runner.findings_md(report))
            rebuilt = runner.build_report(mini_plan, family_cells)
            self.assertEqual(rebuilt, report)
            for name, text in (("comparisons.csv", runner.comparisons_csv(rebuilt)),
                               ("regime_modulation.csv", runner.regime_modulation_csv(rebuilt)),
                               ("findings.md", runner.findings_md(rebuilt))):
                self.assertEqual((output / name).read_text(), text)
            self.assertIn("DESCRIPTIVE", (output / "findings.md").read_text())
            csv_text = (output / "regime_modulation.csv").read_text()
            self.assertIn("falsification_region", csv_text)
            self.assertIn("question_1_regime_modulation", csv_text)


if __name__ == "__main__":
    unittest.main()
