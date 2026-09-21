"""Focused tests for the issue-80 reactive diagnostic runner logic.

Covers the frozen membership spacing, candidate-inventory construction,
regret normalization, the pilot prevalence gate, cell identity plumbing,
and the gate-failure publication path. No engine execution is performed.
"""
import tempfile
import unittest
from pathlib import Path

import scripts.run_short_horizon_reactive_diagnostic as runner


def _synthetic_campaign(members_per_family=2, branches=13, dropped=()):
    members = []
    for index in range(members_per_family * 2):
        identity = f"issue-77-n1-{index + 1:03d}"
        members.append({
            "identity": identity,
            "generator_family": "type010103" if index % 2 == 0 else "type010105",
            "study_role": "predictor_train",
            "engine_seed": 1000 + index,
            "novelty_level": 1,
            "generation_seed": 2000 + index,
            "exposure_role": "training",
            "base_cluster": identity,
            "generated_slots": (),
            "template": {"template": "unused", "constraints": None},
            "scenario": {"identity": identity + ":scenario"},
            "xml": "<xml/>",
        })
    branches_payload = []
    for member in members:
        for ordinal in range(branches):
            branch_identity = f"{member['identity']}-a{ordinal:02d}"
            branches_payload.append({
                "identity": branch_identity,
                "source_member_identity": member["identity"],
                "candidate_ordinal": ordinal,
                "action": {"drag_x": -80, "drag_y": 10,
                           "release_time_ms": 1000, "tap_time_ms": 0},
            })
    coverage = {"branches": {b["identity"]: {"status": "failed" if b["identity"] in dropped
                                             else "admissible"}
                             for b in branches_payload}}
    return {"members": members, "branches": branches_payload}, coverage


class MembershipSpacingTests(unittest.TestCase):
    def test_states_are_evenly_spaced_per_family(self) -> None:
        campaign, coverage = _synthetic_campaign()
        identities = runner.select_states(campaign, coverage)
        self.assertEqual(len(identities), 24)
        for family in runner.FAMILIES:
            pool = sorted(b["identity"] for b in campaign["branches"]
                          if b["identity"] not in dropped_set(coverage)
                          and family_of(campaign, b["identity"].rsplit("-a", 1)[0]) == family)
            expected = [pool[round(i * (len(pool) - 1) / 11)] for i in range(12)]
            chosen = [i for i in identities if family_of(campaign, i.rsplit("-a", 1)[0]) == family]
            self.assertEqual(chosen, expected)

    def test_typed_drops_shrink_inventory_and_set_prior_flag(self) -> None:
        campaign, coverage = _synthetic_campaign(
            dropped=("issue-77-n1-001-a08", "issue-77-n1-001-a03"))
        payload = runner.state_payload(campaign, coverage, "issue-77-n1-001-a00")
        self.assertEqual(len(payload["inventory"]), 11)
        self.assertEqual(payload["typed_dropped_branches"],
                         ["issue-77-n1-001-a03", "issue-77-n1-001-a08"])
        self.assertFalse(payload["prior_available"])
        self.assertEqual(payload["prior_ordinal"], 8)
        payload_ok = runner.state_payload(campaign, coverage, "issue-77-n1-002-a00")
        self.assertTrue(payload_ok["prior_available"])
        self.assertEqual(len(payload_ok["inventory"]), 13)

    def test_membership_refuses_undersized_family(self) -> None:
        campaign, coverage = _synthetic_campaign(
            dropped={f"issue-77-n1-{member:03d}-a{ordinal:02d}"
                     for member in (1, 3) for ordinal in range(13)})
        with self.assertRaisesRegex(ValueError, "unsatisfiable"):
            runner.select_states(campaign, coverage)


def dropped_set(coverage):
    return {identity for identity, entry in coverage["branches"].items()
            if entry["status"] != "admissible"}


def family_of(campaign, member_identity):
    return runner.family_of_member(campaign, member_identity)


class RegretAndGateTests(unittest.TestCase):
    def test_normalized_regret_is_unclipped_and_all_tied_aware(self) -> None:
        outcomes = {"lowest_cost": 1000.0, "highest_cost": 3000.0, "informative": True,
                    "all_tied": False}
        self.assertEqual(runner.normalized_regret(1000.0, outcomes), 0.0)
        self.assertEqual(runner.normalized_regret(3000.0, outcomes), 1.0)
        self.assertEqual(runner.normalized_regret(2500.0, outcomes), 0.75)
        self.assertEqual(runner.normalized_regret(800.0, outcomes), -0.1)
        tied = {"informative": False, "all_tied": True}
        self.assertEqual(runner.normalized_regret(500.0, tied), 0.0)

    def test_gate_prevalence_and_typed_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original_output = runner.OUTPUT
            runner.OUTPUT = Path(temporary)
            try:
                plan = {"identity": runner.IDENTITY,
                        "states": [{"identity": "issue-77-n1-001-a00"}],
                        "pilot_states": ["issue-77-n1-001-a00"]}
                def write_cell(system, seed, success=None, failure=None):
                    identity = runner.cell_identity(system, seed, "issue-77-n1-001-a00")
                    record = {"schema": runner.SCHEMA, "failure": failure,
                              "outcome": None if success is None else
                              {"first_shot_success": success}}
                    runner.write(runner.OUTPUT / "records" / f"{identity}.json", record)
                    runner.write(runner.OUTPUT / "receipts" / f"{identity}.json",
                                 {"identity": identity})
                systems = runner.SYSTEMS
                seeds = runner.SEEDS
                outcomes = [True, True, False, False, None, None]
                index = 0
                for system in systems:
                    for seed in seeds:
                        outcome = outcomes[index % len(outcomes)]
                        index += 1
                        if outcome is None:
                            write_cell(system, seed, failure="decided: no")
                        else:
                            write_cell(system, seed, success=outcome)
                gate = runner.evaluate_gate(plan)
                self.assertEqual(gate["cells"], len(systems) * len(seeds))
                self.assertEqual(gate["valid_executions"], 8)
                self.assertEqual(gate["successes"], 4)
                self.assertEqual(gate["prevalence"], 0.5)
                self.assertFalse(gate["passed"])  # below minimum valid executions
                self.assertEqual(len(gate["typed_failures"]), 4)
                self.assertEqual(gate["disposition_if_failed"],
                                 "readiness_or_precision_insufficient")
            finally:
                runner.OUTPUT = original_output

    def test_gate_requires_terminal_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original_output = runner.OUTPUT
            runner.OUTPUT = Path(temporary)
            try:
                plan = {"identity": runner.IDENTITY,
                        "states": [{"identity": "s"}], "pilot_states": ["s"]}
                with self.assertRaisesRegex(ValueError, "terminal record"):
                    runner.evaluate_gate(plan)
            finally:
                runner.OUTPUT = original_output


class PublicationAndPlumbingTests(unittest.TestCase):
    def test_cell_identity_round_trip(self) -> None:
        identity = runner.cell_identity("hybrid-fixed-h1", 20260908, "issue-77-n1-001-a00")
        self.assertEqual(identity,
                         "hybrid-fixed-h1--seed20260908--issue-77-n1-001-a00")
        cell = runner.split_cell_identity({"states": []}, identity)
        self.assertEqual(cell, {"identity": identity, "system": "hybrid-fixed-h1",
                                "seed": 20260908, "state": "issue-77-n1-001-a00"})

    def test_gate_failure_publication_stops_the_ticket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original_output = runner.OUTPUT
            runner.OUTPUT = Path(temporary)
            try:
                plan = {"identity": runner.IDENTITY,
                        "definitions": runner.definitions(),
                        "states": [{"identity": "issue-77-n1-001-a00"}],
                        "pilot_states": ["issue-77-n1-001-a00"]}
                runner.write(runner.OUTPUT / "ledger.json",
                             {"schema": "issue_80_ledger_v1",
                              "plan_identity": runner.IDENTITY, "status": "running",
                              "wall_seconds_elapsed": 1.0, "gpu_seconds_elapsed": 1.0,
                              "cells": {}})
                gate = {"schema": "issue_80_pilot_gate_v1", "passed": False,
                        "prevalence": 0.0, "floor": 0.10,
                        "valid_executions": 30, "successes": 0, "cells": 48,
                        "minimum_valid_executions": 24, "typed_failures": [],
                        "disposition_if_failed": "readiness_or_precision_insufficient"}
                runner.write(runner.OUTPUT / "pilot-gate.json", gate)
                result = runner.publication(plan)
                self.assertFalse(result["diagnostics_complete"])
                self.assertEqual(result["ticket_disposition"],
                                 "readiness_or_precision_insufficient")
                text = runner.findings_md(result)
                self.assertIn("readiness_or_precision_insufficient", text)
                self.assertIn("pilot gate", text.lower())
            finally:
                runner.OUTPUT = original_output

    def test_publication_requires_full_record_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original_output = runner.OUTPUT
            runner.OUTPUT = Path(temporary)
            try:
                plan = {"identity": runner.IDENTITY, "definitions": runner.definitions(),
                        "states": [{"identity": "s"}], "pilot_states": ["s"]}
                runner.write(runner.OUTPUT / "ledger.json",
                             {"schema": "issue_80_ledger_v1",
                              "plan_identity": runner.IDENTITY, "status": "complete",
                              "wall_seconds_elapsed": 1.0, "gpu_seconds_elapsed": 1.0,
                              "cells": {}})
                with self.assertRaisesRegex(ValueError, "terminal record"):
                    runner.publication(plan)
            finally:
                runner.OUTPUT = original_output

    def test_frozen_definitions_carry_real_numbers(self) -> None:
        definitions = runner.definitions()
        self.assertEqual(definitions["states_per_family"], 12)
        self.assertEqual(definitions["caps"]["wall_cap_seconds"], 86400)
        self.assertEqual(definitions["caps"]["gpu_cap_seconds"], 7200)
        self.assertEqual(definitions["pilot_gate"]["cells"], 48)
        self.assertEqual(definitions["pilot_gate"]["prevalence_floor"], 0.10)
        text = repr(definitions)
        self.assertNotIn("e.g.", text)
        self.assertNotIn("~", text)
        self.assertNotIn("TODO", text)


class ResumeAndFormattingTests(unittest.TestCase):
    def _plan(self):
        return {"identity": runner.IDENTITY, "definitions": runner.definitions(),
                "states": [{"identity": "issue-77-n1-001-a00"}],
                "pilot_states": ["issue-77-n1-001-a00"]}

    def test_partial_persistence_recovery_is_outcome_preserving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original_output = runner.OUTPUT
            runner.OUTPUT = Path(temporary)
            try:
                plan = self._plan()
                identity_a = runner.cell_identity("hybrid-fixed-h1", runner.SEEDS[0],
                                                  "issue-77-n1-001-a00")
                identity_b = runner.cell_identity("continuous-fixed-h1", runner.SEEDS[0],
                                                  "issue-77-n1-001-a00")
                identity_c = runner.cell_identity("continuous-fixed-h5", runner.SEEDS[0],
                                                  "issue-77-n1-001-a00")
                # a: record without receipt (outcome recorded, bookkeeping lost)
                runner.write(runner.OUTPUT / "records" / f"{identity_a}.json",
                             {"schema": runner.SCHEMA, "failure": None,
                              "outcome": {"first_shot_success": True}})
                # b: receipt without record (terminated before any outcome)
                runner.write(runner.OUTPUT / "receipts" / f"{identity_b}.json",
                             {"identity": identity_b, "worker_exitcode": -9,
                              "stop": "attempt_wall_limit"})
                # c: attempt tree with neither file (pure crash)
                (runner.OUTPUT / "attempts" / identity_c).mkdir(parents=True)
                terminal, recovered = runner.terminal_cells(plan)
                self.assertTrue(terminal[identity_a])
                self.assertTrue(
                    (runner.OUTPUT / "receipts" / f"{identity_a}.json").is_file())
                self.assertTrue(terminal[identity_b])
                typed = runner.read(runner.OUTPUT / "records" / f"{identity_b}.json")
                self.assertEqual(typed["failure_kind"], "execution_failure")
                self.assertIsNotNone(typed["failure"])
                self.assertFalse(terminal[identity_c])
                self.assertEqual(recovered, [identity_c])
                self.assertFalse((runner.OUTPUT / "attempts" / identity_c).exists())
            finally:
                runner.OUTPUT = original_output

    def test_findings_render_all_typed_failure_rows(self) -> None:
        result = {
            "diagnostics_complete": True, "execution_ledger_status": "complete",
            "claim_boundary": "b", "comparator_disclosure": "c",
            "definitions": runner.definitions(),
            "pilot_gate": {"prevalence": 1.0, "successes": 1, "valid_executions": 1,
                           "typed_failures": [], "floor": 0.1, "passed": True},
            "per_seed": {"20260908": {"hybrid-fixed-h1": {
                "executed": 0, "mean_regret": None, "success_fraction": None,
                "mean_realized_count_cost": None, "decision_failures": 12,
                "execution_failures": 0, "candidate_counts": [],
                "transition_calls": 0, "linear_macs": 0,
                "mean_decision_wall_seconds": None,
                "mean_engine_wall_seconds": None}}},
            "pooled": {"hybrid-fixed-h1": {
                "executed": 0, "mean_regret": None, "success_fraction": None,
                "transition_calls": 0, "linear_macs": 0, "candidate_counts": []}},
            "contrasts": [{"tested": "hybrid-fixed-h1", "reference": "continuous-fixed-h5",
                           "metric": "regret", "descriptive": None, "paired_states": 0}],
            "ticket_disposition": "not_supported_by_this_experiment",
            "limitations": ["descriptive intervals only"],
        }
        text = runner.findings_md(result)
        self.assertIn("n/a", text)
        self.assertNotIn("NoneType", text)

    def test_diagnostics_complete_requires_run_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original_output = runner.OUTPUT
            runner.OUTPUT = Path(temporary)
            try:
                plan = self._plan()
                runner.write(runner.OUTPUT / "ledger.json",
                             {"schema": "issue_80_ledger_v1",
                              "plan_identity": runner.IDENTITY, "status": "complete",
                              "phase": "pilot", "wall_seconds_elapsed": 1.0,
                              "gpu_seconds_elapsed": 1.0, "cells": {}})
                runner.write(runner.OUTPUT / "pilot-gate.json",
                             {"schema": "issue_80_pilot_gate_v1", "passed": True,
                              "prevalence": 0.5, "floor": 0.10, "successes": 12,
                              "valid_executions": 24, "cells": 48,
                              "minimum_valid_executions": 24, "typed_failures": [],
                              "disposition_if_failed": "readiness_or_precision_insufficient"})
                with self.assertRaisesRegex(ValueError, "terminal record"):
                    runner.publication(plan)
            finally:
                runner.OUTPUT = original_output


if __name__ == "__main__":
    raise SystemExit(unittest.main())
