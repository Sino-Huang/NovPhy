"""Focused tests for the issue-85 engine-side outcome channel logic.

Covers the frozen detection rule (event leg, lifecycle leg, consistency rule
including the two declared failure modes), the Phase-A verification set
resolution and its pre-declared pass rule (positive/negative/anomaly/typed
failure paths), and the runner/module constant wiring. No GPU work and no
engine execution is performed.
"""
import unittest

from world_model.training import engine_outcome_reactive_diagnostic as channel
import scripts.run_engine_outcome_reactive_diagnostic as runner


def event(event_type, participants, step=1000):
    return {"event_type": event_type, "participants": participants,
            "fixed_step": step, "event_id": f"event:{step}:{event_type}:0"}


def sample(entities, step=1000):
    return {"fixed_step": step, "entities": entities}


class DetectionRuleTests(unittest.TestCase):
    def test_direct_pig_removed_event_fires_the_channel(self) -> None:
        verdict = channel.pig_channel(
            [event("pig_removed", ["runtime:pig:0000"]),
             event("collision", ["runtime:block:0000", "runtime:pig:0000"])])
        self.assertTrue(verdict["pig_removed"])
        self.assertEqual(len(verdict["pig_removed_events"]), 1)
        self.assertEqual(verdict["pig_removed_events"][0]["event_type"], "pig_removed")
        self.assertFalse(verdict["structure_destroyed"])

    def test_pig_death_and_destruction_events_fire_the_channel(self) -> None:
        verdict = channel.pig_channel([
            event("entity_death", ["runtime:pig:0000"]),
            event("entity_destroyed", ["runtime:pig:0000"])])
        self.assertTrue(verdict["pig_removed"])
        self.assertEqual(len(verdict["pig_removed_events"]), 2)

    def test_block_destruction_is_not_a_pig_removal(self) -> None:
        verdict = channel.pig_channel([
            event("entity_death", ["runtime:block:0001"]),
            event("entity_destroyed", ["runtime:block:0001"])])
        self.assertFalse(verdict["pig_removed"])
        self.assertTrue(verdict["structure_destroyed"])
        self.assertEqual(verdict["structure_destroyed_events"], 2)

    def test_pig_collision_without_death_is_not_a_removal(self) -> None:
        verdict = channel.pig_channel(
            [event("collision", ["runtime:bird:0000", "runtime:pig:0000"])])
        self.assertFalse(verdict["pig_removed"])
        self.assertFalse(verdict["structure_destroyed"])

    def test_unresolved_pig_participant_is_counted_not_dropped(self) -> None:
        verdict = channel.pig_channel([event("entity_death", ["runtime:block:0000"])])
        self.assertFalse(verdict["pig_removed"])
        self.assertEqual(verdict["pig_identity_failures"], 0)
        suspect = channel.pig_channel([event("entity_death", ["pig:unresolved"])])
        self.assertFalse(suspect["pig_removed"])
        self.assertEqual(suspect["pig_identity_failures"], 1)

    def test_lifecycle_leg_detects_destroyed_pigs_only(self) -> None:
        samples = [
            sample([{"entity_id": "runtime:pig:0000", "lifecycle": "active"},
                    {"entity_id": "runtime:block:0000", "lifecycle": "destroyed"}]),
            sample([{"entity_id": "runtime:pig:0000", "lifecycle": "destroyed"},
                    {"entity_id": "runtime:block:0000", "lifecycle": "destroyed"}], step=1001),
        ]
        self.assertEqual(channel.pig_lifecycle_destroyed(samples), ["runtime:pig:0000"])
        alive = [sample([{"entity_id": "runtime:pig:0000", "lifecycle": "active"}])]
        self.assertEqual(channel.pig_lifecycle_destroyed(alive), [])

    def test_consistency_rule_encodes_the_declared_failure_modes(self) -> None:
        self.assertEqual(channel.channel_consistency(True, True), "agreement")
        self.assertEqual(channel.channel_consistency(False, False), "agreement")
        self.assertEqual(channel.channel_consistency(True, False), "death_at_window_edge")
        self.assertEqual(channel.channel_consistency(False, True), "anomaly")


def fake_states():
    def state(identity, ordinals, engine_seed=764100001):
        return {"identity": identity, "source_member": identity.rsplit("-a", 1)[0],
                "engine_seed": engine_seed,
                "inventory": [{"ordinal": ordinal,
                               "branch_identity": f"{identity.rsplit('-a', 1)[0]}-a{ordinal:02d}",
                               "action": {"drag_x": 0.0, "drag_y": 0.0,
                                          "release_time_ms": 1000, "tap_time_ms": 0}}
                              for ordinal in ordinals]}
    return [state("issue-77-n1-010-a06", range(13), 764100010),
            state("issue-77-n1-016-a12", range(13), 764100016),
            state("issue-77-n1-001-a00", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]),
            state("issue-77-n1-009-a00", range(13), 764100009)]


class VerificationSetTests(unittest.TestCase):
    def test_spec_resolves_against_membership_inventories(self) -> None:
        resolved = channel.verification_set(fake_states())
        self.assertEqual(len(resolved), 4)
        by_state = {entry["state"]: entry for entry in resolved}
        self.assertEqual(by_state["issue-77-n1-010-a06"]["ordinal"], 2)
        self.assertTrue(by_state["issue-77-n1-010-a06"]["expectation"])
        self.assertEqual(by_state["issue-77-n1-016-a12"]["ordinal"], 3)
        self.assertTrue(by_state["issue-77-n1-016-a12"]["expectation"])
        self.assertEqual(by_state["issue-77-n1-001-a00"]["ordinal"], 0)
        self.assertFalse(by_state["issue-77-n1-001-a00"]["expectation"])
        self.assertEqual(by_state["issue-77-n1-009-a00"]["ordinal"], 0)
        self.assertFalse(by_state["issue-77-n1-009-a00"]["expectation"])
        self.assertTrue(all(entry["seed"] == channel.VERIFICATION_SEED
                            for entry in resolved))

    def test_inadmissible_ordinal_refuses_to_freeze(self) -> None:
        states = fake_states()
        states[0]["inventory"] = [item for item in states[0]["inventory"]
                                  if item["ordinal"] != 2]
        self.assertRaises(ValueError, channel.verification_set, states)
        self.assertRaises(ValueError, channel.verification_set, [])

    def test_runner_plan_binds_the_resolved_set(self) -> None:
        resolved = channel.verification_set(fake_states())
        spec = [(entry["role"], entry["state"], entry["ordinal"], entry["expectation"])
                for entry in resolved]
        self.assertEqual(spec, list(channel.VERIFICATION_SPEC))


def verification_records(expectations):
    """Build record pairs for the four frozen controls from a compact map.

    expectations: state -> dict(pig_removed=..., executed=True/False,
    consistency override optional).
    """
    entries = []
    for role, state, ordinal, expectation in channel.VERIFICATION_SPEC:
        flags = expectations[state]
        executed = flags.get("executed", True)
        pig_removed = flags.get("pig_removed", expectation)
        lifecycle = flags.get("lifecycle", pig_removed)
        consistency = (channel.channel_consistency(pig_removed, lifecycle)
                       if executed else None)
        record = {"cell": {"identity": f"verification--{role}--{state}"},
                  "failure": None if executed else "TimeoutError: deadline",
                  "engine_channel": {"pig_removed": pig_removed,
                                     "pig_lifecycle_destroyed": lifecycle,
                                     "bird_launches": 1 if executed else 0,
                                     "consistency": consistency}}
        entries.append(({"role": role, "state": state, "ordinal": ordinal,
                         "branch_identity": "branch", "action": {},
                         "engine_seed": 1, "seed": 20260908,
                         "expectation": expectation}, record))
    return entries


class VerificationVerdictTests(unittest.TestCase):
    def test_matching_controls_verify_the_channel(self) -> None:
        entries = verification_records({
            "issue-77-n1-010-a06": {"pig_removed": True},
            "issue-77-n1-016-a12": {"pig_removed": True},
            "issue-77-n1-001-a00": {"pig_removed": False},
            "issue-77-n1-009-a00": {"pig_removed": False},
        })
        verdict = channel.verification_verdict(entries)
        self.assertTrue(verdict["verified"])
        self.assertIsNone(verdict["blocker"])
        self.assertEqual(verdict["anomalies"], [])

    def test_one_positive_removal_demonstrates_the_channel(self) -> None:
        entries = verification_records({
            "issue-77-n1-010-a06": {"pig_removed": False},
            "issue-77-n1-016-a12": {"pig_removed": True},
            "issue-77-n1-001-a00": {"pig_removed": False},
            "issue-77-n1-009-a00": {"pig_removed": False},
        })
        verdict = channel.verification_verdict(entries)
        self.assertTrue(verdict["verified"])

    def test_no_positive_removal_names_the_blocker(self) -> None:
        entries = verification_records({
            "issue-77-n1-010-a06": {"pig_removed": False},
            "issue-77-n1-016-a12": {"pig_removed": False},
            "issue-77-n1-001-a00": {"pig_removed": False},
            "issue-77-n1-009-a00": {"pig_removed": False},
        })
        verdict = channel.verification_verdict(entries)
        self.assertFalse(verdict["verified"])
        self.assertIn("not positively verified", verdict["blocker"])

    def test_negative_removal_refutes_the_channel(self) -> None:
        entries = verification_records({
            "issue-77-n1-010-a06": {"pig_removed": True},
            "issue-77-n1-016-a12": {"pig_removed": True},
            "issue-77-n1-001-a00": {"pig_removed": True},
            "issue-77-n1-009-a00": {"pig_removed": False},
        })
        verdict = channel.verification_verdict(entries)
        self.assertFalse(verdict["verified"])
        self.assertIn("negative control recorded an engine-truth pig removal",
                      verdict["blocker"])

    def test_state_stream_destruction_without_events_is_an_anomaly(self) -> None:
        entries = verification_records({
            "issue-77-n1-010-a06": {"pig_removed": True},
            "issue-77-n1-016-a12": {"pig_removed": True},
            "issue-77-n1-001-a00": {"pig_removed": False, "lifecycle": True},
            "issue-77-n1-009-a00": {"pig_removed": False},
        })
        verdict = channel.verification_verdict(entries)
        self.assertFalse(verdict["verified"])
        self.assertIn("anomaly", verdict["blocker"])
        self.assertEqual(verdict["anomalies"], ["verification--negative--issue-77-n1-001-a00"])

    def test_typed_execution_failure_is_retained_and_blocks(self) -> None:
        entries = verification_records({
            "issue-77-n1-010-a06": {"executed": False},
            "issue-77-n1-016-a12": {"pig_removed": True},
            "issue-77-n1-001-a00": {"pig_removed": False},
            "issue-77-n1-009-a00": {"pig_removed": False},
        })
        verdict = channel.verification_verdict(entries)
        self.assertFalse(verdict["verified"])
        self.assertIn("typed failures retained", verdict["blocker"])
        self.assertEqual(verdict["failures"],
                         ["verification--positive--issue-77-n1-010-a06"])

    def test_death_at_window_edge_is_not_an_anomaly(self) -> None:
        entries = verification_records({
            "issue-77-n1-010-a06": {"pig_removed": True, "lifecycle": False},
            "issue-77-n1-016-a12": {"pig_removed": True},
            "issue-77-n1-001-a00": {"pig_removed": False},
            "issue-77-n1-009-a00": {"pig_removed": False},
        })
        verdict = channel.verification_verdict(entries)
        self.assertTrue(verdict["verified"])
        detail = next(detail for detail in verdict["cells"]
                      if detail["state"] == "issue-77-n1-010-a06")
        self.assertEqual(detail["consistency"], "death_at_window_edge")


def media_review(visibilities):
    return {"schema": channel.MEDIA_REVIEW_SCHEMA,
            "reviewed_by": "test", "reviewed_at_utc": "2026-09-22T00:00:00Z",
            "cells": [{"identity": f"verification--{role}--{state}",
                       "pig_visible_in_final_frame": visible, "notes": ""}
                      for (role, state, _ordinal, _expectation), visible
                      in zip(channel.VERIFICATION_SPEC, visibilities)]}


class MediaReviewTests(unittest.TestCase):
    def matching_entries(self):
        return verification_records({
            "issue-77-n1-010-a06": {"pig_removed": True},
            "issue-77-n1-016-a12": {"pig_removed": True},
            "issue-77-n1-001-a00": {"pig_removed": False},
            "issue-77-n1-009-a00": {"pig_removed": False},
        })

    def test_pending_media_review_keeps_the_channel_unverified(self) -> None:
        verdict = channel.apply_media_review(
            channel.verification_verdict(self.matching_entries()), None)
        self.assertFalse(verdict["verified"])
        self.assertTrue(verdict["media_review_pending"])
        self.assertIn("media review", verdict["blocker"])

    def test_agreeing_media_review_verifies_the_channel(self) -> None:
        # positives: pig gone (False); negatives: pig visible (True)
        verdict = channel.apply_media_review(
            channel.verification_verdict(self.matching_entries()),
            media_review([False, False, True, True]))
        self.assertTrue(verdict["verified"])
        self.assertFalse(verdict["media_review_pending"])
        self.assertEqual(verdict["media_disagreements"], [])

    def test_disagreeing_media_review_blocks_the_channel(self) -> None:
        verdict = channel.apply_media_review(
            channel.verification_verdict(self.matching_entries()),
            media_review([True, False, True, True]))
        self.assertFalse(verdict["verified"])
        self.assertEqual(verdict["media_disagreements"],
                         ["verification--positive--issue-77-n1-010-a06"])
        self.assertIn("media review disagrees", verdict["blocker"])

    def test_incomplete_media_review_blocks_the_channel(self) -> None:
        review = media_review([False, False, True, True])
        review["cells"] = review["cells"][:3]
        verdict = channel.apply_media_review(
            channel.verification_verdict(self.matching_entries()), review)
        self.assertFalse(verdict["verified"])
        self.assertEqual(verdict["media_missing"],
                         ["verification--negative--issue-77-n1-009-a00"])


if __name__ == "__main__":
    unittest.main()
