"""Tests for the `physics_macro_labels_v1` artifact (fixture-only Milestone 0a).

The expectations pinned here are the independent hand-computed oracle for the nine
`physics_capture_v1_macro` fixtures.  They cover the per-state macro predicates,
the fixed-step event intervals, the shot outcomes, the header/vocabulary contract,
canonical bytes, fail-closed reading/validation, and the derivation CLI.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path

from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_macro_labels import (
    ABSORBING_PREDICATES,
    DERIVATION_SPEC_VERSION,
    MACRO_LABEL_SCHEMA_VERSION,
    MACRO_LABEL_SIDECAR,
    PIG_CLASS_SET,
    PREDICATE_SEMANTIC_STATUS,
    Availability,
    MacroLabelError,
    MacroPredicate,
    OutcomeClass,
    SemanticStatus,
    TerminalEquilibrium,
    derivation_spec_digest,
    derive_macro_labels_for_shot,
    read_macro_labels,
    validate_macro_labels,
    write_macro_labels,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "physics_capture_v1_macro"
CASE_NAMES = (
    "accepted_smoke_one_state",
    "canonical_multistate",
    "collapse_disappearance",
    "collapse_support_loss_only",
    "no_events",
    "pig_tags",
    "same_step_cluster",
    "settled_nonterminal",
    "terminal_after_last_state",
)
ENGINE = SemanticStatus.ENGINE_VERIFIED
HYPOTHESIS = SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION
AVAILABLE = Availability.AVAILABLE
NO_PREDECESSOR = Availability.UNAVAILABLE_NO_PREDECESSOR
INSUFFICIENT = Availability.UNAVAILABLE_INSUFFICIENT_STATE_EVIDENCE


def shot_dir(case: str) -> Path:
    return FIXTURE_ROOT / case / "shot_001"


def derive(case: str):
    return derive_macro_labels_for_shot(shot_dir(case))


def frame_by_step(labels, fixed_step: int):
    for frame in labels.frames:
        if frame.identity.fixed_step == fixed_step:
            return frame
    raise AssertionError(f"no frame label at fixed_step {fixed_step}")


def predicate_summary(frame, name: str):
    """(value, availability, evidence event_ids) for one macro predicate."""
    label = frame.predicate(MacroPredicate(name))
    return (label.value, label.availability, [citation.event_id for citation in label.evidence])


def active_names(frame) -> list[str]:
    return [predicate.value for predicate in frame.active_macro_states]


def interval_summary(labels) -> list[tuple]:
    return [
        (
            interval.interval_type,
            interval.start_fixed_step,
            interval.end_fixed_step,
            interval.semantic_status,
            [citation.event_id for citation in interval.evidence],
        )
        for interval in labels.intervals
    ]


def jsonl_records(labels) -> list[dict]:
    return [json.loads(line) for line in labels.to_jsonl().splitlines()]


def frame_label_lines(labels) -> list[str]:
    return [
        line
        for line in labels.to_jsonl().splitlines()
        if json.loads(line)["record_type"] == "frame_label"
    ]


def copy_shot(case: str, destination: Path) -> Path:
    shot = destination / case / "shot_001"
    shot.mkdir(parents=True)
    for name in ("physics_state.jsonl", "physics_events.jsonl"):
        shutil.copy(shot_dir(case) / name, shot / name)
    return shot


class VocabularyAndSpecTests(unittest.TestCase):
    """M0a vocabulary pins: predicate set/order, absorbing set, statuses, spec ids."""

    def test_predicates_are_the_pinned_sorted_closed_set(self) -> None:
        values = [predicate.value for predicate in MacroPredicate]
        self.assertEqual(
            values,
            ["cascade-active", "collapsed", "pigs-cleared", "steady-state", "structure-unstable"],
        )
        self.assertEqual(values, sorted(values))

    def test_absorbing_predicates_are_collapsed_and_pigs_cleared(self) -> None:
        self.assertEqual(
            {predicate.value for predicate in ABSORBING_PREDICATES},
            {"collapsed", "pigs-cleared"},
        )

    def test_semantic_status_mapping(self) -> None:
        self.assertEqual(
            {predicate.value: status.value for predicate, status in PREDICATE_SEMANTIC_STATUS},
            {
                "cascade-active": "hypothesis_pending_representative_validation",
                "collapsed": "hypothesis_pending_representative_validation",
                "pigs-cleared": "hypothesis_pending_representative_validation",
                "steady-state": "engine_verified",
                "structure-unstable": "engine_verified",
            },
        )

    def test_pig_class_set_is_pinned_and_sorted(self) -> None:
        self.assertEqual(PIG_CLASS_SET, ("PigBig", "PigMedium", "PigSmall"))

    def test_artifact_constants(self) -> None:
        self.assertEqual(MACRO_LABEL_SCHEMA_VERSION, "physics_macro_labels_v1")
        self.assertEqual(MACRO_LABEL_SIDECAR, "physics_macro_labels.jsonl")
        self.assertEqual(DERIVATION_SPEC_VERSION, "macro_labels_derivation_v1")

    def test_derivation_spec_digest_is_stable_lowercase_hex(self) -> None:
        digest = derivation_spec_digest()
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", digest))
        self.assertEqual(digest, derivation_spec_digest())


class CanonicalMultistateTests(unittest.TestCase):
    """M0a oracle fixture 1 (canonical_multistate): steps 10..15, cleared outcome."""

    def setUp(self) -> None:
        self.labels = derive("canonical_multistate")

    def test_intervals(self) -> None:
        self.assertEqual(
            interval_summary(self.labels),
            [
                ("cascade-active", 12, 15, HYPOTHESIS, ["event:00000001", "event:00000005"]),
                ("steady-state", 15, None, ENGINE, ["event:00000005"]),
            ],
        )

    def test_frame_labels(self) -> None:
        expected = {
            10: {
                "cascade-active": (False, AVAILABLE, []),
                "collapsed": (False, AVAILABLE, []),
                "pigs-cleared": (False, AVAILABLE, []),
                "steady-state": (True, AVAILABLE, ["event:00000000"]),
                "structure-unstable": (None, NO_PREDECESSOR, []),
                "active": ["steady-state"],
            },
            11: {
                "cascade-active": (False, AVAILABLE, []),
                "collapsed": (False, AVAILABLE, []),
                "pigs-cleared": (False, AVAILABLE, []),
                "steady-state": (False, AVAILABLE, []),
                "structure-unstable": (True, AVAILABLE, []),
                "active": ["structure-unstable"],
            },
            12: {
                "cascade-active": (True, AVAILABLE, ["event:00000001", "event:00000005"]),
                "collapsed": (False, AVAILABLE, []),
                "pigs-cleared": (False, AVAILABLE, []),
                "steady-state": (False, AVAILABLE, []),
                "structure-unstable": (False, AVAILABLE, []),
                "active": ["cascade-active"],
            },
            13: {
                # Support set unchanged at 13: 202:0 keeps incoming support despite
                # its destruction event, so structure-unstable is false.
                "cascade-active": (True, AVAILABLE, ["event:00000001", "event:00000005"]),
                "collapsed": (False, AVAILABLE, []),
                "pigs-cleared": (False, AVAILABLE, []),
                "steady-state": (False, AVAILABLE, []),
                "structure-unstable": (False, AVAILABLE, []),
                "active": ["cascade-active"],
            },
            14: {
                "cascade-active": (True, AVAILABLE, ["event:00000001", "event:00000005"]),
                "collapsed": (True, AVAILABLE, ["event:00000002", "event:00000003", "event:00000004"]),
                "pigs-cleared": (True, AVAILABLE, ["event:00000003", "event:00000004"]),
                "steady-state": (False, AVAILABLE, []),
                "structure-unstable": (True, AVAILABLE, []),
                "active": ["cascade-active", "collapsed", "pigs-cleared", "structure-unstable"],
            },
            15: {
                # 15 is the cascade interval's exclusive end; steady interval opens.
                "cascade-active": (False, AVAILABLE, []),
                "collapsed": (True, AVAILABLE, ["event:00000002", "event:00000003", "event:00000004"]),
                "pigs-cleared": (True, AVAILABLE, ["event:00000003", "event:00000004"]),
                "steady-state": (True, AVAILABLE, ["event:00000005"]),
                "structure-unstable": (False, AVAILABLE, []),
                "active": ["collapsed", "pigs-cleared", "steady-state"],
            },
        }
        self.assertEqual([frame.identity.fixed_step for frame in self.labels.frames], [10, 11, 12, 13, 14, 15])
        for step, wanted in expected.items():
            with self.subTest(fixed_step=step):
                frame = frame_by_step(self.labels, step)
                for name in (
                    "cascade-active",
                    "collapsed",
                    "pigs-cleared",
                    "steady-state",
                    "structure-unstable",
                ):
                    self.assertEqual(predicate_summary(frame, name), wanted[name], name)
                self.assertEqual(active_names(frame), wanted["active"])

    def test_outcome_cleared_stable_terminal(self) -> None:
        outcome = self.labels.outcome
        self.assertEqual(outcome.outcome_class, OutcomeClass.CLEARED)
        self.assertEqual(outcome.score, 50000)
        self.assertIsNone(outcome.reason)
        self.assertIsNotNone(outcome.terminal_event)
        self.assertEqual(outcome.terminal_event.event_id, "event:00000006")
        self.assertEqual(outcome.terminal_event.fixed_step, 15)
        self.assertEqual(outcome.terminal_equilibrium, TerminalEquilibrium.STABLE_TERMINAL)
        self.assertIsNotNone(outcome.terminal_state)
        terminal_state = outcome.terminal_state.to_json()
        self.assertEqual(terminal_state["state_sequence"], 6)
        self.assertEqual(terminal_state["render_frame"], 105)
        self.assertEqual(terminal_state["fixed_step"], 15)
        self.assertEqual(terminal_state["rgb_relative_path"], "frames/frame_000105.png")

    def test_citation_shape_and_order(self) -> None:
        for record in jsonl_records(self.labels):
            for evidence in _evidence_lists(record):
                for citation in evidence:
                    self.assertEqual(
                        set(citation.keys()),
                        {"capture_id", "shot_id", "event_sequence", "event_id", "fixed_step"},
                    )
                keys = [(citation["fixed_step"], citation["event_sequence"]) for citation in evidence]
                self.assertEqual(keys, sorted(keys))
                self.assertEqual(len(set(keys)), len(keys))


def _evidence_lists(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence" and isinstance(item, list):
                yield item
            else:
                yield from _evidence_lists(item)
    elif isinstance(value, list):
        for item in value:
            yield from _evidence_lists(item)


class AcceptedSmokeOneStateTests(unittest.TestCase):
    """M0a oracle fixture 2 (accepted_smoke_one_state): single-state reduced smoke."""

    def setUp(self) -> None:
        self.labels = derive("accepted_smoke_one_state")

    def test_intervals(self) -> None:
        self.assertEqual(
            interval_summary(self.labels),
            [
                ("steady-state", 267, 317, ENGINE, ["event:00000000", "event:00000002"]),
                ("cascade-active", 361, 417, HYPOTHESIS, ["event:00000003", "event:00000010"]),
                ("steady-state", 418, None, ENGINE, ["event:00000011"]),
            ],
        )

    def test_single_frame_label(self) -> None:
        self.assertEqual(len(self.labels.frames), 1)
        frame = self.labels.frames[0]
        self.assertEqual(frame.identity.state_sequence, 2)
        self.assertEqual(frame.identity.render_frame, 15207)
        self.assertEqual(frame.identity.fixed_step, 476)
        self.assertEqual(frame.identity.rgb_relative_path, "frames/frame_000000.png")
        self.assertEqual(
            predicate_summary(frame, "steady-state"), (True, AVAILABLE, ["event:00000011"])
        )
        self.assertEqual(
            predicate_summary(frame, "structure-unstable"), (None, NO_PREDECESSOR, [])
        )
        self.assertEqual(
            predicate_summary(frame, "collapsed"), (None, INSUFFICIENT, [])
        )
        self.assertEqual(predicate_summary(frame, "cascade-active"), (False, AVAILABLE, []))
        self.assertEqual(predicate_summary(frame, "pigs-cleared"), (False, AVAILABLE, []))
        self.assertEqual(active_names(frame), ["steady-state"])

    def test_outcome_failed_stable_terminal(self) -> None:
        outcome = self.labels.outcome
        self.assertEqual(outcome.outcome_class, OutcomeClass.FAILED)
        self.assertEqual(outcome.reason, "no_playable_birds")
        self.assertIsNone(outcome.score)
        self.assertIsNotNone(outcome.terminal_event)
        self.assertEqual(outcome.terminal_event.event_id, "event:00000012")
        self.assertEqual(outcome.terminal_event.fixed_step, 441)
        self.assertEqual(outcome.terminal_equilibrium, TerminalEquilibrium.STABLE_TERMINAL)
        self.assertIsNotNone(outcome.terminal_state)
        self.assertEqual(outcome.terminal_state.state_sequence, 2)
        self.assertEqual(outcome.terminal_state.fixed_step, 476)

    def test_event_render_frames_are_uniform_provenance_only(self) -> None:
        # The producer batch-stamps every event with the serialization snapshot's
        # render_frame; interval boundaries above are split purely by fixed_step,
        # which proves no derivation grouped events by render_frame.
        capture = load_physics_capture(
            shot_dir("accepted_smoke_one_state") / "physics_state.jsonl",
            shot_dir("accepted_smoke_one_state") / "physics_events.jsonl",
        )
        self.assertEqual(len(capture.events), 13)
        self.assertTrue(all(event.clock.render_frame == 15207 for event in capture.events))


class SameStepClusterTests(unittest.TestCase):
    """M0a oracle fixture 3 (same_step_cluster): same-step causal cluster at 21."""

    def setUp(self) -> None:
        self.labels = derive("same_step_cluster")

    def test_single_cascade_interval(self) -> None:
        self.assertEqual(
            interval_summary(self.labels),
            [
                (
                    "cascade-active",
                    20,
                    22,
                    HYPOTHESIS,
                    ["event:00000001", "event:00000002", "event:00000003", "event:00000004", "event:00000005"],
                )
            ],
        )

    def test_frame_labels(self) -> None:
        expected = {
            20: {
                "cascade-active": (
                    True,
                    AVAILABLE,
                    ["event:00000001", "event:00000002", "event:00000003", "event:00000004", "event:00000005"],
                ),
                "collapsed": (False, AVAILABLE, []),
                "pigs-cleared": (False, AVAILABLE, []),
                "steady-state": (False, AVAILABLE, []),
                "structure-unstable": (None, NO_PREDECESSOR, []),
                "active": ["cascade-active"],
            },
            21: {
                "cascade-active": (
                    True,
                    AVAILABLE,
                    ["event:00000001", "event:00000002", "event:00000003", "event:00000004", "event:00000005"],
                ),
                "collapsed": (False, AVAILABLE, []),
                "pigs-cleared": (True, AVAILABLE, ["event:00000005"]),
                "steady-state": (False, AVAILABLE, []),
                # Empty support set unchanged between states 20 and 21.
                "structure-unstable": (False, AVAILABLE, []),
                "active": ["cascade-active", "pigs-cleared"],
            },
            22: {
                # 22 is the cascade interval's exclusive end.
                "cascade-active": (False, AVAILABLE, []),
                "collapsed": (False, AVAILABLE, []),
                "pigs-cleared": (True, AVAILABLE, ["event:00000005"]),
                "steady-state": (False, AVAILABLE, []),
                "structure-unstable": (False, AVAILABLE, []),
                "active": ["pigs-cleared"],
            },
        }
        for step, wanted in expected.items():
            with self.subTest(fixed_step=step):
                frame = frame_by_step(self.labels, step)
                for name in (
                    "cascade-active",
                    "collapsed",
                    "pigs-cleared",
                    "steady-state",
                    "structure-unstable",
                ):
                    self.assertEqual(predicate_summary(frame, name), wanted[name], name)
                self.assertEqual(active_names(frame), wanted["active"])

    def test_outcome_unsettled_nonterminal(self) -> None:
        outcome = self.labels.outcome
        self.assertEqual(outcome.outcome_class, OutcomeClass.UNSETTLED_NONTERMINAL)
        self.assertIsNone(outcome.terminal_event)
        self.assertIsNone(outcome.terminal_state)
        self.assertEqual(outcome.terminal_equilibrium, TerminalEquilibrium.NOT_OBSERVED)


class NoEventsTests(unittest.TestCase):
    """M0a oracle fixture 4 (no_events): empty events sidecar, nothing fires."""

    def setUp(self) -> None:
        self.labels = derive("no_events")

    def test_no_intervals_and_zero_header_counts(self) -> None:
        self.assertEqual(self.labels.intervals, ())
        header = self.labels.header_json()
        self.assertEqual(header["event_count"], 0)
        self.assertEqual(header["interval_count"], 0)

    def test_frame_labels(self) -> None:
        expected = {
            30: (None, NO_PREDECESSOR),
            31: (False, AVAILABLE),
        }
        for step, unstable in expected.items():
            with self.subTest(fixed_step=step):
                frame = frame_by_step(self.labels, step)
                for name in ("cascade-active", "collapsed", "pigs-cleared", "steady-state"):
                    self.assertEqual(predicate_summary(frame, name), (False, AVAILABLE, []), name)
                self.assertEqual(
                    predicate_summary(frame, "structure-unstable"),
                    (unstable[0], unstable[1], []),
                )
                self.assertEqual(active_names(frame), [])

    def test_outcome_unsettled_nonterminal(self) -> None:
        self.assertEqual(self.labels.outcome.outcome_class, OutcomeClass.UNSETTLED_NONTERMINAL)


class SettledNonterminalTests(unittest.TestCase):
    """M0a oracle fixture 5 (settled_nonterminal): open steady interval, no terminal."""

    def setUp(self) -> None:
        self.labels = derive("settled_nonterminal")

    def test_open_steady_interval(self) -> None:
        self.assertEqual(
            interval_summary(self.labels),
            [("steady-state", 40, None, ENGINE, ["event:00000000"])],
        )

    def test_steady_true_at_both_states(self) -> None:
        for step in (40, 41):
            with self.subTest(fixed_step=step):
                frame = frame_by_step(self.labels, step)
                self.assertEqual(
                    predicate_summary(frame, "steady-state"), (True, AVAILABLE, ["event:00000000"])
                )
                # No pig identity is ever observed, so pigs-cleared stays false.
                self.assertEqual(predicate_summary(frame, "pigs-cleared"), (False, AVAILABLE, []))
                self.assertEqual(predicate_summary(frame, "collapsed"), (False, AVAILABLE, []))

    def test_outcome_settled_nonterminal(self) -> None:
        outcome = self.labels.outcome
        self.assertEqual(outcome.outcome_class, OutcomeClass.SETTLED_NONTERMINAL)
        self.assertIsNone(outcome.terminal_event)
        self.assertEqual(outcome.terminal_equilibrium, TerminalEquilibrium.NOT_OBSERVED)


class TerminalAfterLastStateTests(unittest.TestCase):
    """M0a oracle fixture 6 (terminal_after_last_state): terminal event at step 60."""

    def setUp(self) -> None:
        self.labels = derive("terminal_after_last_state")
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-macro-terminal-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)

    def test_cascade_interval(self) -> None:
        self.assertEqual(
            interval_summary(self.labels),
            [("cascade-active", 49, 51, HYPOTHESIS, ["event:00000001", "event:00000002"])],
        )

    def test_pre_first_state_onset_reaches_first_projection(self) -> None:
        state50 = frame_by_step(self.labels, 50)
        self.assertEqual(
            predicate_summary(state50, "cascade-active"),
            (True, AVAILABLE, ["event:00000001", "event:00000002"]),
        )
        state51 = frame_by_step(self.labels, 51)
        # 51 is the exclusive end of the cascade interval.
        self.assertEqual(predicate_summary(state51, "cascade-active"), (False, AVAILABLE, []))

    def test_outcome_failed_without_terminal_state(self) -> None:
        outcome = self.labels.outcome
        self.assertEqual(outcome.outcome_class, OutcomeClass.FAILED)
        self.assertEqual(outcome.reason, "no_playable_birds")
        self.assertIsNotNone(outcome.terminal_event)
        self.assertEqual(outcome.terminal_event.event_id, "event:00000003")
        self.assertEqual(outcome.terminal_event.fixed_step, 60)
        self.assertIsNone(outcome.terminal_state)
        self.assertEqual(outcome.terminal_equilibrium, TerminalEquilibrium.NOT_OBSERVED)

    def test_terminal_event_changes_no_frame_label_bytes(self) -> None:
        # Frame labels are computed as if the terminal event did not exist: dropping
        # the level_failed event from a copy must leave frame-label bytes identical.
        variant_shot = copy_shot("terminal_after_last_state", self.temporary / "variant")
        event_path = variant_shot / "physics_events.jsonl"
        kept = [
            line
            for line in event_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event_type"] != "level_failed"
        ]
        self.assertEqual(len(kept), 3)
        event_path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        variant_labels = derive_macro_labels_for_shot(variant_shot)
        self.assertEqual(frame_label_lines(variant_labels), frame_label_lines(self.labels))


class CollapseSupportLossOnlyTests(unittest.TestCase):
    """M0a oracle fixture 7: support loss alone never latches collapsed."""

    def setUp(self) -> None:
        self.labels = derive("collapse_support_loss_only")

    def test_no_intervals(self) -> None:
        self.assertEqual(self.labels.intervals, ())

    def test_frame_labels(self) -> None:
        expected = {
            70: {"collapsed": (False, AVAILABLE, []), "structure-unstable": (None, NO_PREDECESSOR, [])},
            71: {"collapsed": (False, AVAILABLE, []), "structure-unstable": (True, AVAILABLE, [])},
            72: {"collapsed": (False, AVAILABLE, []), "structure-unstable": (True, AVAILABLE, [])},
        }
        for step, wanted in expected.items():
            with self.subTest(fixed_step=step):
                frame = frame_by_step(self.labels, step)
                for name, summary in wanted.items():
                    self.assertEqual(predicate_summary(frame, name), summary, name)

    def test_outcome_unsettled_nonterminal(self) -> None:
        self.assertEqual(self.labels.outcome.outcome_class, OutcomeClass.UNSETTLED_NONTERMINAL)


class CollapseDisappearanceTests(unittest.TestCase):
    """M0a oracle fixture 8: support loss plus disappearance latches collapsed."""

    def setUp(self) -> None:
        self.labels = derive("collapse_disappearance")

    def test_frame_labels(self) -> None:
        expected = {
            80: {"collapsed": (False, AVAILABLE, []), "structure-unstable": (None, NO_PREDECESSOR, [])},
            81: {"collapsed": (False, AVAILABLE, []), "structure-unstable": (True, AVAILABLE, [])},
            # Collapsed latches at 82 with EMPTY evidence: the disappearance has no event.
            82: {"collapsed": (True, AVAILABLE, []), "structure-unstable": (True, AVAILABLE, [])},
        }
        for step, wanted in expected.items():
            with self.subTest(fixed_step=step):
                frame = frame_by_step(self.labels, step)
                for name, summary in wanted.items():
                    self.assertEqual(predicate_summary(frame, name), summary, name)


class PigTagsTests(unittest.TestCase):
    """M0a oracle fixture 9 (pig_tags): same-step pig_removed cluster clears all pigs."""

    def setUp(self) -> None:
        self.labels = derive("pig_tags")

    def test_cascade_interval(self) -> None:
        self.assertEqual(
            interval_summary(self.labels),
            [("cascade-active", 91, 92, HYPOTHESIS, ["event:00000001", "event:00000002", "event:00000003"])],
        )

    def test_frame_labels(self) -> None:
        state90 = frame_by_step(self.labels, 90)
        for name in ("cascade-active", "collapsed", "pigs-cleared", "steady-state"):
            self.assertEqual(predicate_summary(state90, name), (False, AVAILABLE, []), name)
        self.assertEqual(
            predicate_summary(state90, "structure-unstable"), (None, NO_PREDECESSOR, [])
        )
        self.assertEqual(active_names(state90), [])

        state91 = frame_by_step(self.labels, 91)
        self.assertEqual(
            predicate_summary(state91, "cascade-active"),
            (True, AVAILABLE, ["event:00000001", "event:00000002", "event:00000003"]),
        )
        self.assertEqual(
            predicate_summary(state91, "pigs-cleared"),
            (True, AVAILABLE, ["event:00000001", "event:00000002", "event:00000003"]),
        )
        self.assertEqual(predicate_summary(state91, "collapsed"), (False, AVAILABLE, []))

    def test_outcome_unsettled_nonterminal(self) -> None:
        self.assertEqual(self.labels.outcome.outcome_class, OutcomeClass.UNSETTLED_NONTERMINAL)

    def test_removed_pigs_cover_the_pinned_tag_set(self) -> None:
        capture = load_physics_capture(
            shot_dir("pig_tags") / "physics_state.jsonl",
            shot_dir("pig_tags") / "physics_events.jsonl",
        )
        state90 = next(state for state in capture.states if state.clock.fixed_step == 90)
        classes_by_entity = {str(node.entity_id): node.object_class for node in state90.nodes}
        removed = [
            str(event.participants[0])
            for event in capture.events
            if event.event_type.value == "pig_removed"
        ]
        self.assertEqual(len(removed), 3)
        self.assertEqual({classes_by_entity[entity] for entity in removed}, set(PIG_CLASS_SET))


class HeaderContractTests(unittest.TestCase):
    """M0a cross-cutting: header shape, record order, counts, digests, vocabulary."""

    def setUp(self) -> None:
        self.labels = derive("canonical_multistate")
        self.records = jsonl_records(self.labels)

    def test_record_order(self) -> None:
        self.assertEqual(
            [record["record_type"] for record in self.records],
            ["macro_label_header", "event_interval", "event_interval"]
            + ["frame_label"] * 6
            + ["shot_outcome"],
        )

    def test_header_exact_field_set(self) -> None:
        self.assertEqual(
            set(self.records[0].keys()),
            {
                "record_type",
                "schema_version",
                "capture_schema_version",
                "capture_id",
                "shot_id",
                "derivation_spec_version",
                "derivation_spec_digest",
                "event_clock",
                "macro_vocabulary",
                "pig_class_set",
                "sources",
                "state_count",
                "event_count",
                "interval_count",
                "frame_label_count",
            },
        )

    def test_header_versions_clock_and_digest(self) -> None:
        header = self.records[0]
        self.assertEqual(header["schema_version"], "physics_macro_labels_v1")
        self.assertEqual(header["capture_schema_version"], "physics_capture_v1")
        self.assertEqual(header["derivation_spec_version"], "macro_labels_derivation_v1")
        self.assertEqual(header["derivation_spec_digest"], derivation_spec_digest())
        self.assertEqual(
            header["event_clock"],
            {"occurrence_authority": "fixed_step", "render_frame_role": "provenance_only"},
        )

    def test_header_counts_match_records(self) -> None:
        header = self.records[0]
        frames = [record for record in self.records if record["record_type"] == "frame_label"]
        intervals = [record for record in self.records if record["record_type"] == "event_interval"]
        self.assertEqual(header["state_count"], 6)
        self.assertEqual(header["state_count"], len(frames))
        self.assertEqual(header["event_count"], 7)
        self.assertEqual(header["interval_count"], 2)
        self.assertEqual(header["interval_count"], len(intervals))
        self.assertEqual(header["frame_label_count"], 6)
        self.assertEqual(header["frame_label_count"], len(frames))

    def test_header_sources_bind_actual_fixture_digests(self) -> None:
        header = self.records[0]
        sources = header["sources"]
        self.assertEqual(sources["physics_state_path"], "physics_state.jsonl")
        self.assertEqual(sources["physics_events_path"], "physics_events.jsonl")
        self.assertEqual(
            sources["physics_state_sha256"],
            hashlib.sha256((shot_dir("canonical_multistate") / "physics_state.jsonl").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            sources["physics_events_sha256"],
            hashlib.sha256((shot_dir("canonical_multistate") / "physics_events.jsonl").read_bytes()).hexdigest(),
        )

    def test_header_vocabulary_and_pig_class_set(self) -> None:
        header = self.records[0]
        self.assertEqual(
            header["macro_vocabulary"],
            [
                {
                    "predicate": "cascade-active",
                    "absorbing": False,
                    "semantic_status": "hypothesis_pending_representative_validation",
                },
                {
                    "predicate": "collapsed",
                    "absorbing": True,
                    "semantic_status": "hypothesis_pending_representative_validation",
                },
                {
                    "predicate": "pigs-cleared",
                    "absorbing": True,
                    "semantic_status": "hypothesis_pending_representative_validation",
                },
                {"predicate": "steady-state", "absorbing": False, "semantic_status": "engine_verified"},
                {"predicate": "structure-unstable", "absorbing": False, "semantic_status": "engine_verified"},
            ],
        )
        self.assertEqual(header["pig_class_set"], ["PigBig", "PigMedium", "PigSmall"])


class NoMilestone0bContaminationTests(unittest.TestCase):
    """M0a cross-cutting: no oracle/KE/contact-threshold vocabulary anywhere."""

    BANNED = ("oracle", "kinetic", "threshold", "contact_activity", "active_contact")

    def _keys(self, value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from self._keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from self._keys(item)

    def test_no_milestone_0b_key_appears_in_any_fixture_derivation(self) -> None:
        for case in CASE_NAMES:
            with self.subTest(case=case):
                for record in jsonl_records(derive(case)):
                    for key in self._keys(record):
                        for banned in self.BANNED:
                            self.assertNotIn(banned, key)


class CanonicalBytesTests(unittest.TestCase):
    """M0a cross-cutting: canonical JSONL bytes, determinism, read round-trip."""

    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-macro-bytes-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)

    def test_to_jsonl_is_canonical_for_every_fixture(self) -> None:
        for case in CASE_NAMES:
            with self.subTest(case=case):
                text = derive(case).to_jsonl()
                self.assertTrue(text.endswith("\n"))
                self.assertFalse(text.endswith("\n\n"))
                self.assertNotIn("\r", text)
                for line in text[:-1].split("\n"):
                    self.assertEqual(
                        line,
                        json.dumps(json.loads(line), sort_keys=True, separators=(",", ":")),
                    )

    def test_repeated_derivation_is_byte_identical(self) -> None:
        for case in CASE_NAMES:
            with self.subTest(case=case):
                self.assertEqual(derive(case).to_jsonl(), derive(case).to_jsonl())

    def test_write_read_round_trip_preserves_bytes(self) -> None:
        shot = copy_shot("canonical_multistate", self.temporary)
        path = write_macro_labels(shot)
        self.assertEqual(read_macro_labels(path).to_jsonl(), path.read_text(encoding="utf-8"))


class AtomicWriteTests(unittest.TestCase):
    """M0a cross-cutting: atomic write leaves only the sidecar, no temp residue."""

    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-macro-write-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)

    def test_write_leaves_exactly_three_files(self) -> None:
        shot = copy_shot("same_step_cluster", self.temporary)
        write_macro_labels(shot)
        self.assertEqual(
            sorted(path.name for path in shot.iterdir()),
            ["physics_events.jsonl", "physics_macro_labels.jsonl", "physics_state.jsonl"],
        )
        self.assertEqual(list(shot.rglob("*.tmp")), [])


class FailClosedMutationTests(unittest.TestCase):
    """M0a cross-cutting: every contract drift is rejected by read and/or validate."""

    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-macro-mutate-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)
        self.shot = copy_shot("canonical_multistate", self.temporary)
        self.label_path = write_macro_labels(self.shot)

    def _records(self) -> list[dict]:
        return [json.loads(line) for line in self.label_path.read_text(encoding="utf-8").splitlines()]

    def _rewrite(self, records: list[dict]) -> None:
        self.label_path.write_text(
            "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )

    def _assert_read_and_validate_reject(self) -> None:
        with self.assertRaises(MacroLabelError):
            read_macro_labels(self.label_path)
        with self.assertRaises(MacroLabelError):
            validate_macro_labels(self.shot)

    def test_unknown_frame_label_field_is_rejected(self) -> None:
        records = self._records()
        frame = next(record for record in records if record["record_type"] == "frame_label")
        frame["unexpected_field"] = True
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_outcome_not_last_is_rejected(self) -> None:
        records = self._records()
        outcome = records.pop()
        records.insert(1, outcome)
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_header_frame_label_count_drift_is_rejected(self) -> None:
        records = self._records()
        records[0]["frame_label_count"] += 1
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_stale_source_digest_is_rejected_by_validate_only(self) -> None:
        records = self._records()
        records[0]["sources"]["physics_state_sha256"] = "0" * 64
        self._rewrite(records)
        read_macro_labels(self.label_path)  # strict parse still accepts the shape
        with self.assertRaises(MacroLabelError):
            validate_macro_labels(self.shot)

    def test_flipped_predicate_value_is_rejected(self) -> None:
        records = self._records()
        frames = [record for record in records if record["record_type"] == "frame_label"]
        frame = next(record for record in frames if record["fixed_step"] == 11)
        frame["predicates"]["steady-state"]["value"] = True
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_absorbing_reversion_is_rejected(self) -> None:
        records = self._records()
        frames = [record for record in records if record["record_type"] == "frame_label"]
        last = frames[-1]
        self.assertTrue(last["predicates"]["collapsed"]["value"])
        last["predicates"]["collapsed"]["value"] = False
        last["active_macro_states"] = [
            name for name in last["active_macro_states"] if name != "collapsed"
        ]
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_reordered_pig_class_set_is_rejected(self) -> None:
        records = self._records()
        records[0]["pig_class_set"] = ["PigSmall", "PigMedium", "PigBig"]
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_unsorted_citations_are_rejected(self) -> None:
        records = self._records()
        cascade = next(
            record
            for record in records
            if record["record_type"] == "event_interval" and record["interval_type"] == "cascade-active"
        )
        cascade["evidence"] = list(reversed(cascade["evidence"]))
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_missing_trailing_newline_is_rejected(self) -> None:
        text = self.label_path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.label_path.write_text(text[:-1], encoding="utf-8")
        self._assert_read_and_validate_reject()

    def test_unknown_availability_value_is_rejected(self) -> None:
        records = self._records()
        frames = [record for record in records if record["record_type"] == "frame_label"]
        frame = next(record for record in frames if record["fixed_step"] == 10)
        frame["predicates"]["structure-unstable"]["availability"] = "unavailable_pending"
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_duplicate_frame_label_identity_is_rejected(self) -> None:
        records = self._records()
        frame_indices = [
            index for index, record in enumerate(records) if record["record_type"] == "frame_label"
        ]
        records[frame_indices[-1]] = dict(records[frame_indices[0]])
        self._rewrite(records)
        self._assert_read_and_validate_reject()

    def test_permuted_event_intervals_are_rejected(self) -> None:
        # Swapping the two interval records keeps counts and every individual record
        # valid; only the canonical-order check can fire.
        records = self._records()
        interval_indices = [
            index for index, record in enumerate(records) if record["record_type"] == "event_interval"
        ]
        self.assertEqual(len(interval_indices), 2)
        first, second = interval_indices
        records[first], records[second] = records[second], records[first]
        self._rewrite(records)
        with self.assertRaises(MacroLabelError) as raised:
            read_macro_labels(self.label_path)
        self.assertIn("canonical order", str(raised.exception))
        with self.assertRaises(MacroLabelError):
            validate_macro_labels(self.shot)

    def test_permuted_frame_labels_are_rejected(self) -> None:
        # Swapping the frame labels at fixed steps 11 and 12 keeps counts unchanged
        # and identities unique; only the accepted-state-order check can fire.
        records = self._records()
        frame_indices = {
            record["fixed_step"]: index
            for index, record in enumerate(records)
            if record["record_type"] == "frame_label"
        }
        first, second = frame_indices[11], frame_indices[12]
        records[first], records[second] = records[second], records[first]
        self._rewrite(records)
        with self.assertRaises(MacroLabelError) as raised:
            read_macro_labels(self.label_path)
        self.assertIn("accepted state order", str(raised.exception))
        with self.assertRaises(MacroLabelError):
            validate_macro_labels(self.shot)


class DerivationRejectionTests(unittest.TestCase):
    """M0a cross-cutting: contradictory stability and render_frame non-joining."""

    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-macro-reject-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)

    def test_same_step_stable_entered_and_exited_is_rejected(self) -> None:
        shot = copy_shot("no_events", self.temporary)
        header = json.loads((shot / "physics_state.jsonl").read_text(encoding="utf-8").splitlines()[0])
        template = {
            "record_type": "event",
            "schema_version": "physics_capture_v1",
            "capture_id": header["capture_id"],
            "shot_id": "shot_001",
            "render_frame": 301,
            "render_time": 1.0 + 301 / 60.0,
            "fixed_step": 30,
            "fixed_time": 0.6,
            "coordinates": header["coordinates"],
            "participants": [],
            "payload": {"debounce_fixed_steps": 2},
        }
        entered = {**template, "sequence": 0, "event_id": "event:00000000", "event_type": "stable_entered"}
        exited = {**template, "sequence": 1, "event_id": "event:00000001", "event_type": "stable_exited"}
        (shot / "physics_events.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in (entered, exited)),
            encoding="utf-8",
        )
        # The frozen validator accepts this (entered ranks before exited at one step)...
        capture = load_physics_capture(shot / "physics_state.jsonl", shot / "physics_events.jsonl")
        self.assertEqual(len(capture.events), 2)
        # ...but the macro layer must refuse the contradictory cluster.
        with self.assertRaises(MacroLabelError):
            derive_macro_labels_for_shot(shot)

    def test_event_render_frame_never_joins(self) -> None:
        shot = copy_shot("canonical_multistate", self.temporary)
        event_path = shot / "physics_events.jsonl"
        rewritten = []
        for line in event_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record["render_frame"] = 99999
            rewritten.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        event_path.write_text("".join(line + "\n" for line in rewritten), encoding="utf-8")
        # Still a valid frozen capture...
        load_physics_capture(shot / "physics_state.jsonl", shot / "physics_events.jsonl")
        # ...and frame labels are byte-identical because occurrence authority is fixed_step.
        relabeled = derive_macro_labels_for_shot(shot)
        self.assertEqual(frame_label_lines(relabeled), frame_label_lines(derive("canonical_multistate")))


class DeriveCliTests(unittest.TestCase):
    """M0a CLI: deterministic tree writes, validate-only, tamper detection, exit codes."""

    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="novphy-macro-cli-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)

    def _run(self, argv: list[str]) -> tuple[int, dict | None]:
        from scripts.derive_physics_macro_labels import main

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = main(argv)
        text = stdout.getvalue()
        # Exit-2 usage errors print their JSON to stderr and leave stdout empty.
        return code, (json.loads(text) if text.strip() else None)

    @staticmethod
    def _tree_digests(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_writes_a_deterministic_tree_for_the_fixture_root(self) -> None:
        output_a = self.temporary / "out-a"
        code, report = self._run(
            ["--target", str(FIXTURE_ROOT), "--output-dir", str(output_a), "--json"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(report["shots_ok"], 9)
        self.assertEqual(report["shots_failed"], 0)
        tree_a = self._tree_digests(output_a)
        self.assertEqual(len(tree_a), 9)
        self.assertEqual(
            sorted(tree_a),
            sorted(f"{case}/shot_001/physics_macro_labels.jsonl" for case in CASE_NAMES),
        )
        output_b = self.temporary / "out-b"
        code, _ = self._run(
            ["--target", str(FIXTURE_ROOT), "--output-dir", str(output_b), "--json"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(tree_a, self._tree_digests(output_b))

    def test_validate_only_accepts_a_written_tree(self) -> None:
        output = self.temporary / "out"
        code, _ = self._run(["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--json"])
        self.assertEqual(code, 0)
        code, report = self._run(
            ["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--validate-only", "--json"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(report["shots_ok"], 9)

    def test_validate_only_reports_a_tampered_file(self) -> None:
        output = self.temporary / "out"
        code, _ = self._run(["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--json"])
        self.assertEqual(code, 0)
        target = output / "canonical_multistate" / "shot_001" / "physics_macro_labels.jsonl"
        records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        outcome = records[-1]
        self.assertEqual(outcome["record_type"], "shot_outcome")
        outcome["score"] = 50001
        target.write_text(
            "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
        code, report = self._run(
            ["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--validate-only", "--json"]
        )
        self.assertEqual(code, 1)
        self.assertEqual(report["shots_failed"], 1)
        self.assertIn("canonical_multistate", report["failures"][0]["shot"])

    def test_write_mode_without_output_dir_exits_2(self) -> None:
        code, report = self._run(["--target", str(FIXTURE_ROOT), "--json"])
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertEqual(list(self.temporary.rglob(MACRO_LABEL_SIDECAR)), [])

    def test_write_destination_parent_with_sidecars_is_refused(self) -> None:
        # A mirror shot dir that already holds frozen sidecars is a shot/cohort
        # directory, not a writable mirror tree.
        output = self.temporary / "out"
        staged = output / "pig_tags" / "shot_001"
        staged.mkdir(parents=True)
        for name in ("physics_state.jsonl", "physics_events.jsonl"):
            shutil.copy(shot_dir("pig_tags") / name, staged / name)
        code, report = self._run(
            ["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--json"]
        )
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertFalse((staged / MACRO_LABEL_SIDECAR).exists())

    def test_preflight_refusal_writes_nothing_for_any_shot(self) -> None:
        # Sidecars staged under exactly one mirror: the other eight shots would pass,
        # but the preflight refuses the whole run before any file is written.
        output = self.temporary / "out"
        staged = output / "no_events" / "shot_001"
        staged.mkdir(parents=True)
        for name in ("physics_state.jsonl", "physics_events.jsonl"):
            shutil.copy(shot_dir("no_events") / name, staged / name)
        code, _ = self._run(
            ["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--json"]
        )
        self.assertEqual(code, 2)
        self.assertEqual(
            [path for path in output.rglob(MACRO_LABEL_SIDECAR)],
            [],
        )

    def test_validate_only_without_output_dir_accepts_in_shot_labels(self) -> None:
        shot = copy_shot("pig_tags", self.temporary)
        write_macro_labels(shot)  # library call on a temp copy, not the CLI
        code, report = self._run(["--target", str(shot), "--validate-only", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(report["shots_ok"], 1)
        self.assertEqual(report["output_dir"], None)

    def test_output_dir_inside_active_cohort_is_refused(self) -> None:
        from scripts.derive_physics_macro_labels import ACTIVE_COHORT_DIR_NAME

        output = self.temporary / ACTIVE_COHORT_DIR_NAME / "labels"
        with self.assertRaises(SystemExit) as raised:
            self._run(["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--json"])
        self.assertIn("active cohort", str(raised.exception))
        self.assertFalse(output.exists())

    def _run_capturing(self, argv: list[str]) -> tuple[int, dict | None, str]:
        from scripts.derive_physics_macro_labels import main

        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        text = stdout.getvalue()
        return code, (json.loads(text) if text.strip() else None), stderr.getvalue()

    def test_sidecar_free_mirror_inside_a_real_cohort_is_refused(self) -> None:
        # The label parents under <cohort>/macro-label-mirror hold no sidecars, so
        # only the capture-tree containment guard can fire.
        cohort = self.temporary / "fake_cohort"
        staged = cohort / "train" / "episode_001" / "shot_001"
        staged.mkdir(parents=True)
        for name in ("physics_state.jsonl", "physics_events.jsonl"):
            shutil.copy(shot_dir("canonical_multistate") / name, staged / name)
        output = cohort / "macro-label-mirror"
        code, report, stderr = self._run_capturing(
            ["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--json"]
        )
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertIn("contains physics capture records", stderr)
        self.assertEqual(list(cohort.rglob(MACRO_LABEL_SIDECAR)), [])

    def test_deeply_nested_cohort_mirror_is_refused(self) -> None:
        # Four directory levels below the cohort root: beyond any fixed depth bound.
        cohort = self.temporary / "real-cohort"
        staged = cohort / "region" / "train" / "episode_001" / "shot_001"
        staged.mkdir(parents=True)
        for name in ("physics_state.jsonl", "physics_events.jsonl"):
            shutil.copy(shot_dir("canonical_multistate") / name, staged / name)
        output = cohort / "macro-label-mirror"
        code, report, stderr = self._run_capturing(
            ["--target", str(FIXTURE_ROOT), "--output-dir", str(output), "--json"]
        )
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertIn("contains physics capture records", stderr)
        self.assertEqual(list(cohort.rglob(MACRO_LABEL_SIDECAR)), [])

    def test_output_dir_outside_the_temporary_root_exits_2(self) -> None:
        code, report, stderr = self._run_capturing(
            ["--target", str(FIXTURE_ROOT), "--output-dir", "/nonexistent-novphy-root/out", "--json"]
        )
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertIn("temporary", stderr)
        self.assertFalse(Path("/nonexistent-novphy-root").exists())

    def test_rewriting_the_same_temporary_mirror_is_allowed(self) -> None:
        # The mirror tree holds only label files, no capture markers, so a second
        # write into the same temporary root passes every guard.
        argv = ["--target", str(FIXTURE_ROOT), "--output-dir", str(self.temporary / "out"), "--json"]
        code, _ = self._run(argv)
        self.assertEqual(code, 0)
        code, report = self._run(argv)
        self.assertEqual(code, 0)
        self.assertEqual(report["shots_ok"], 9)

    def test_unknown_target_exits_2(self) -> None:
        code, _ = self._run(
            ["--target", str(self.temporary / "does-not-exist"), "--output-dir", str(self.temporary / "o"), "--json"]
        )
        self.assertEqual(code, 2)

    def test_root_without_shots_exits_2(self) -> None:
        empty = self.temporary / "empty"
        empty.mkdir()
        code, _ = self._run(["--target", str(empty), "--output-dir", str(self.temporary / "o"), "--json"])
        self.assertEqual(code, 2)


class FrozenDataclassTests(unittest.TestCase):
    """M0a reader face: label dataclasses are immutable."""

    def test_label_dataclasses_are_frozen(self) -> None:
        labels = derive("canonical_multistate")
        frame = labels.frames[0]
        with self.assertRaises(FrozenInstanceError):
            frame.identity = frame.identity  # type: ignore[misc]
        predicate = frame.predicate(MacroPredicate.STEADY_STATE)
        with self.assertRaises(FrozenInstanceError):
            predicate.value = False  # type: ignore[misc]
        citation = labels.intervals[0].evidence[0]
        with self.assertRaises(FrozenInstanceError):
            citation.fixed_step = 0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
