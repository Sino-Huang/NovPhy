"""Deterministic engine-anchored macro/outcome labels over frozen physics_capture_v1.

This module implements the fixture-only Milestone 0a artifact family
`physics_macro_labels_v1`: per-state macro predicates (steady-state,
structure-unstable, cascade-active, collapsed, pigs-cleared), fixed-step event
intervals, and one shot-outcome record, all derived purely from validated
`physics_capture_v1` sidecars.

Clock contract (critical): event occurrence authority is `fixed_step`/`fixed_time`.
The producer stamps every buffered event with the serialization snapshot's
`render_frame`, so an event's `render_frame` is provenance only.  Nothing here ever
joins or groups events to states by event `render_frame`; events are consumed as
atomic fixed-step clusters and projected onto states by fixed-step bracketing (all
clusters with `fixed_step <= state.fixed_step`).  `debounce_fixed_steps` is never
converted into render frames.

No kinetic-energy or contact-activity threshold appears anywhere in this layer (that
oracle gate is Milestone 0b and out of scope), and no learned or vision-derived
signal is consulted.  Semantic status is explicit: steady-state and
structure-unstable are `engine_verified`, while the cascade-active termination rule,
the collapsed rule, and the pigs-cleared taxonomy coverage are hypotheses pending
representative validation.

An unavailable predicate keeps `value: null` with an explicit availability reason;
it is never silently converted to false for training.

The artifact binds the SHA-256 of both frozen source sidecars, serializes to
canonical JSONL (ASCII-sorted keys, compact separators, finite numbers only, LF line
endings, exactly one trailing newline), writes atomically, and validates fail-closed
by strict parsing plus re-derivation byte comparison.  The frozen capture contract
is not touched.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, unique
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Final, TypeAlias

from scripts.physics_capture_contract import (
    EVENT_SIDECAR,
    STATE_SIDECAR,
    load_physics_capture,
)
from scripts.physics_capture_types import (
    EventRecord,
    EventType,
    PhysicsCapture,
)

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

MACRO_LABEL_SCHEMA_VERSION: Final = "physics_macro_labels_v1"
CAPTURE_SCHEMA_VERSION: Final = "physics_capture_v1"
MACRO_LABEL_SIDECAR: Final = "physics_macro_labels.jsonl"
DERIVATION_SPEC_VERSION: Final = "macro_labels_derivation_v1"

#: Versioned closed set of Unity tags the exporter uses for pigs.  Ordered and
#: ASCII-sorted; any drift (missing, reordered, duplicated, different) is a hard
#: rejection, never a silent relabel.
PIG_CLASS_SET: Final = ("PigBig", "PigMedium", "PigSmall")

#: Event types that can open or sustain a cascade, ordered by value.
CAUSAL_EVENT_TYPES: Final = (
    EventType.COLLISION,
    EventType.ENTITY_DESTROYED,
    EventType.EXPLOSION,
    EventType.PIG_REMOVED,
)


@unique
class MacroPredicate(StrEnum):
    """Per-state macro predicates, declared in ASCII-sorted value order."""

    CASCADE_ACTIVE = "cascade-active"
    COLLAPSED = "collapsed"
    PIGS_CLEARED = "pigs-cleared"
    STEADY_STATE = "steady-state"
    STRUCTURE_UNSTABLE = "structure-unstable"


#: Absorbing predicates: once true they remain true by vocabulary design.
ABSORBING_PREDICATES: Final = (MacroPredicate.COLLAPSED, MacroPredicate.PIGS_CLEARED)


@unique
class Availability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE_NO_PREDECESSOR = "unavailable_no_predecessor"
    UNAVAILABLE_INSUFFICIENT_STATE_EVIDENCE = "unavailable_insufficient_state_evidence"


@unique
class SemanticStatus(StrEnum):
    ENGINE_VERIFIED = "engine_verified"
    HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION = "hypothesis_pending_representative_validation"


@unique
class OutcomeClass(StrEnum):
    CLEARED = "cleared"
    FAILED = "failed"
    SETTLED_NONTERMINAL = "settled_nonterminal"
    UNSETTLED_NONTERMINAL = "unsettled_nonterminal"


@unique
class TerminalEquilibrium(StrEnum):
    STABLE_TERMINAL = "stable_terminal"
    NOT_OBSERVED = "not_observed"


#: Predicate -> semantic status, ordered by predicate value.  steady-state and
#: structure-unstable are engine-verified; cascade-active, collapsed, and
#: pigs-cleared are hypotheses pending representative validation.
PREDICATE_SEMANTIC_STATUS: Final = (
    (MacroPredicate.CASCADE_ACTIVE, SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION),
    (MacroPredicate.COLLAPSED, SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION),
    (MacroPredicate.PIGS_CLEARED, SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION),
    (MacroPredicate.STEADY_STATE, SemanticStatus.ENGINE_VERIFIED),
    (MacroPredicate.STRUCTURE_UNSTABLE, SemanticStatus.ENGINE_VERIFIED),
)

_PREDICATE_STATUS: Final = dict(PREDICATE_SEMANTIC_STATUS)

#: Explicit clock declaration carried by the header and the derivation spec.
EVENT_CLOCK_JSON: Final = {
    "occurrence_authority": "fixed_step",
    "render_frame_role": "provenance_only",
}


@dataclass(frozen=True, slots=True)
class MacroLabelError(ValueError):
    """Raised when macro labels are absent, stale, malformed, or disagree with their sidecars."""

    location: str
    detail: str

    def __str__(self) -> str:
        return f"invalid macro labels at {self.location}: {self.detail}"


@dataclass(frozen=True, slots=True)
class EventCitation:
    """Deterministic identity of one source event: (capture_id, shot_id, event_sequence, event_id, fixed_step)."""

    capture_id: str
    shot_id: str
    event_sequence: int
    event_id: str
    fixed_step: int

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "shot_id": self.shot_id,
            "event_sequence": self.event_sequence,
            "event_id": self.event_id,
            "fixed_step": self.fixed_step,
        }


@dataclass(frozen=True, slots=True)
class PredicateLabel:
    """One macro predicate value with its availability and ordered evidence citations."""

    value: bool | None
    availability: Availability
    evidence: tuple[EventCitation, ...]

    def to_json(self) -> JsonObject:
        return {
            "value": self.value,
            "availability": self.availability.value,
            "evidence": [citation.to_json() for citation in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class StateIdentity:
    """State-label identity key: (capture_id, shot_id, state_sequence, render_frame, fixed_step, rgb_relative_path)."""

    capture_id: str
    shot_id: str
    state_sequence: int
    render_frame: int
    fixed_step: int
    rgb_relative_path: str

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "shot_id": self.shot_id,
            "state_sequence": self.state_sequence,
            "render_frame": self.render_frame,
            "fixed_step": self.fixed_step,
            "rgb_relative_path": self.rgb_relative_path,
        }


@dataclass(frozen=True, slots=True)
class EventInterval:
    """Half-open fixed-step interval; `interval_type` is "steady-state" or "cascade-active"."""

    interval_type: str
    start_fixed_step: int
    end_fixed_step: int | None
    semantic_status: SemanticStatus
    evidence: tuple[EventCitation, ...]

    def contains(self, fixed_step: int) -> bool:
        return self.start_fixed_step <= fixed_step and (
            self.end_fixed_step is None or fixed_step < self.end_fixed_step
        )

    def to_json(self) -> JsonObject:
        return {
            "record_type": "event_interval",
            "interval_type": self.interval_type,
            "start_fixed_step": self.start_fixed_step,
            "end_fixed_step": self.end_fixed_step,
            "semantic_status": self.semantic_status.value,
            "evidence": [citation.to_json() for citation in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class MacroFrameLabel:
    """All five macro predicates bound to one accepted state and its RGB frame."""

    identity: StateIdentity
    predicates: tuple[tuple[MacroPredicate, PredicateLabel], ...]

    @property
    def active_macro_states(self) -> tuple[MacroPredicate, ...]:
        """Sorted predicate names whose value is true."""
        return tuple(predicate for predicate, label in self.predicates if label.value is True)

    def predicate(self, name: MacroPredicate) -> PredicateLabel:
        for predicate, label in self.predicates:
            if predicate is name:
                return label
        raise KeyError(name)

    def to_json(self) -> JsonObject:
        return {
            "record_type": "frame_label",
            "capture_id": self.identity.capture_id,
            "shot_id": self.identity.shot_id,
            "state_sequence": self.identity.state_sequence,
            "render_frame": self.identity.render_frame,
            "fixed_step": self.identity.fixed_step,
            "rgb_relative_path": self.identity.rgb_relative_path,
            "predicates": {predicate.value: label.to_json() for predicate, label in self.predicates},
            "active_macro_states": [predicate.value for predicate in self.active_macro_states],
        }


@dataclass(frozen=True, slots=True)
class ShotOutcomeLabel:
    """Shot-level outcome with terminal-event provenance and equilibrium observation."""

    outcome_class: OutcomeClass
    score: int | None
    reason: str | None
    terminal_event: EventCitation | None
    terminal_equilibrium: TerminalEquilibrium
    terminal_state: StateIdentity | None
    semantic_status: SemanticStatus

    def to_json(self) -> JsonObject:
        return {
            "record_type": "shot_outcome",
            "outcome_class": self.outcome_class.value,
            "score": self.score,
            "reason": self.reason,
            "terminal_event": None if self.terminal_event is None else self.terminal_event.to_json(),
            "terminal_equilibrium": self.terminal_equilibrium.value,
            "terminal_state": None if self.terminal_state is None else self.terminal_state.to_json(),
            "semantic_status": self.semantic_status.value,
        }


@dataclass(frozen=True, slots=True)
class MacroLabels:
    """All macro/outcome labels derived from one accepted shot.

    `event_count` is the number of source event records.  It travels with the labels
    because the pinned header must declare it and it is not recoverable from the
    citations: uncited events (for example `bird_exhausted` or non-onset collisions)
    leave no trace elsewhere in the artifact.
    """

    capture_id: str
    shot_id: str
    state_sha256: str
    events_sha256: str
    event_count: int
    intervals: tuple[EventInterval, ...]
    frames: tuple[MacroFrameLabel, ...]
    outcome: ShotOutcomeLabel

    def header_json(self) -> JsonObject:
        return {
            "record_type": "macro_label_header",
            "schema_version": MACRO_LABEL_SCHEMA_VERSION,
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "capture_id": self.capture_id,
            "shot_id": self.shot_id,
            "derivation_spec_version": DERIVATION_SPEC_VERSION,
            "derivation_spec_digest": derivation_spec_digest(),
            "event_clock": {**EVENT_CLOCK_JSON},
            "macro_vocabulary": _macro_vocabulary_json(),
            "pig_class_set": list(PIG_CLASS_SET),
            "sources": {
                "physics_state_path": STATE_SIDECAR,
                "physics_state_sha256": self.state_sha256,
                "physics_events_path": EVENT_SIDECAR,
                "physics_events_sha256": self.events_sha256,
            },
            "state_count": len(self.frames),
            "event_count": self.event_count,
            "interval_count": len(self.intervals),
            "frame_label_count": len(self.frames),
        }

    def to_jsonl(self) -> str:
        records: list[JsonObject] = [self.header_json()]
        records.extend(interval.to_json() for interval in self.intervals)
        records.extend(frame.to_json() for frame in self.frames)
        records.append(self.outcome.to_json())
        return "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for record in records
        )


def _macro_vocabulary_json() -> list[JsonObject]:
    return [
        {
            "predicate": predicate.value,
            "absorbing": predicate in ABSORBING_PREDICATES,
            "semantic_status": status.value,
        }
        for predicate, status in PREDICATE_SEMANTIC_STATUS
    ]


def derivation_spec_json() -> JsonObject:
    """Return the fixed canonical description of `macro_labels_derivation_v1`.

    Deterministic module-level constant: no timestamps, no paths, no environment
    input, so identical module versions always produce identical bytes.
    """
    return {
        "version": DERIVATION_SPEC_VERSION,
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "macro_vocabulary": _macro_vocabulary_json(),
        "pig_class_set": list(PIG_CLASS_SET),
        "causal_event_set": sorted(event_type.value for event_type in CAUSAL_EVENT_TYPES),
        "event_clock": {**EVENT_CLOCK_JSON},
        "interval_semantics": "half_open_fixed_step",
        "cascade_termination": "min(first_later_stable_entered,last_causal_fixed_step+1)",
        "steady_pre_launch_rule": "state_fixed_step_before_bird_launched_is_steady",
        "projection_rule": "clusters_with_fixed_step_lte_state",
    }


def derivation_spec_digest() -> str:
    """SHA-256 of the canonical JSON of `derivation_spec_json()`."""
    encoded = json.dumps(derivation_spec_json(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _citation(event: EventRecord) -> EventCitation:
    return EventCitation(
        capture_id=str(event.clock.capture_id),
        shot_id=str(event.clock.shot_id),
        event_sequence=event.clock.sequence,
        event_id=str(event.event_id),
        fixed_step=event.clock.fixed_step,
    )


def _order_citations(citations: Iterable[EventCitation]) -> tuple[EventCitation, ...]:
    """Canonical evidence order: sorted by (fixed_step, event_sequence), deduplicated."""
    return tuple(sorted(set(citations), key=lambda citation: (citation.fixed_step, citation.event_sequence)))


def _clusters(events: tuple[EventRecord, ...]) -> tuple[tuple[EventRecord, ...], ...]:
    """Group events into atomic fixed-step clusters (events are fixed-step ordered)."""
    clusters: list[list[EventRecord]] = []
    for event in events:
        if clusters and clusters[-1][0].clock.fixed_step == event.clock.fixed_step:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return tuple(tuple(cluster) for cluster in clusters)


def _steady_intervals(events: tuple[EventRecord, ...]) -> tuple[EventInterval, ...]:
    """Half-open [stable_entered, stable_exited) intervals over fixed steps.

    The frozen validator guarantees alternating stability transitions and at most one
    stability event per type per cluster.  A cluster holding both `stable_entered`
    and `stable_exited` is allowed by the frozen validator's ordering rule, so the
    macro layer rejects it here.  A leading `stable_exited` with no open interval
    closes nothing.  A trailing `stable_entered` stays open (end None).
    """
    intervals: list[EventInterval] = []
    opened: EventRecord | None = None
    for cluster in _clusters(events):
        entered = next((event for event in cluster if event.event_type is EventType.STABLE_ENTERED), None)
        exited = next((event for event in cluster if event.event_type is EventType.STABLE_EXITED), None)
        if entered is not None and exited is not None:
            raise MacroLabelError(
                f"fixed_step {cluster[0].clock.fixed_step}",
                "one fixed-step cluster must not contain both stable_entered and stable_exited",
            )
        if entered is not None:
            opened = entered
        elif exited is not None and opened is not None:
            intervals.append(
                EventInterval(
                    interval_type=MacroPredicate.STEADY_STATE.value,
                    start_fixed_step=opened.clock.fixed_step,
                    end_fixed_step=exited.clock.fixed_step,
                    semantic_status=SemanticStatus.ENGINE_VERIFIED,
                    evidence=_order_citations((_citation(opened), _citation(exited))),
                )
            )
            opened = None
    if opened is not None:
        intervals.append(
            EventInterval(
                interval_type=MacroPredicate.STEADY_STATE.value,
                start_fixed_step=opened.clock.fixed_step,
                end_fixed_step=None,
                semantic_status=SemanticStatus.ENGINE_VERIFIED,
                evidence=(_citation(opened),),
            )
        )
    return tuple(intervals)


def _cascade_interval(
    clusters: tuple[tuple[EventRecord, ...], ...],
    launch_step: int | None,
) -> EventInterval | None:
    """The single [onset, termination) cascade interval, or None.

    Onset is the first causal cluster at or after the launch step; without a launch
    or any causal cluster at step >= launch there is no cascade interval.
    Termination is the earlier of the first later stable_entered cluster and one
    fixed step after the last causal cluster.  Evidence cites the causal events of
    the onset cluster plus the termination witness: the stable_entered event when it
    bounds the interval, otherwise the causal events of the final causal cluster.
    """
    if launch_step is None:
        return None
    causal_clusters = [
        cluster
        for cluster in clusters
        if cluster[0].clock.fixed_step >= launch_step
        and any(event.event_type in CAUSAL_EVENT_TYPES for event in cluster)
    ]
    if not causal_clusters:
        return None
    onset_cluster = causal_clusters[0]
    final_cluster = causal_clusters[-1]
    onset = onset_cluster[0].clock.fixed_step
    last_causal = final_cluster[0].clock.fixed_step
    later_stable = next(
        (
            event
            for cluster in clusters
            if cluster[0].clock.fixed_step > onset
            for event in cluster
            if event.event_type is EventType.STABLE_ENTERED
        ),
        None,
    )
    end = last_causal + 1 if later_stable is None else min(later_stable.clock.fixed_step, last_causal + 1)
    evidence = [_citation(event) for event in onset_cluster if event.event_type in CAUSAL_EVENT_TYPES]
    if later_stable is not None and later_stable.clock.fixed_step <= last_causal + 1:
        evidence.append(_citation(later_stable))
    else:
        evidence.extend(_citation(event) for event in final_cluster if event.event_type in CAUSAL_EVENT_TYPES)
    return EventInterval(
        interval_type=MacroPredicate.CASCADE_ACTIVE.value,
        start_fixed_step=onset,
        end_fixed_step=end,
        semantic_status=SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION,
        evidence=_order_citations(evidence),
    )


def derive_macro_labels(capture: PhysicsCapture, *, state_sha256: str, events_sha256: str) -> MacroLabels:
    """Derive macro/outcome labels from a parsed, frozen-validator-accepted capture.

    Pure: the same capture and digests always produce identical records and bytes.
    The caller is responsible for having validated the capture (`load_physics_capture`
    does); derivation consumes only documented frozen fields.
    """
    capture_id = str(capture.header.clock.capture_id)
    shot_id = str(capture.header.clock.shot_id)
    events = capture.events
    states = capture.states

    clusters = _clusters(events)
    steady_intervals = _steady_intervals(events)
    launch = next((event for event in events if event.event_type is EventType.BIRD_LAUNCHED), None)
    launch_step = None if launch is None else launch.clock.fixed_step
    cascade = _cascade_interval(clusters, launch_step)
    cascade_intervals = () if cascade is None else (cascade,)
    intervals = tuple(
        sorted(
            (*steady_intervals, *cascade_intervals),
            key=lambda interval: (interval.start_fixed_step, interval.interval_type),
        )
    )

    frames: list[MacroFrameLabel] = []
    destruction_by_entity: dict[str, list[EventCitation]] = {}
    pig_removal_by_entity: dict[str, EventCitation] = {}
    event_index = 0
    previous_supports: frozenset[tuple[str, str]] | None = None
    supported_seen: set[str] = set()
    nodes_seen: set[str] = set()
    pigs_seen: set[str] = set()
    collapsed_latched = False
    collapsed_evidence: tuple[EventCitation, ...] = ()
    pigs_latched = False
    pigs_evidence: tuple[EventCitation, ...] = ()
    multi_state = len(states) >= 2

    for index, state in enumerate(states):
        step = state.clock.fixed_step

        # Consume every event cluster with fixed_step <= S before evaluating at S.
        while event_index < len(events) and events[event_index].clock.fixed_step <= step:
            event = events[event_index]
            if event.event_type in (EventType.ENTITY_DESTROYED, EventType.PIG_REMOVED):
                citation = _citation(event)
                for participant in event.participants:
                    destruction_by_entity.setdefault(str(participant), []).append(citation)
                    if event.event_type is EventType.PIG_REMOVED:
                        pig_removal_by_entity[str(participant)] = citation
            event_index += 1

        nodes_now = {str(node.entity_id) for node in state.nodes}
        supported_now = {str(edge.supported_id) for edge in state.support_edges}
        supports = frozenset((str(edge.supporter_id), str(edge.supported_id)) for edge in state.support_edges)
        pigs_now = {str(node.entity_id) for node in state.nodes if node.object_class in PIG_CLASS_SET}

        # steady-state: inside a steady interval, or before the launch step.
        containing = next((interval for interval in steady_intervals if interval.contains(step)), None)
        pre_launch = launch_step is not None and step < launch_step
        steady_value = containing is not None or pre_launch
        if containing is not None:
            steady_evidence = containing.evidence
        elif pre_launch and launch is not None:
            steady_evidence = (_citation(launch),)
        else:
            steady_evidence = ()
        steady_label = PredicateLabel(steady_value, Availability.AVAILABLE, steady_evidence)

        # structure-unstable: not steady and the directed support-edge set changed.
        if index == 0:
            unstable_label = PredicateLabel(None, Availability.UNAVAILABLE_NO_PREDECESSOR, ())
        else:
            unstable_value = (not steady_value) and supports != previous_supports
            unstable_label = PredicateLabel(unstable_value, Availability.AVAILABLE, ())

        # cascade-active: inside the cascade interval.
        cascade_value = cascade is not None and cascade.contains(step)
        cascade_label = PredicateLabel(
            cascade_value,
            Availability.AVAILABLE,
            cascade.evidence if cascade_value and cascade is not None else (),
        )

        # collapsed (absorbing): a previously supported entity lost all incoming
        # support AND was destroyed/removed by S or disappeared from the state.
        if not multi_state:
            collapsed_label = PredicateLabel(None, Availability.UNAVAILABLE_INSUFFICIENT_STATE_EVIDENCE, ())
        else:
            if not collapsed_latched:
                qualifying = [
                    candidate
                    for candidate in sorted(supported_seen)
                    if candidate not in supported_now
                    and (
                        bool(destruction_by_entity.get(candidate))
                        or (candidate in nodes_seen and candidate not in nodes_now)
                    )
                ]
                if qualifying:
                    collapsed_latched = True
                    collapsed_evidence = _order_citations(
                        citation
                        for candidate in qualifying
                        for citation in destruction_by_entity.get(candidate, [])
                    )
            collapsed_label = PredicateLabel(
                collapsed_latched,
                Availability.AVAILABLE,
                collapsed_evidence if collapsed_latched else (),
            )

        # pigs-cleared (absorbing): no pig node remains and every pig identity
        # observed in an earlier accepted state is pig_removed by S.
        if not pigs_latched and not pigs_now and pigs_seen and all(
            pig in pig_removal_by_entity for pig in pigs_seen
        ):
            pigs_latched = True
            pigs_evidence = _order_citations(pig_removal_by_entity[pig] for pig in pigs_seen)
        pigs_label = PredicateLabel(
            pigs_latched,
            Availability.AVAILABLE,
            pigs_evidence if pigs_latched else (),
        )

        identity = StateIdentity(
            capture_id=capture_id,
            shot_id=shot_id,
            state_sequence=state.clock.sequence,
            render_frame=state.clock.render_frame,
            fixed_step=step,
            rgb_relative_path=state.rgb_frame.relative_path,
        )
        labels_by_predicate = {
            MacroPredicate.CASCADE_ACTIVE: cascade_label,
            MacroPredicate.COLLAPSED: collapsed_label,
            MacroPredicate.PIGS_CLEARED: pigs_label,
            MacroPredicate.STEADY_STATE: steady_label,
            MacroPredicate.STRUCTURE_UNSTABLE: unstable_label,
        }
        predicates = tuple(
            sorted(labels_by_predicate.items(), key=lambda item: item[0].value)
        )
        frames.append(MacroFrameLabel(identity=identity, predicates=predicates))

        # Candidate pools for state i+1 cover states j <= i.
        supported_seen.update(str(edge.supported_id) for edge in state.support_edges)
        nodes_seen.update(str(node.entity_id) for node in state.nodes)
        pigs_seen.update(pigs_now)
        previous_supports = supports

    outcome = _derive_outcome(events, frames)

    return MacroLabels(
        capture_id=capture_id,
        shot_id=shot_id,
        state_sha256=state_sha256,
        events_sha256=events_sha256,
        event_count=len(events),
        intervals=intervals,
        frames=tuple(frames),
        outcome=outcome,
    )


def _derive_outcome(
    events: tuple[EventRecord, ...],
    frames: list[MacroFrameLabel],
) -> ShotOutcomeLabel:
    """One shot-level outcome record; engine_verified in all classes."""
    terminal = next(
        (event for event in events if event.event_type in (EventType.LEVEL_CLEARED, EventType.LEVEL_FAILED)),
        None,
    )
    if terminal is not None:
        payload = json.loads(terminal.payload_json)
        projection = next(
            (frame for frame in frames if frame.identity.fixed_step >= terminal.clock.fixed_step),
            None,
        )
        terminal_state = None if projection is None else projection.identity
        equilibrium = (
            TerminalEquilibrium.STABLE_TERMINAL
            if projection is not None
            and projection.predicate(MacroPredicate.STEADY_STATE).value is True
            else TerminalEquilibrium.NOT_OBSERVED
        )
        terminal_citation = _citation(terminal)
        if terminal.event_type is EventType.LEVEL_CLEARED:
            return ShotOutcomeLabel(
                outcome_class=OutcomeClass.CLEARED,
                score=payload["score"],
                reason=None,
                terminal_event=terminal_citation,
                terminal_equilibrium=equilibrium,
                terminal_state=terminal_state,
                semantic_status=SemanticStatus.ENGINE_VERIFIED,
            )
        return ShotOutcomeLabel(
            outcome_class=OutcomeClass.FAILED,
            score=None,
            reason=payload["reason"],
            terminal_event=terminal_citation,
            terminal_equilibrium=equilibrium,
            terminal_state=terminal_state,
            semantic_status=SemanticStatus.ENGINE_VERIFIED,
        )
    outcome_class = (
        OutcomeClass.SETTLED_NONTERMINAL
        if frames and frames[-1].predicate(MacroPredicate.STEADY_STATE).value is True
        else OutcomeClass.UNSETTLED_NONTERMINAL
    )
    return ShotOutcomeLabel(
        outcome_class=outcome_class,
        score=None,
        reason=None,
        terminal_event=None,
        terminal_equilibrium=TerminalEquilibrium.NOT_OBSERVED,
        terminal_state=None,
        semantic_status=SemanticStatus.ENGINE_VERIFIED,
    )


def derive_macro_labels_for_shot(shot_dir: Path) -> MacroLabels:
    """Load a shot's frozen sidecars and derive its labels, binding the source digests."""
    state_path = shot_dir / STATE_SIDECAR
    event_path = shot_dir / EVENT_SIDECAR
    if not state_path.is_file() or not event_path.is_file():
        raise MacroLabelError(str(shot_dir), "physics capture sidecars are missing")
    capture = load_physics_capture(state_path, event_path)
    return derive_macro_labels(
        capture,
        state_sha256=_sha256_file(state_path),
        events_sha256=_sha256_file(event_path),
    )


def write_macro_label_file(labels: MacroLabels, destination: Path) -> Path:
    """Atomically write canonical label bytes beside `destination`'s directory.

    Writes a random same-directory temporary file, flushes, fsyncs, then renames.
    Temporary residue is removed on failure.  Never touches the frozen sidecars,
    `frames/`, or `metadata.json`.
    """
    temporary = destination.parent / f".{MACRO_LABEL_SIDECAR}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(labels.to_jsonl())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise MacroLabelError(str(destination), "macro labels could not be written") from error
    return destination


def write_macro_labels(shot_dir: Path) -> Path:
    """Derive and atomically write a shot's `physics_macro_labels_v1` sidecar."""
    labels = derive_macro_labels_for_shot(shot_dir)
    return write_macro_label_file(labels, shot_dir / MACRO_LABEL_SIDECAR)


HEADER_FIELDS: Final = frozenset((
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
))
EVENT_CLOCK_FIELDS: Final = frozenset(("occurrence_authority", "render_frame_role"))
VOCABULARY_FIELDS: Final = frozenset(("predicate", "absorbing", "semantic_status"))
SOURCE_FIELDS: Final = frozenset((
    "physics_state_path",
    "physics_state_sha256",
    "physics_events_path",
    "physics_events_sha256",
))
CITATION_FIELDS: Final = frozenset(("capture_id", "shot_id", "event_sequence", "event_id", "fixed_step"))
STATE_IDENTITY_FIELDS: Final = frozenset((
    "capture_id",
    "shot_id",
    "state_sequence",
    "render_frame",
    "fixed_step",
    "rgb_relative_path",
))
PREDICATE_LABEL_FIELDS: Final = frozenset(("value", "availability", "evidence"))
INTERVAL_FIELDS: Final = frozenset((
    "record_type",
    "interval_type",
    "start_fixed_step",
    "end_fixed_step",
    "semantic_status",
    "evidence",
))
FRAME_FIELDS: Final = frozenset((
    "record_type",
    "capture_id",
    "shot_id",
    "state_sequence",
    "render_frame",
    "fixed_step",
    "rgb_relative_path",
    "predicates",
    "active_macro_states",
))
OUTCOME_FIELDS: Final = frozenset((
    "record_type",
    "outcome_class",
    "score",
    "reason",
    "terminal_event",
    "terminal_equilibrium",
    "terminal_state",
    "semantic_status",
))

_DIGEST_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class _ParsedHeader:
    capture_id: str
    shot_id: str
    state_sha256: str
    events_sha256: str
    state_count: int
    event_count: int
    interval_count: int
    frame_label_count: int


def _expect_fields(record: JsonObject, fields: frozenset[str], location: str) -> None:
    unknown = record.keys() - fields
    if unknown:
        raise MacroLabelError(location, f"unknown field: {min(unknown)}")
    missing = fields - record.keys()
    if missing:
        raise MacroLabelError(location, f"missing field: {min(missing)}")


def _expect_string(record: JsonObject, field: str, location: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value:
        raise MacroLabelError(location, f"{field} must be a non-empty string")
    return value


def _expect_count(record: JsonObject, field: str, location: str) -> int:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MacroLabelError(location, f"{field} must be a nonnegative integer")
    return value


def _expect_digest(record: JsonObject, field: str, location: str) -> str:
    value = record[field]
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise MacroLabelError(location, f"{field} must be 64 lowercase hex characters")
    return value


def _reject_constant(constant: str) -> None:
    raise ValueError(f"non-finite number: {constant}")


def _parse_record(line: str, location: str) -> JsonObject:
    try:
        value: JsonValue = json.loads(line, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise MacroLabelError(location, f"record is not valid JSON: {error.msg}") from error
    except ValueError as error:
        raise MacroLabelError(location, str(error)) from error
    if not isinstance(value, dict):
        raise MacroLabelError(location, "record must be an object")
    return value


def _parse_citation(value: JsonValue, location: str) -> EventCitation:
    if not isinstance(value, dict):
        raise MacroLabelError(location, "citation must be an object")
    _expect_fields(value, CITATION_FIELDS, location)
    return EventCitation(
        capture_id=_expect_string(value, "capture_id", location),
        shot_id=_expect_string(value, "shot_id", location),
        event_sequence=_expect_count(value, "event_sequence", location),
        event_id=_expect_string(value, "event_id", location),
        fixed_step=_expect_count(value, "fixed_step", location),
    )


def _parse_evidence(value: JsonValue, location: str) -> tuple[EventCitation, ...]:
    if not isinstance(value, list):
        raise MacroLabelError(location, "evidence must be an array")
    citations = tuple(_parse_citation(item, location) for item in value)
    keys = tuple((citation.fixed_step, citation.event_sequence) for citation in citations)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise MacroLabelError(
            location,
            "evidence citations must be sorted by (fixed_step, event_sequence) without duplicates",
        )
    return citations


def _parse_state_identity(value: JsonValue, location: str) -> StateIdentity:
    if not isinstance(value, dict):
        raise MacroLabelError(location, "state identity must be an object")
    _expect_fields(value, STATE_IDENTITY_FIELDS, location)
    return StateIdentity(
        capture_id=_expect_string(value, "capture_id", location),
        shot_id=_expect_string(value, "shot_id", location),
        state_sequence=_expect_count(value, "state_sequence", location),
        render_frame=_expect_count(value, "render_frame", location),
        fixed_step=_expect_count(value, "fixed_step", location),
        rgb_relative_path=_expect_string(value, "rgb_relative_path", location),
    )


def _parse_predicate_label(value: JsonValue, location: str) -> PredicateLabel:
    if not isinstance(value, dict):
        raise MacroLabelError(location, "predicate label must be an object")
    _expect_fields(value, PREDICATE_LABEL_FIELDS, location)
    raw_value = value["value"]
    if raw_value is not None and not isinstance(raw_value, bool):
        raise MacroLabelError(location, "predicate value must be true, false, or null")
    try:
        availability = Availability(value["availability"])
    except ValueError as error:
        raise MacroLabelError(location, "unknown availability value") from error
    if (raw_value is None) == (availability is Availability.AVAILABLE):
        raise MacroLabelError(
            location,
            "a null value requires an unavailable availability and a non-null value requires available",
        )
    evidence = _parse_evidence(value["evidence"], location)
    return PredicateLabel(raw_value, availability, evidence)


def _parse_vocabulary(value: JsonValue, location: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise MacroLabelError(location, "macro_vocabulary must be an array")
    entries: list[JsonObject] = []
    for item in value:
        if not isinstance(item, dict):
            raise MacroLabelError(location, "macro_vocabulary entries must be objects")
        _expect_fields(item, VOCABULARY_FIELDS, location)
        predicate = item["predicate"]
        absorbing = item["absorbing"]
        status = item["semantic_status"]
        if not isinstance(predicate, str) or not isinstance(absorbing, bool) or not isinstance(status, str):
            raise MacroLabelError(location, "macro_vocabulary entry fields have wrong types")
        entries.append({"predicate": predicate, "absorbing": absorbing, "semantic_status": status})
    return entries


def _parse_header(record: JsonObject, location: str) -> _ParsedHeader:
    _expect_fields(record, HEADER_FIELDS, location)
    if record["schema_version"] != MACRO_LABEL_SCHEMA_VERSION:
        raise MacroLabelError(location, "unsupported macro label schema version")
    if record["capture_schema_version"] != CAPTURE_SCHEMA_VERSION:
        raise MacroLabelError(location, "unsupported capture schema version")
    capture_id = _expect_string(record, "capture_id", location)
    shot_id = _expect_string(record, "shot_id", location)
    if record["derivation_spec_version"] != DERIVATION_SPEC_VERSION:
        raise MacroLabelError(location, "unsupported derivation spec version")
    if _expect_digest(record, "derivation_spec_digest", location) != derivation_spec_digest():
        raise MacroLabelError(location, "derivation spec digest differs from this module")
    clock = record["event_clock"]
    if not isinstance(clock, dict):
        raise MacroLabelError(location, "event_clock must be an object")
    _expect_fields(clock, EVENT_CLOCK_FIELDS, location)
    if clock["occurrence_authority"] != "fixed_step" or clock["render_frame_role"] != "provenance_only":
        raise MacroLabelError(location, "event_clock declaration differs from v1")
    if _parse_vocabulary(record["macro_vocabulary"], location) != _macro_vocabulary_json():
        raise MacroLabelError(location, "macro_vocabulary differs from the pinned ordered vocabulary")
    pig_classes = record["pig_class_set"]
    if not isinstance(pig_classes, list) or pig_classes != list(PIG_CLASS_SET):
        raise MacroLabelError(location, "pig_class_set differs from the pinned v1 set")
    sources = record["sources"]
    if not isinstance(sources, dict):
        raise MacroLabelError(location, "sources must be an object")
    _expect_fields(sources, SOURCE_FIELDS, location)
    if sources["physics_state_path"] != STATE_SIDECAR or sources["physics_events_path"] != EVENT_SIDECAR:
        raise MacroLabelError(location, "source paths differ from v1")
    return _ParsedHeader(
        capture_id=capture_id,
        shot_id=shot_id,
        state_sha256=_expect_digest(sources, "physics_state_sha256", location),
        events_sha256=_expect_digest(sources, "physics_events_sha256", location),
        state_count=_expect_count(record, "state_count", location),
        event_count=_expect_count(record, "event_count", location),
        interval_count=_expect_count(record, "interval_count", location),
        frame_label_count=_expect_count(record, "frame_label_count", location),
    )


def _parse_interval(record: JsonObject, location: str) -> EventInterval:
    _expect_fields(record, INTERVAL_FIELDS, location)
    interval_type = record["interval_type"]
    if interval_type not in (MacroPredicate.STEADY_STATE.value, MacroPredicate.CASCADE_ACTIVE.value):
        raise MacroLabelError(location, "interval_type must be steady-state or cascade-active")
    start = _expect_count(record, "start_fixed_step", location)
    raw_end = record["end_fixed_step"]
    if raw_end is not None and (isinstance(raw_end, bool) or not isinstance(raw_end, int) or raw_end < 0):
        raise MacroLabelError(location, "end_fixed_step must be a nonnegative integer or null")
    try:
        status = SemanticStatus(record["semantic_status"])
    except ValueError as error:
        raise MacroLabelError(location, "unknown semantic_status value") from error
    if status is not _PREDICATE_STATUS[MacroPredicate(interval_type)]:
        raise MacroLabelError(location, "interval semantic_status differs from the pinned vocabulary")
    if interval_type == MacroPredicate.CASCADE_ACTIVE.value and raw_end is None:
        raise MacroLabelError(location, "cascade-active interval must have a non-null end")
    if raw_end is not None and raw_end <= start:
        raise MacroLabelError(location, "interval end must be greater than its start")
    return EventInterval(
        interval_type=interval_type,
        start_fixed_step=start,
        end_fixed_step=raw_end,
        semantic_status=status,
        evidence=_parse_evidence(record["evidence"], location),
    )


def _parse_frame(record: JsonObject, location: str) -> MacroFrameLabel:
    _expect_fields(record, FRAME_FIELDS, location)
    identity = StateIdentity(
        capture_id=_expect_string(record, "capture_id", location),
        shot_id=_expect_string(record, "shot_id", location),
        state_sequence=_expect_count(record, "state_sequence", location),
        render_frame=_expect_count(record, "render_frame", location),
        fixed_step=_expect_count(record, "fixed_step", location),
        rgb_relative_path=_expect_string(record, "rgb_relative_path", location),
    )
    raw_predicates = record["predicates"]
    if not isinstance(raw_predicates, dict):
        raise MacroLabelError(location, "predicates must be an object")
    if set(raw_predicates.keys()) != {predicate.value for predicate in MacroPredicate}:
        raise MacroLabelError(location, "predicates must cover exactly the five macro predicates")
    predicates = tuple(
        (
            predicate,
            _parse_predicate_label(raw_predicates[predicate.value], f"{location}.predicates.{predicate.value}"),
        )
        for predicate in MacroPredicate
    )
    raw_active = record["active_macro_states"]
    if not isinstance(raw_active, list) or not all(isinstance(item, str) for item in raw_active):
        raise MacroLabelError(location, "active_macro_states must be an array of predicate names")
    try:
        active = tuple(MacroPredicate(item) for item in raw_active)
    except ValueError as error:
        raise MacroLabelError(location, "active_macro_states carries an unknown predicate") from error
    if active != tuple(predicate for predicate, label in predicates if label.value is True):
        raise MacroLabelError(location, "active_macro_states is inconsistent with predicate values")
    return MacroFrameLabel(identity=identity, predicates=predicates)


def _parse_outcome(record: JsonObject, location: str) -> ShotOutcomeLabel:
    _expect_fields(record, OUTCOME_FIELDS, location)
    try:
        outcome_class = OutcomeClass(record["outcome_class"])
    except ValueError as error:
        raise MacroLabelError(location, "unknown outcome_class value") from error
    try:
        equilibrium = TerminalEquilibrium(record["terminal_equilibrium"])
    except ValueError as error:
        raise MacroLabelError(location, "unknown terminal_equilibrium value") from error
    try:
        status = SemanticStatus(record["semantic_status"])
    except ValueError as error:
        raise MacroLabelError(location, "unknown semantic_status value") from error
    if status is not SemanticStatus.ENGINE_VERIFIED:
        raise MacroLabelError(location, "shot outcome semantic_status must be engine_verified")
    raw_score = record["score"]
    raw_reason = record["reason"]
    raw_terminal = record["terminal_event"]
    raw_state = record["terminal_state"]
    terminal_event = None if raw_terminal is None else _parse_citation(raw_terminal, location)
    terminal_state = None if raw_state is None else _parse_state_identity(raw_state, location)
    score: int | None = None
    reason: str | None = None
    match outcome_class:
        case OutcomeClass.CLEARED:
            if isinstance(raw_score, bool) or not isinstance(raw_score, int):
                raise MacroLabelError(location, "cleared outcome requires an integer score")
            if raw_reason is not None or terminal_event is None:
                raise MacroLabelError(location, "cleared outcome requires a null reason and a terminal event")
            score = raw_score
        case OutcomeClass.FAILED:
            if not isinstance(raw_reason, str) or not raw_reason:
                raise MacroLabelError(location, "failed outcome requires a nonempty string reason")
            if raw_score is not None or terminal_event is None:
                raise MacroLabelError(location, "failed outcome requires a null score and a terminal event")
            reason = raw_reason
        case OutcomeClass.SETTLED_NONTERMINAL | OutcomeClass.UNSETTLED_NONTERMINAL:
            if (
                raw_score is not None
                or raw_reason is not None
                or terminal_event is not None
                or terminal_state is not None
            ):
                raise MacroLabelError(
                    location,
                    "nonterminal outcomes require null score, reason, terminal event, and terminal state",
                )
            if equilibrium is not TerminalEquilibrium.NOT_OBSERVED:
                raise MacroLabelError(location, "nonterminal outcomes must not observe a terminal equilibrium")
    if equilibrium is TerminalEquilibrium.STABLE_TERMINAL and terminal_state is None:
        raise MacroLabelError(location, "stable_terminal requires a terminal state")
    return ShotOutcomeLabel(
        outcome_class=outcome_class,
        score=score,
        reason=reason,
        terminal_event=terminal_event,
        terminal_equilibrium=equilibrium,
        terminal_state=terminal_state,
        semantic_status=status,
    )


def read_macro_labels(path: Path) -> MacroLabels:
    """Strictly parse a macro-label sidecar, failing closed on any contract drift."""
    location = str(path)
    if not path.is_file():
        raise MacroLabelError(location, "macro label sidecar is missing")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise MacroLabelError(location, "macro label sidecar is unreadable") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MacroLabelError(location, "macro label sidecar is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise MacroLabelError(location, "macro label sidecar must use LF line endings and end with a newline")

    records = [
        (number, _parse_record(line, f"{location}:{number}"))
        for number, line in enumerate(text[:-1].split("\n"), start=1)
    ]

    header_record: JsonObject | None = None
    interval_records: list[tuple[int, JsonObject]] = []
    frame_records: list[tuple[int, JsonObject]] = []
    outcome_record: JsonObject | None = None
    phase = "header"
    for number, record in records:
        line_location = f"{location}:{number}"
        record_type = record.get("record_type")
        if record_type == "macro_label_header":
            if phase != "header":
                raise MacroLabelError(line_location, "exactly one macro_label_header is allowed and it must be first")
            header_record = record
            phase = "intervals"
        elif record_type == "event_interval":
            if phase == "header":
                raise MacroLabelError(line_location, "macro_label_header must be the first record")
            if phase != "intervals":
                raise MacroLabelError(line_location, "event_interval records must precede frame_label records")
            interval_records.append((number, record))
        elif record_type == "frame_label":
            if phase == "header":
                raise MacroLabelError(line_location, "macro_label_header must be the first record")
            if phase == "outcome":
                raise MacroLabelError(line_location, "frame_label records must precede the shot_outcome record")
            phase = "frames"
            frame_records.append((number, record))
        elif record_type == "shot_outcome":
            if phase == "header":
                raise MacroLabelError(line_location, "macro_label_header must be the first record")
            if number != len(records):
                raise MacroLabelError(line_location, "shot_outcome must be the last record")
            outcome_record = record
            phase = "outcome"
        else:
            raise MacroLabelError(line_location, f"unknown record_type: {record_type!r}")
    if header_record is None or outcome_record is None:
        raise MacroLabelError(location, "macro label sidecar must carry exactly one header and one outcome")

    header = _parse_header(header_record, f"{location}:1")
    intervals = tuple(_parse_interval(record, f"{location}:{number}") for number, record in interval_records)
    frames = tuple(_parse_frame(record, f"{location}:{number}") for number, record in frame_records)
    outcome = _parse_outcome(outcome_record, f"{location}:{len(records)}")

    if header.state_count != len(frames) or header.frame_label_count != len(frames):
        raise MacroLabelError(location, "state/frame-label counts disagree with the header")
    if header.interval_count != len(intervals):
        raise MacroLabelError(location, "interval count disagrees with the header")
    identities = [frame.identity for frame in frames]
    if len(set(identities)) != len(identities):
        raise MacroLabelError(location, "duplicate frame-label identity")
    for absorbing in ABSORBING_PREDICATES:
        seen_true = False
        for frame in frames:
            if seen_true and frame.predicate(absorbing).value is not True:
                raise MacroLabelError(location, f"absorbing predicate {absorbing.value} must not revert")
            seen_true = seen_true or frame.predicate(absorbing).value is True

    # Same-type records must carry the canonical emission order, not just valid
    # individual shapes: intervals are ordered by (start_fixed_step, interval_type)
    # and frame labels follow the accepted-state order (strictly increasing
    # state_sequence).  A permutation that leaves counts and identities intact is
    # still contract drift and fails closed here.
    if intervals != tuple(
        sorted(intervals, key=lambda interval: (interval.start_fixed_step, interval.interval_type))
    ):
        raise MacroLabelError(location, "event intervals differ from canonical order")
    state_sequences = tuple(frame.identity.state_sequence for frame in frames)
    if any(previous >= current for previous, current in zip(state_sequences, state_sequences[1:])):
        raise MacroLabelError(location, "frame labels differ from accepted state order")

    return MacroLabels(
        capture_id=header.capture_id,
        shot_id=header.shot_id,
        state_sha256=header.state_sha256,
        events_sha256=header.events_sha256,
        event_count=header.event_count,
        intervals=intervals,
        frames=frames,
        outcome=outcome,
    )


def validate_macro_labels(shot_dir: Path, label_path: Path | None = None) -> MacroLabels:
    """Re-derive a shot's labels and reject any stored file that disagrees.

    Fails closed on: a missing or malformed sidecar (via `read_macro_labels`), stale
    source digests (the capture changed after labelling), or any byte that differs
    from a fresh derivation (noncanonical order/bytes included).
    """
    path = label_path if label_path is not None else shot_dir / MACRO_LABEL_SIDECAR
    location = str(path)
    if not path.is_file():
        raise MacroLabelError(location, "macro label sidecar is missing")
    stored = read_macro_labels(path)
    expected = derive_macro_labels_for_shot(shot_dir)
    if stored.state_sha256 != expected.state_sha256 or stored.events_sha256 != expected.events_sha256:
        raise MacroLabelError(location, "source sidecar digests are stale")
    if path.read_bytes() != expected.to_jsonl().encode("utf-8"):
        raise MacroLabelError(location, "stored labels disagree with a fresh derivation")
    return stored
