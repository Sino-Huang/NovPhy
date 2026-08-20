"""Deterministic macro-state, outcome, and oracle-gate label derivation.

Every label produced here is a pure function of engine-exported `physics_capture_v1`
facts: scene nodes with kinematics, raw contacts, derived support edges, and the
frozen ten-event taxonomy.  Nothing is inferred from RGB, no model is consulted, and
no threshold is implicit -- the full `OracleGateSpec` travels with every derived
record so a cohort labelled under different thresholds can never be silently mixed
with another.

The frozen `physics_capture_v1` contract is not touched. Labels live in a separate
`physics_derived_labels_v1` sidecar written beside the capture sidecars.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
import json
import math
import os
from pathlib import Path
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
    StateFrame,
)

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

DERIVED_LABEL_SCHEMA_VERSION: Final = "physics_derived_labels_v1"
DERIVED_LABEL_SIDECAR: Final = "physics_derived_labels.jsonl"
PIG_OBJECT_CLASS: Final = "pig"

#: Fixed, documented field order of the per-frame numeric vector.  A Milestone 1
#: collator stacks this without re-deriving semantics, so the order is part of the
#: contract and is asserted by tests.
DERIVED_LABEL_VECTOR_FIELDS: Final = (
    "oracle_gate",
    "structure_unstable",
    "cascade_active",
    "collapsed",
    "pigs_cleared",
    "steady_state",
    "total_kinetic_energy",
    "active_contact_count",
    "raw_contact_count",
    "support_edge_count",
    "dynamic_node_count",
    "pig_count",
)

DESTRUCTION_EVENTS: Final = (
    EventType.COLLISION,
    EventType.EXPLOSION,
    EventType.ENTITY_DESTROYED,
    EventType.PIG_REMOVED,
)


@unique
class MacroState(StrEnum):
    """Structure-level (hydrodynamic) predicates over a shot."""

    CASCADE_ACTIVE = "cascade-active"
    COLLAPSED = "collapsed"
    PIGS_CLEARED = "pigs-cleared"
    STEADY_STATE = "steady-state"
    STRUCTURE_UNSTABLE = "structure-unstable"


#: Absorbing predicates: once true for a shot they remain true by vocabulary design.
ABSORBING_MACRO_STATES: Final = (MacroState.COLLAPSED, MacroState.PIGS_CLEARED)


@unique
class ShotOutcomeClass(StrEnum):
    """Terminal macro state a cascade relaxes into."""

    CLEARED = "cleared"
    FAILED = "failed"
    SETTLED = "settled"
    UNSETTLED = "unsettled"


@dataclass(frozen=True, slots=True)
class DerivedLabelError(ValueError):
    """Raised when derived labels are absent, stale, or disagree with their sidecars."""

    location: str
    detail: str

    def __str__(self) -> str:
        return f"invalid derived labels at {self.location}: {self.detail}"


@dataclass(frozen=True, slots=True)
class OracleGateSpec:
    """Thresholds for the oracle scale-separation gate (proposal section 2.3).

    `kinetic_energy_threshold` and `active_contact_threshold` are the two indicator
    conditions of phi*.  `contact_activity_speed` defines when a raw contact counts as
    *active*: a resting stack keeps its contacts but none of them are active, so a
    settled structure reads as quiescent rather than contact-saturated.
    """

    kinetic_energy_threshold: float = 0.01
    active_contact_threshold: int = 1
    contact_activity_speed: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("kinetic_energy_threshold", self.kinetic_energy_threshold),
            ("contact_activity_speed", self.contact_activity_speed),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.active_contact_threshold, bool) or not isinstance(
            self.active_contact_threshold, int
        ):
            raise ValueError("active_contact_threshold must be an integer")
        if self.active_contact_threshold < 1:
            raise ValueError("active_contact_threshold must be positive")

    def to_json(self) -> JsonObject:
        return {
            "kinetic_energy_threshold": float(self.kinetic_energy_threshold),
            "active_contact_threshold": int(self.active_contact_threshold),
            "contact_activity_speed": float(self.contact_activity_speed),
        }

    def identity(self) -> str:
        return (
            "oracle-gate-spec-v1:"
            f"{self.kinetic_energy_threshold}:{self.active_contact_threshold}:"
            f"{self.contact_activity_speed}"
        )

    @classmethod
    def from_json(cls, payload: JsonValue, location: str) -> "OracleGateSpec":
        if not isinstance(payload, dict):
            raise DerivedLabelError(location, "oracle_gate_spec must be an object")
        expected = {"kinetic_energy_threshold", "active_contact_threshold", "contact_activity_speed"}
        if payload.keys() != expected:
            raise DerivedLabelError(location, "oracle_gate_spec fields differ from v1")
        try:
            return cls(
                kinetic_energy_threshold=float(payload["kinetic_energy_threshold"]),  # type: ignore[arg-type]
                active_contact_threshold=int(payload["active_contact_threshold"]),  # type: ignore[arg-type]
                contact_activity_speed=float(payload["contact_activity_speed"]),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as error:
            raise DerivedLabelError(location, "oracle_gate_spec values are invalid") from error


@dataclass(frozen=True, slots=True)
class DerivedFrameLabel:
    """Per-render-frame labels and the scalar evidence they were derived from."""

    render_frame: int
    render_time: float
    fixed_step: int
    fixed_time: float
    oracle_gate: bool
    macro_states: tuple[MacroState, ...]
    total_kinetic_energy: float
    active_contact_count: int
    raw_contact_count: int
    support_edge_count: int
    dynamic_node_count: int
    pig_count: int

    def has(self, state: MacroState) -> bool:
        return state in self.macro_states

    def to_vector(self) -> tuple[float, ...]:
        """Return the fixed-order numeric vector declared by DERIVED_LABEL_VECTOR_FIELDS."""
        return (
            float(self.oracle_gate),
            float(self.has(MacroState.STRUCTURE_UNSTABLE)),
            float(self.has(MacroState.CASCADE_ACTIVE)),
            float(self.has(MacroState.COLLAPSED)),
            float(self.has(MacroState.PIGS_CLEARED)),
            float(self.has(MacroState.STEADY_STATE)),
            float(self.total_kinetic_energy),
            float(self.active_contact_count),
            float(self.raw_contact_count),
            float(self.support_edge_count),
            float(self.dynamic_node_count),
            float(self.pig_count),
        )

    def to_json(self) -> JsonObject:
        return {
            "record_type": "frame_label",
            "render_frame": self.render_frame,
            "render_time": self.render_time,
            "fixed_step": self.fixed_step,
            "fixed_time": self.fixed_time,
            "oracle_gate": self.oracle_gate,
            "macro_states": [state.value for state in self.macro_states],
            "total_kinetic_energy": self.total_kinetic_energy,
            "active_contact_count": self.active_contact_count,
            "raw_contact_count": self.raw_contact_count,
            "support_edge_count": self.support_edge_count,
            "dynamic_node_count": self.dynamic_node_count,
            "pig_count": self.pig_count,
        }


@dataclass(frozen=True, slots=True)
class ShotOutcome:
    """Terminal macro state (the equilibrium a cascade relaxes into) for one shot."""

    outcome_class: ShotOutcomeClass
    score: int | None = None
    reason: str | None = None
    terminal_macro_states: tuple[MacroState, ...] = ()

    def to_json(self) -> JsonObject:
        return {
            "record_type": "shot_outcome",
            "outcome_class": self.outcome_class.value,
            "score": self.score,
            "reason": self.reason,
            "terminal_macro_states": [state.value for state in self.terminal_macro_states],
        }


@dataclass(frozen=True, slots=True)
class DerivedLabels:
    """All labels derived from one accepted shot."""

    schema_version: str
    capture_id: str
    shot_id: str
    oracle_gate_spec: OracleGateSpec
    frames: tuple[DerivedFrameLabel, ...]
    outcome: ShotOutcome

    def header_json(self) -> JsonObject:
        return {
            "record_type": "derived_label_header",
            "schema_version": self.schema_version,
            "capture_schema_version": "physics_capture_v1",
            "capture_id": self.capture_id,
            "shot_id": self.shot_id,
            "oracle_gate_spec": self.oracle_gate_spec.to_json(),
            "macro_state_taxonomy": [state.value for state in MacroState],
            "vector_fields": list(DERIVED_LABEL_VECTOR_FIELDS),
            "frame_count": len(self.frames),
            "source": {
                "physics_state_path": STATE_SIDECAR,
                "physics_events_path": EVENT_SIDECAR,
            },
        }

    def to_jsonl(self) -> str:
        records: list[JsonObject] = [self.header_json()]
        records.extend(frame.to_json() for frame in self.frames)
        records.append(self.outcome.to_json())
        return "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
        )


def _total_kinetic_energy(state: StateFrame) -> float:
    total = 0.0
    for node in state.nodes:
        if node.body.present and node.body.kinetic_energy_unity_units is not None:
            total += node.body.kinetic_energy_unity_units
    return total


def _active_contact_count(state: StateFrame, spec: OracleGateSpec) -> int:
    active = 0
    for contact in state.raw_contacts:
        relative = contact.relative_velocity_a_to_b
        speed = math.hypot(relative.x, relative.y)
        if speed >= spec.contact_activity_speed:
            active += 1
    return active


def _pig_count(state: StateFrame) -> int:
    return sum(1 for node in state.nodes if node.object_class == PIG_OBJECT_CLASS)


def _dynamic_node_count(state: StateFrame) -> int:
    return sum(1 for node in state.nodes if node.body.present)


def _support_pairs(state: StateFrame) -> frozenset[tuple[str, str]]:
    return frozenset((str(edge.supporter_id), str(edge.supported_id)) for edge in state.support_edges)


def _events_by_frame(events: tuple[EventRecord, ...]) -> dict[int, tuple[EventRecord, ...]]:
    grouped: dict[int, list[EventRecord]] = {}
    for event in events:
        grouped.setdefault(event.clock.render_frame, []).append(event)
    return {frame: tuple(items) for frame, items in grouped.items()}


def _first_frame_of(events: tuple[EventRecord, ...], event_type: EventType) -> int | None:
    for event in events:
        if event.event_type is event_type:
            return event.clock.render_frame
    return None


def _steady_intervals(events: tuple[EventRecord, ...]) -> tuple[tuple[int, int | None], ...]:
    """Half-open [entered, exited) stability intervals from the engine's debounced events.

    The engine emits `stable_entered`/`stable_exited` only on debounced transitions and
    the contract guarantees they alternate, so the intervals are well formed by
    construction.  A trailing `stable_entered` runs to the end of the capture.
    """
    intervals: list[tuple[int, int | None]] = []
    opened: int | None = None
    for event in events:
        if event.event_type is EventType.STABLE_ENTERED:
            opened = event.clock.render_frame
        elif event.event_type is EventType.STABLE_EXITED and opened is not None:
            intervals.append((opened, event.clock.render_frame))
            opened = None
    if opened is not None:
        intervals.append((opened, None))
    return tuple(intervals)


def _in_steady_interval(frame: int, intervals: tuple[tuple[int, int | None], ...]) -> bool:
    for start, end in intervals:
        if frame >= start and (end is None or frame < end):
            return True
    return False


def _cascade_frames(
    events: tuple[EventRecord, ...],
    launch_frame: int | None,
    debounce_frames: int,
) -> tuple[int, int] | None:
    """Return the [first, last] render-frame span of the post-launch destruction burst."""
    destruction = [
        event.clock.render_frame
        for event in events
        if event.event_type in DESTRUCTION_EVENTS
        and (launch_frame is None or event.clock.render_frame >= launch_frame)
    ]
    if not destruction:
        return None
    return min(destruction), max(destruction) + debounce_frames


def _debounce_frames(events: tuple[EventRecord, ...]) -> int:
    """Read the engine's own debounce window from a stability event, defaulting to one frame."""
    for event in events:
        if event.event_type in (EventType.STABLE_ENTERED, EventType.STABLE_EXITED):
            payload = json.loads(event.payload_json)
            debounce = payload.get("debounce_fixed_steps")
            if isinstance(debounce, int) and not isinstance(debounce, bool) and debounce >= 1:
                return debounce
    return 1


def _terminal_event(events: tuple[EventRecord, ...]) -> EventRecord | None:
    for event in events:
        if event.event_type in (EventType.LEVEL_CLEARED, EventType.LEVEL_FAILED):
            return event
    return None


def derive_labels(capture: PhysicsCapture, spec: OracleGateSpec | None = None) -> DerivedLabels:
    """Derive macro-state, outcome, and oracle-gate labels from a parsed capture.

    Pure: the same capture and spec always produce identical records.  The caller is
    responsible for having validated the capture (`load_physics_capture` does).
    """
    gate_spec = spec or OracleGateSpec()
    events = capture.events
    launch_frame = _first_frame_of(events, EventType.BIRD_LAUNCHED)
    debounce = _debounce_frames(events)
    steady = _steady_intervals(events)
    cascade = _cascade_frames(events, launch_frame, debounce)
    events_by_frame = _events_by_frame(events)

    destroyed_by_frame: dict[int, set[str]] = {}
    for event in events:
        if event.event_type in (EventType.ENTITY_DESTROYED, EventType.PIG_REMOVED):
            destroyed_by_frame.setdefault(event.clock.render_frame, set()).update(
                str(participant) for participant in event.participants
            )

    frames: list[DerivedFrameLabel] = []
    previous_supports: frozenset[tuple[str, str]] | None = None
    collapsed_latched = False
    pigs_cleared_latched = False
    destroyed_so_far: set[str] = set()
    removed_pigs = 0

    for state in capture.states:
        frame = state.clock.render_frame
        supports = _support_pairs(state)
        total_ke = _total_kinetic_energy(state)
        active_contacts = _active_contact_count(state, gate_spec)
        pig_count = _pig_count(state)

        for event in events_by_frame.get(frame, ()):
            if event.event_type is EventType.PIG_REMOVED:
                removed_pigs += 1
        destroyed_so_far.update(destroyed_by_frame.get(frame, set()))

        steady_state = _in_steady_interval(frame, steady) or (
            launch_frame is not None and frame < launch_frame
        )

        support_changed = previous_supports is not None and supports != previous_supports
        structure_unstable = (not steady_state) and (
            support_changed or total_ke >= gate_spec.kinetic_energy_threshold
        )

        cascade_active = (
            cascade is not None and cascade[0] <= frame <= cascade[1] and not steady_state
        )

        # `collapsed`: a support edge that existed earlier is gone and the entity it
        # carried was destroyed or lost all of its support.  Engine-evidenced, absorbing.
        if not collapsed_latched and previous_supports is not None:
            lost = previous_supports - supports
            for supporter, supported in lost:
                still_supported = any(pair[1] == supported for pair in supports)
                if supported in destroyed_so_far or supporter in destroyed_so_far or not still_supported:
                    collapsed_latched = True
                    break

        # `pigs-cleared`: no pig node remains and the removals are event-accounted.
        if not pigs_cleared_latched and pig_count == 0 and removed_pigs > 0:
            pigs_cleared_latched = True

        macro: list[MacroState] = []
        if cascade_active:
            macro.append(MacroState.CASCADE_ACTIVE)
        if collapsed_latched:
            macro.append(MacroState.COLLAPSED)
        if pigs_cleared_latched:
            macro.append(MacroState.PIGS_CLEARED)
        if steady_state:
            macro.append(MacroState.STEADY_STATE)
        if structure_unstable:
            macro.append(MacroState.STRUCTURE_UNSTABLE)

        oracle_gate = (
            total_ke < gate_spec.kinetic_energy_threshold
            and active_contacts < gate_spec.active_contact_threshold
        )

        frames.append(
            DerivedFrameLabel(
                render_frame=frame,
                render_time=state.clock.render_time,
                fixed_step=state.clock.fixed_step,
                fixed_time=state.clock.fixed_time,
                oracle_gate=oracle_gate,
                macro_states=tuple(sorted(macro, key=lambda item: item.value)),
                total_kinetic_energy=total_ke,
                active_contact_count=active_contacts,
                raw_contact_count=len(state.raw_contacts),
                support_edge_count=len(state.support_edges),
                dynamic_node_count=_dynamic_node_count(state),
                pig_count=pig_count,
            )
        )
        previous_supports = supports

    terminal = _terminal_event(events)
    terminal_states = frames[-1].macro_states if frames else ()
    if terminal is not None and terminal.event_type is EventType.LEVEL_CLEARED:
        payload = json.loads(terminal.payload_json)
        outcome = ShotOutcome(
            ShotOutcomeClass.CLEARED,
            score=payload.get("score"),
            terminal_macro_states=terminal_states,
        )
    elif terminal is not None:
        payload = json.loads(terminal.payload_json)
        outcome = ShotOutcome(
            ShotOutcomeClass.FAILED,
            reason=payload.get("reason"),
            terminal_macro_states=terminal_states,
        )
    elif frames and MacroState.STEADY_STATE in frames[-1].macro_states:
        outcome = ShotOutcome(ShotOutcomeClass.SETTLED, terminal_macro_states=terminal_states)
    else:
        outcome = ShotOutcome(ShotOutcomeClass.UNSETTLED, terminal_macro_states=terminal_states)

    return DerivedLabels(
        schema_version=DERIVED_LABEL_SCHEMA_VERSION,
        capture_id=str(capture.header.clock.capture_id),
        shot_id=str(capture.header.clock.shot_id),
        oracle_gate_spec=gate_spec,
        frames=tuple(frames),
        outcome=outcome,
    )


def derive_labels_for_shot(shot_dir: Path, spec: OracleGateSpec | None = None) -> DerivedLabels:
    """Load a shot's frozen sidecars and derive its labels."""
    state_path = shot_dir / STATE_SIDECAR
    event_path = shot_dir / EVENT_SIDECAR
    if not state_path.is_file() or not event_path.is_file():
        raise DerivedLabelError(str(shot_dir), "physics capture sidecars are missing")
    capture = load_physics_capture(state_path, event_path)
    return derive_labels(capture, spec)


def write_derived_labels(shot_dir: Path, spec: OracleGateSpec | None = None) -> Path:
    """Derive and atomically write a shot's `physics_derived_labels_v1` sidecar.

    Never writes under `frames/` and never touches `metadata.json`, so a shot that
    passed `validate_physics_shot_artifact` still passes afterwards.
    """
    labels = derive_labels_for_shot(shot_dir, spec)
    destination = shot_dir / DERIVED_LABEL_SIDECAR
    temporary = shot_dir / f".{DERIVED_LABEL_SIDECAR}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(labels.to_jsonl())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise DerivedLabelError(str(destination), "derived labels could not be written") from error
    return destination


def read_derived_labels(shot_dir: Path) -> DerivedLabels:
    """Parse a derived-label sidecar without re-deriving it."""
    path = shot_dir / DERIVED_LABEL_SIDECAR
    location = str(path)
    if not path.is_file():
        raise DerivedLabelError(location, "derived label sidecar is missing")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DerivedLabelError(location, "derived label sidecar is unreadable") from error
    if len(lines) < 2:
        raise DerivedLabelError(location, "derived label sidecar must carry a header and an outcome")
    try:
        records = [json.loads(line) for line in lines]
    except json.JSONDecodeError as error:
        raise DerivedLabelError(location, "derived label sidecar is not JSONL") from error
    if not all(isinstance(record, dict) for record in records):
        raise DerivedLabelError(location, "derived label records must be objects")

    header = records[0]
    if header.get("record_type") != "derived_label_header":
        raise DerivedLabelError(location, "first record must be derived_label_header")
    if header.get("schema_version") != DERIVED_LABEL_SCHEMA_VERSION:
        raise DerivedLabelError(location, "unsupported derived label schema version")
    spec = OracleGateSpec.from_json(header.get("oracle_gate_spec"), location)
    source = header.get("source")
    if not isinstance(source, dict):
        raise DerivedLabelError(location, "header must carry a source object")

    outcome_record = records[-1]
    if outcome_record.get("record_type") != "shot_outcome":
        raise DerivedLabelError(location, "last record must be shot_outcome")
    try:
        outcome_class = ShotOutcomeClass(outcome_record.get("outcome_class"))
    except ValueError as error:
        raise DerivedLabelError(location, "unknown outcome class") from error

    frames: list[DerivedFrameLabel] = []
    for index, record in enumerate(records[1:-1], start=2):
        if record.get("record_type") != "frame_label":
            raise DerivedLabelError(f"{location}:{index}", "expected frame_label")
        try:
            macro = tuple(MacroState(value) for value in record["macro_states"])
            frames.append(
                DerivedFrameLabel(
                    render_frame=record["render_frame"],
                    render_time=record["render_time"],
                    fixed_step=record["fixed_step"],
                    fixed_time=record["fixed_time"],
                    oracle_gate=record["oracle_gate"],
                    macro_states=macro,
                    total_kinetic_energy=record["total_kinetic_energy"],
                    active_contact_count=record["active_contact_count"],
                    raw_contact_count=record["raw_contact_count"],
                    support_edge_count=record["support_edge_count"],
                    dynamic_node_count=record["dynamic_node_count"],
                    pig_count=record["pig_count"],
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DerivedLabelError(f"{location}:{index}", "malformed frame_label") from error

    terminal_states = tuple(
        MacroState(value) for value in outcome_record.get("terminal_macro_states", [])
    )
    return DerivedLabels(
        schema_version=DERIVED_LABEL_SCHEMA_VERSION,
        capture_id=str(header.get("capture_id", "")),
        shot_id=str(header.get("shot_id", "")),
        oracle_gate_spec=spec,
        frames=tuple(frames),
        outcome=ShotOutcome(
            outcome_class,
            score=outcome_record.get("score"),
            reason=outcome_record.get("reason"),
            terminal_macro_states=terminal_states,
        ),
    )


def validate_derived_labels(shot_dir: Path, spec: OracleGateSpec | None = None) -> DerivedLabels:
    """Re-derive a shot's labels and reject any stored file that disagrees.

    Fails closed on a missing sidecar, a threshold spec other than the requested one,
    a differing label body, or an absorbing predicate that reverts.
    """
    expected_spec = spec or OracleGateSpec()
    stored = read_derived_labels(shot_dir)
    location = str(shot_dir / DERIVED_LABEL_SIDECAR)

    if stored.oracle_gate_spec != expected_spec:
        raise DerivedLabelError(location, "stored thresholds differ from the requested spec")

    expected = derive_labels_for_shot(shot_dir, expected_spec)
    if stored.to_jsonl() != expected.to_jsonl():
        raise DerivedLabelError(location, "stored labels disagree with a fresh derivation")

    for absorbing in ABSORBING_MACRO_STATES:
        seen = False
        for frame in stored.frames:
            active = frame.has(absorbing)
            if seen and not active:
                raise DerivedLabelError(location, f"{absorbing.value} must not revert")
            seen = seen or active
    return stored
