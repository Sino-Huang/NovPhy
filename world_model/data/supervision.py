from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.physics_capture_contract import PhysicsContractError, load_physics_capture
from scripts.physics_label_derivation import (
    DERIVED_LABEL_VECTOR_FIELDS,
    DerivedFrameLabel,
    DerivedLabelError,
    OracleGateSpec,
    validate_derived_labels,
)
from scripts.physics_macro_labels import (
    MACRO_LABEL_SIDECAR,
    PREDICATE_SEMANTIC_STATUS,
    MacroFrameLabel,
    MacroLabelError,
    SemanticStatus,
    validate_macro_labels,
)
from scripts.physics_relational_supervision import (
    RelationalFrameLabel,
    RelationalSupervisionError,
    validate_relational_supervision,
)
from scripts.physics_capture_types import (
    CoordinateDeclaration,
    EntityId,
    EventId,
    EventRecord,
    EventType,
    PhysicsCapture,
    RawContact,
    SceneNode,
    ShotId,
    SupportEdge,
    Vector2,
)
from world_model.data.types import ContractValueError


@dataclass(frozen=True, slots=True)
class PhysicsEvent:
    shot_id: ShotId
    render_frame: int
    fixed_step: int
    event_id: EventId
    event_type: EventType
    participants: tuple[EntityId, ...]
    payload_json: str


@dataclass(frozen=True, slots=True)
class PhysicsFrameSupervision:
    frame_index: int
    shot_id: ShotId
    render_frame: int
    render_time: float
    fixed_step: int
    fixed_time: float
    coordinates: CoordinateDeclaration
    rgb_source: str
    nodes: tuple[SceneNode, ...]
    raw_contacts: tuple[RawContact, ...]
    support_edges: tuple[SupportEdge, ...]
    events: tuple[PhysicsEvent, ...]
    derived_labels: DerivedFrameLabel | None = None
    macro_labels: MacroFrameLabel | None = None
    relational_labels: RelationalFrameLabel | None = None

    @property
    def relational_supervision(self) -> RelationalFrameLabel | None:
        return self.relational_labels


@dataclass(frozen=True, slots=True)
class PhysicsSupervisionRequest:
    required_capabilities: tuple[str, ...] = ()
    include_raw_contacts: bool = False
    include_events: bool = False
    include_derived_labels: bool = False
    include_macro_labels: bool = False
    oracle_gate_spec: OracleGateSpec | None = None
    include_relational_labels: bool = False

    def __post_init__(self) -> None:
        if type(self.required_capabilities) is not tuple:
            raise ContractValueError("required_capabilities", "must be an immutable tuple")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            raise ContractValueError("required_capabilities", "must be unique")
        if not all(isinstance(capability, str) and capability.strip() for capability in self.required_capabilities):
            raise ContractValueError("required_capabilities", "must contain nonempty strings")
        if (
            type(self.include_raw_contacts) is not bool
            or type(self.include_events) is not bool
            or type(self.include_derived_labels) is not bool
            or type(self.include_macro_labels) is not bool
            or type(self.include_relational_labels) is not bool
        ):
            raise ContractValueError("supervision flags", "must be booleans")
        if self.oracle_gate_spec is not None and not isinstance(self.oracle_gate_spec, OracleGateSpec):
            raise ContractValueError("oracle_gate_spec", "must be an OracleGateSpec")


def _event(event: EventRecord) -> PhysicsEvent:
    return PhysicsEvent(
        shot_id=event.clock.shot_id,
        render_frame=event.clock.render_frame,
        fixed_step=event.clock.fixed_step,
        event_id=event.event_id,
        event_type=event.event_type,
        participants=event.participants,
        payload_json=event.payload_json,
    )


def read_physics_shot(
    shot_dir: Path,
    shot_name: str,
    frame_paths: tuple[str, ...],
    request: PhysicsSupervisionRequest,
) -> tuple[PhysicsFrameSupervision, ...]:
    from scripts.rollout_artifacts import validate_physics_shot_artifact
    from scripts.rollout_validation_types import PhysicsArtifactError

    try:
        validate_physics_shot_artifact(shot_dir)
        capture: PhysicsCapture = load_physics_capture(
            shot_dir / "physics_state.jsonl", shot_dir / "physics_events.jsonl"
        )
    except (OSError, PhysicsContractError, PhysicsArtifactError) as error:
        raise ContractValueError("physics sidecars", str(error)) from error
    states_by_path = {}
    # Events are exposed by fixed-step bracketing, never by event render_frame: the
    # producer stamps every buffered event with the serialization snapshot's
    # render_frame, so event render_frame is provenance only and event occurrence
    # authority is fixed_step (milestone-0a plan section 3.4).  State i exposes
    # exactly the events with fixed_step in (states[i-1].fixed_step, states[i].fixed_step]
    # (no lower bound for the first state); events after the last accepted state are
    # not exposed on any frame.  Events are fixed-step ordered by the frozen
    # contract, so a single cursor yields deterministic brackets; two states sharing
    # one fixed step give the later ones an empty bracket.
    events_by_state: dict[tuple[str, int], tuple[PhysicsEvent, ...]] = {}
    event_cursor = 0
    for state in capture.states:
        if str(state.clock.shot_id) != shot_name:
            raise ContractValueError("physics shot_id", "does not match shot directory")
        key = (str(state.clock.shot_id), state.clock.render_frame)
        if key in states_by_path:
            raise ContractValueError("physics render_frame", "duplicate state key")
        states_by_path[key] = state
        bracket_start = event_cursor
        while event_cursor < len(capture.events) and capture.events[event_cursor].clock.fixed_step <= state.clock.fixed_step:
            event_cursor += 1
        events_by_state[key] = tuple(_event(event) for event in capture.events[bracket_start:event_cursor])
    path_to_state = {state.rgb_frame.relative_path: state for state in capture.states}
    if len(path_to_state) != len(capture.states):
        raise ContractValueError("physics rgb mapping", "duplicate state frame path")
    labels_by_frame: dict[int, DerivedFrameLabel] = {}
    if request.include_derived_labels:
        # Re-derives from the frozen sidecars and rejects a stale, mutated, or
        # differently-thresholded label file, so a bad label can never reach training.
        try:
            derived = validate_derived_labels(shot_dir, request.oracle_gate_spec or OracleGateSpec())
        except (OSError, DerivedLabelError, PhysicsContractError) as error:
            raise ContractValueError("physics derived labels", str(error)) from error
        for label in derived.frames:
            if label.render_frame in labels_by_frame:
                raise ContractValueError("physics derived labels", "duplicate label render_frame")
            labels_by_frame[label.render_frame] = label
        if set(labels_by_frame) != {state.clock.render_frame for state in capture.states}:
            raise ContractValueError(
                "physics derived labels", "label frames do not match the accepted state frames"
            )
    macro_by_identity: dict[tuple[str, str, int, int, int, str], MacroFrameLabel] = {}
    if request.include_macro_labels:
        # validate_macro_labels re-derives from the frozen sidecars and byte-compares,
        # so a stale or tampered label file fails closed before any label reaches a sample.
        try:
            macro = validate_macro_labels(shot_dir)
        except (OSError, MacroLabelError, PhysicsContractError) as error:
            raise ContractValueError("physics macro labels", str(error)) from error
        pending_predicates = tuple(
            predicate.value
            for predicate, status in PREDICATE_SEMANTIC_STATUS
            if status == SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION
        )
        if pending_predicates:
            raise ContractValueError(
                "physics macro labels",
                "pending predicates require representative validation: "
                + ", ".join(pending_predicates),
            )
        for label in macro.frames:
            identity = label.identity
            identity_key = (
                identity.capture_id,
                identity.shot_id,
                identity.state_sequence,
                identity.render_frame,
                identity.fixed_step,
                identity.rgb_relative_path,
            )
            if identity_key in macro_by_identity:
                raise ContractValueError("physics macro labels", "duplicate frame-label identity")
            macro_by_identity[identity_key] = label
        state_identities = {
            (
                str(state.clock.capture_id),
                str(state.clock.shot_id),
                state.clock.sequence,
                state.clock.render_frame,
                state.clock.fixed_step,
                state.rgb_frame.relative_path,
            )
            for state in capture.states
        }
        if set(macro_by_identity) != state_identities:
            raise ContractValueError(
                "physics macro labels", "frame labels do not match the accepted state identities"
            )
    relational_by_identity: dict[tuple[str, str, int, int, int, str], RelationalFrameLabel] = {}
    if request.include_relational_labels:
        # A relational label must describe exactly the accepted source state before
        # it is allowed into a supervision sample.
        try:
            relational = validate_relational_supervision(shot_dir)
        except (OSError, RelationalSupervisionError, PhysicsContractError) as error:
            raise ContractValueError("physics relational labels", str(error)) from error
        for label in relational.frames:
            identity = label.identity
            identity_key = (
                identity.capture_id,
                identity.shot_id,
                identity.state_sequence,
                identity.render_frame,
                identity.fixed_step,
                identity.rgb_relative_path,
            )
            if identity_key in relational_by_identity:
                raise ContractValueError(
                    "physics relational labels", "duplicate frame-label identity"
                )
            relational_by_identity[identity_key] = label
        state_identities = {
            (
                str(state.clock.capture_id),
                str(state.clock.shot_id),
                state.clock.sequence,
                state.clock.render_frame,
                state.clock.fixed_step,
                state.rgb_frame.relative_path,
            )
            for state in capture.states
        }
        if set(relational_by_identity) != state_identities:
            raise ContractValueError(
                "physics relational labels",
                "frame labels do not match the accepted state identities",
            )
    result: list[PhysicsFrameSupervision] = []
    for frame_index, relative_path in enumerate(frame_paths):
        state = path_to_state.get(relative_path)
        if state is None:
            raise ContractValueError("physics rgb mapping", "no state for RGB frame")
        key = (str(state.clock.shot_id), state.clock.render_frame)
        if states_by_path.get(key) != state:
            raise ContractValueError("physics rgb mapping", "non-unique shot/render-frame state")
        identity_key = (
            str(state.clock.capture_id),
            str(state.clock.shot_id),
            state.clock.sequence,
            state.clock.render_frame,
            state.clock.fixed_step,
            state.rgb_frame.relative_path,
        )
        result.append(PhysicsFrameSupervision(
            frame_index=frame_index,
            shot_id=state.clock.shot_id,
            render_frame=state.clock.render_frame,
            render_time=state.clock.render_time,
            fixed_step=state.clock.fixed_step,
            fixed_time=state.clock.fixed_time,
            coordinates=state.clock.coordinates,
            rgb_source=state.rgb_frame.source,
            nodes=state.nodes,
            raw_contacts=state.raw_contacts if request.include_raw_contacts else (),
            support_edges=state.support_edges,
            events=events_by_state[key] if request.include_events else (),
            derived_labels=labels_by_frame.get(state.clock.render_frame) if request.include_derived_labels else None,
            macro_labels=macro_by_identity[identity_key] if request.include_macro_labels else None,
            relational_labels=relational_by_identity[identity_key] if request.include_relational_labels else None,
        ))
    return tuple(result)
