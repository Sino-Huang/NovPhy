"""Issue #110 representation contract v2 for every NovPhy scenario x novelty cell.

Frozen before any #110 capture or fit. It extends the frozen 18-slot / 236-value contract
(issue-70 v6 vocabulary, used by #71/#77/#99/#100/#112) and never edits it:

* slot vocabulary v2 (28 slots: 4 birds, 8 blocks, 1 novelty source, 1 pig, 11 platforms,
  slingshot, 2 landscape) and a 4-value scene-state block in the carrier header;
* the symmetric action contract (level 5 puts the slingshot at x = +12);
* the force laws of the canonical capture player (constants recovered from the original
  NovPhy assembly and prefabs; ``scripts/prepare_novelty_coverage.py`` re-verifies every
  constant against those sources) and the force-aware predicates derived from them;
* the task objective, declared for every level including level 7.

Predicates are engine-side derivation labels computed from the native trace plus the
authored scenario; nothing here is an agent observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Iterable, Mapping, Sequence

CONTRACT_IDENTITY: Final = "issue-110-representation-contract-v2"

# ----------------------------------------------------------------- vocabulary

SLOT_CAPACITY: Final = {"bird": 4, "block": 8, "novelty": 1, "pig": 1, "platform": 11, "slingshot": 1}
LANDSCAPE_SLOTS: Final = ("world:landscape:0000", "world:landscape:0001")
VOCABULARY_V1: Final = (
    "bird:0000", "bird:0001", "bird:0002",
    "block:0000", "block:0001", "block:0002", "block:0003", "block:0004",
    "pig:0000",
    "platform:0000", "platform:0001", "platform:0002", "platform:0003", "platform:0004", "platform:0005",
    "slingshot:0000", "world:landscape:0000", "world:landscape:0001",
)
VOCABULARY_V2: Final = tuple(sorted(
    [f"{kind}:{index:04}" for kind, count in SLOT_CAPACITY.items() for index in range(count)]
    + list(LANDSCAPE_SLOTS)))
CARRIER_HEADER: Final = 2          # v1 header: has-prior flag, elapsed seconds
SCENE_FIELDS: Final = ("gravity_y_normalized", "gravity_available", "storm_active", "storm_available")
SLOT_WIDTH: Final = 13             # v1 ENTITY_FEATURES, unchanged
CARRIER_DIM_V1: Final = CARRIER_HEADER + SLOT_WIDTH * len(VOCABULARY_V1)
CARRIER_DIM_V2: Final = CARRIER_HEADER + len(SCENE_FIELDS) + SLOT_WIDTH * len(VOCABULARY_V2)
STANDARD_GRAVITY: Final = 9.8      # gravity_y_normalized = world gravity y / 9.8

# Visual kind per novelty type: what a perception head can be asked to read. The three
# air-turbulence variants share one sprite and colour, Magnet uses the ordinary wood
# Circle sprite, and InverseGravity/Storm have no renderer before the storm overlay.
KIND_VOCABULARY_V2: Final = ("bird", "pig", "block", "platform", "slingshot", "world",
                             "fan", "air-turbulence", "other")
NOVELTY_VISUAL_KIND: Final = {
    "PinkCircle": "block", "PinkRectFat": "block", "PinkSquareHole": "block", "Magnet": "block",
    "Fan": "fan",
    "NonNovelAirTurbulence": "air-turbulence", "NovelAirTurbulence": "air-turbulence",
    "InverseAirTurbulence": "air-turbulence",
    "InverseGravity": None, "Storm": None,
}

# ------------------------------------------------------------- force laws

SPEED_GATE: Final = 0.1            # AirTurbulence / Storm act only on bodies faster than this
TURBULENCE_BOX_LOCAL: Final = 0.64  # BoxCollider2D size in local units; world size = 0.64 * template scale
TURBULENCE_FORCE: Final = {"NonNovelAirTurbulence": 1.5, "NovelAirTurbulence": 4.0, "InverseAirTurbulence": -10.0}
TURBULENCE_EXCLUDED_KINDS: Final = ("pig", "platform")
FAN_THRUST: Final = 25.0
FAN_WIND_WIDTH: Final = 2.0
FAN_WIND_HEIGHT: Final = 3.0
MAGNET_FORCE: Final = 15.0
MAGNET_RADIUS: Final = 4.0
INVERSE_GRAVITY: Final = (0.0, 6.0)
INVERSE_GRAVITY_BIRD_SCALE: Final = -1.0
STORM_WIND: Final = (3.0, 0.0)

FORCE_LAWS: Final = {
    "region_push": {
        "types": tuple(TURBULENCE_FORCE),
        "rule": ("force = transform.up * turbulenceForce on every body inside the trigger box whose speed "
                 "exceeds 0.1; pigs and platforms excluded"),
        "region": "box of 0.64 * (scaleX, scaleY) world units centred on the authored position",
        "visibility": "visible_region_hidden_law",
    },
    "bird_zone_push": {
        "types": ("Fan",),
        "rule": ("force = -transform.right * 25 on the launched bird while 0 < fan.x - bird.x < 2 and "
                 "|bird.y - fan.y| < 1.5"),
        "region": "4 x 3 trigger box; the law uses the left half only",
        "visibility": "visible_entity",
    },
    "radial_linear": {
        "types": ("Magnet",),
        "rule": ("force = 15 * (magnet - body) on birds and on blocks of another material, the reverse on "
                 "blocks of the magnet's material, inside radius 4; a body that collides with the magnet "
                 "leaves the force lists until it re-enters the field"),
        "region": "circle of radius 4",
        "visibility": "hidden_law_normal_appearance",
    },
    "global_gravity": {
        "types": ("InverseGravity",),
        "rule": "Physics2D.gravity = (0, 6); birds not on the slingshot at load get gravityScale -1",
        "region": "global",
        "visibility": "invisible_global",
    },
    "event_wind": {
        "types": ("Storm",),
        "rule": ("after the first bird is destroyed: force = (3, 0) on flying or dying birds and on blocks "
                 "whose speed exceeds 0.1; a grey sprite (35.8 x 19.2 world units, alpha 0.5) is instantiated "
                 "at onset, so the storm becomes visible only then"),
        "region": "global",
        "visibility": "invisible_trigger_visible_after_onset",
    },
}
LAW_OF_TYPE: Final = {kind: law for law, spec in FORCE_LAWS.items() for kind in spec["types"]}

# ---------------------------------------------------------- action contract

ACTION_BOUNDS_V2: Final = {"drag_x_magnitude": [10, 160], "drag_y": [-80, 80],
                           "tap_time_ms": [0, 1000], "release_time_ms": [600, 1000]}
ACTION_ENCODING: Final = "[drag_x / 480, drag_y / 480, release_time_ms / 1000, tap_time_ms / 1000, 1] (v1, unchanged)"


def slingshot_side(slingshot_x: float) -> str:
    """'left' launches to the right (pull drag_x < 0); 'right' (level 5, x = +12) launches left."""
    return "left" if slingshot_x < 0 else "right"


def mirror_action(action: Mapping[str, int]) -> dict:
    """Reflect a slingshot-relative action about the vertical axis (drag_x -> -drag_x)."""
    return {**action, "drag_x": -action["drag_x"]}


def action_for_side(action: Mapping[str, int], side: str) -> dict:
    """Inventories are authored for the left slingshot; the right slingshot uses their mirror."""
    if side not in ("left", "right"):
        raise ValueError(f"unknown slingshot side: {side}")
    return dict(action) if side == "left" else mirror_action(action)


def action_within_v2(action: Mapping[str, int], side: str) -> bool:
    """Pull away from the targets: drag_x < 0 for the left slingshot, > 0 for the right one."""
    low, high = ACTION_BOUNDS_V2["drag_x_magnitude"]
    pull = -action["drag_x"] if side == "left" else action["drag_x"]
    return (low <= pull <= high
            and ACTION_BOUNDS_V2["drag_y"][0] <= action["drag_y"] <= ACTION_BOUNDS_V2["drag_y"][1]
            and ACTION_BOUNDS_V2["tap_time_ms"][0] <= action["tap_time_ms"] <= ACTION_BOUNDS_V2["tap_time_ms"][1]
            and ACTION_BOUNDS_V2["release_time_ms"][0] <= action["release_time_ms"]
            <= ACTION_BOUNDS_V2["release_time_ms"][1])

# ------------------------------------------------------------- objective

TASK_OBJECTIVE: Final = {
    "every_level": ("first-shot success = engine pig removal (the #85 channel's pig_removed); decision cost = "
                    "the #99 tie-free cost on pig:0000 (presence - 0.1 * displacement), lower is better; "
                    "terminal kinds native_clear / native_fail / stable_without_clear / right_censored"),
    "level_7": ("unchanged. The changed goal is the air-turbulence agent's: InverseAirTurbulence pushes down "
                "(turbulenceForce -10) where NonNovelAirTurbulence pushes up (+1.5). The engine clear "
                "condition (every pig dead) and fail condition (birds exhausted) do not change"),
    "decision_cost_unit": "single pig per level (all 80 templates author exactly one pig)",
}

# -------------------------------------------------------------- predicates

# Where a law depends on engine state the native trace does not carry, the label uses the
# closest recorded quantity. The pre-campaign rendered smoke checks each against the
# residual acceleration of contact-free bodies.
LABEL_APPROXIMATIONS: Final = {
    "region_push": "trigger overlap approximated by the body centre lying inside the box",
    "radial_linear": ("trigger overlap approximated by centre distance < radius; 'removed after a collision "
                      "until re-entry' approximated by 'not in contact with the magnet at this step'"),
    "event_wind": "flying-or-dying bird approximated by 'launched and still active'",
}

MICRO_PREDICATES_V2: Final = ("contact", "supports", "force-on")
MACRO_FORCE_PREDICATES: Final = ("external-force-active",)
SCENE_LABELS: Final = ("gravity-inverted", "storm-active")


@dataclass(frozen=True)
class Source:
    """The authored novelty source of a level (at most one per template)."""
    slot: str
    type: str
    position: tuple[float, float]
    rotation_degrees: float = 0.0
    scale: tuple[float, float] = (1.0, 1.0)
    material: str | None = None


@dataclass(frozen=True)
class Body:
    """One active entity at one native step, from the trace plus its authored kind/material."""
    slot: str
    kind: str
    position: tuple[float, float]
    velocity: tuple[float, float]
    material: str | None = None
    launched: bool = False
    touching_source: bool = False

    @property
    def speed(self) -> float:
        return math.hypot(*self.velocity)


def _rotate(vector: Sequence[float], degrees: float) -> tuple[float, float]:
    angle = math.radians(degrees)
    return (vector[0] * math.cos(angle) - vector[1] * math.sin(angle),
            vector[0] * math.sin(angle) + vector[1] * math.cos(angle))


def force_on(source: Source, body: Body, *, storm_active: bool = False) -> tuple[float, float] | None:
    """The force the source applies to the body at this step, or None when its law is inactive."""
    if body.slot == source.slot:
        return None
    law = LAW_OF_TYPE.get(source.type)
    if law == "region_push":
        if body.kind in TURBULENCE_EXCLUDED_KINDS or body.kind not in ("bird", "block", "novelty"):
            return None
        local = _rotate((body.position[0] - source.position[0], body.position[1] - source.position[1]),
                        -source.rotation_degrees)
        half = (TURBULENCE_BOX_LOCAL * abs(source.scale[0]) / 2, TURBULENCE_BOX_LOCAL * abs(source.scale[1]) / 2)
        if abs(local[0]) > half[0] or abs(local[1]) > half[1] or body.speed <= SPEED_GATE:
            return None
        return _rotate((0.0, TURBULENCE_FORCE[source.type]), source.rotation_degrees)
    if law == "bird_zone_push":
        if body.kind != "bird" or not body.launched:
            return None
        dx, dy = body.position[0] - source.position[0], body.position[1] - source.position[1]
        if not (dx < 0 and abs(dx) < FAN_WIND_WIDTH and abs(dy) < FAN_WIND_HEIGHT / 2):
            return None
        right = _rotate((1.0, 0.0), source.rotation_degrees)
        return (-right[0] * FAN_THRUST, -right[1] * FAN_THRUST)
    if law == "radial_linear":
        if body.kind not in ("bird", "block") or body.touching_source:
            return None
        offset = (source.position[0] - body.position[0], source.position[1] - body.position[1])
        if math.hypot(*offset) >= MAGNET_RADIUS * abs(source.scale[0]):
            return None
        sign = -1.0 if body.kind == "block" and body.material == source.material else 1.0
        return (sign * MAGNET_FORCE * offset[0], sign * MAGNET_FORCE * offset[1])
    if law == "event_wind":
        if not storm_active:
            return None
        if body.kind == "bird" and body.launched:
            return STORM_WIND
        if body.kind in ("block", "novelty") and body.speed > SPEED_GATE:
            return STORM_WIND
        return None
    return None  # global_gravity has no pairwise edge; normal levels have no source


def force_edges(source: Source | None, bodies: Iterable[Body], *, storm_active: bool = False) -> set:
    """Micro 'force-on' edges (source slot, body slot) active at this step."""
    if source is None:
        return set()
    return {(source.slot, body.slot) for body in bodies if force_on(source, body, storm_active=storm_active)}


def gravity_inverted(gravity_vector: Sequence[float]) -> bool:
    return gravity_vector[1] > 0


def storm_onset_step(events: Iterable[Mapping], bird_entities: Iterable[str]) -> int | None:
    """First native step at which a bird entity is destroyed (the Storm law's trigger)."""
    birds = set(bird_entities)
    steps = [event["fixed_step"] for event in events
             if event["event_type"] == "entity_destroyed" and birds & set(event["participants"])]
    return min(steps) if steps else None


def window_any(flags: Sequence[bool], t: int, delta: int) -> bool:
    """A per-Delta predicate observed at its own horizon: true when any frame in (t, t + delta] is true."""
    if delta < 1:
        raise ValueError("delta must be positive")
    return any(flags[t + 1:t + delta + 1])


def residual_acceleration(previous: Body, current: Body, gravity: Sequence[float], gravity_scale: float,
                          seconds: float) -> tuple[float, float]:
    """Observed acceleration minus scaled gravity between two contact-free native steps."""
    return ((current.velocity[0] - previous.velocity[0]) / seconds - gravity[0] * gravity_scale,
            (current.velocity[1] - previous.velocity[1]) / seconds - gravity[1] * gravity_scale)


# --------------------------------------------- trace / scenario adapters

def source_from_scenario(level_xml_root) -> Source | None:
    """The authored Novelty node of a materialized level (ElementTree root), if any."""
    nodes = [node for node in level_xml_root.iter("Novelty")]
    if len(nodes) > 1:
        raise ValueError("contract v2 declares at most one novelty source per level")
    if not nodes:
        return None
    node = nodes[0]
    material = node.attrib.get("material") or None
    if node.attrib["type"] == "Magnet" and material is None:
        material = "wood"  # the Magnet prefab is the wood Circle ("existing circular wood object")
    return Source(slot=node.attrib["scenarioObjectId"], type=node.attrib["type"],
                  position=(float(node.attrib["x"]), float(node.attrib["y"])),
                  rotation_degrees=float(node.attrib.get("rotation", 0.0)),
                  scale=(float(node.attrib.get("scaleX", 1.0)), float(node.attrib.get("scaleY", 1.0))),
                  material=material)


def slot_kind(slot: str) -> str:
    return "world" if slot.startswith("world:") else slot.split(":")[0]


def bodies_from_sample(sample: Mapping, materials: Mapping[str, str | None], launched: Iterable[str],
                       source_slot: str | None = None) -> list[Body]:
    """Active bodied entities of one native-trace sample (canonical_native_chunk_v1)."""
    launched, touching = set(launched), set()
    if source_slot is not None:
        source_entity = "runtime:" + source_slot
        for contact in sample["contacts"]:
            pair = {contact["entity_a_id"], contact["entity_b_id"]}
            if source_entity in pair:
                touching |= pair - {source_entity}
    bodies = []
    for entity in sample["entities"]:
        if entity["lifecycle"] != "active" or not entity["body_present"]:
            continue
        slot, body = entity["scenario_object_id"], entity["body"]
        bodies.append(Body(slot=slot, kind=slot_kind(slot), position=tuple(body["position"]),
                           velocity=tuple(body["velocity"]), material=materials.get(slot),
                           launched=slot in launched, touching_source=entity["entity_id"] in touching))
    return bodies
