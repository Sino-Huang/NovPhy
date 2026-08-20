from __future__ import annotations

"""Promote the F1/F2 emitter-ordering probe to a tracked regression case.

The GitHub issue #44 runtime probe
probe_raw_contact_order.py`) proved that the emitter's cumulative, step-major
`raw_contacts` array violates the frozen parser contract at
`scripts/physics_capture_parsing.py:281`, and that the same set globally sorted
parses. The producer fix sorts once at finalization, and retention (F7) keeps
exactly the last two full fixed steps plus collision-cited rows. These fixtures
feed that retention-shaped, globally-sorted emitter output through the real
production entry point `_parse_state` and assert it parses; the unsorted
step-major shapes must keep failing, so a regression to the ordering fix cannot
go green.
"""

import unittest
from typing import Any, TypeAlias

from scripts.physics_capture_contract import (
    ContractErrorCode,
    EXPECTED_COORDINATES,
    SCHEMA_VERSION,
    PhysicsContractError,
)
from scripts.physics_capture_parsing import _parse_state


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

# Two pairs that a real shot has at once, canonical (entity_a_id < entity_b_id)
# as _parse_contact requires. The trailing ints are the collider ids, which the
# emitter folds into every id it mints.
PAIR_LOW = ("100:0", "200:0", 11, 21)
PAIR_HIGH = ("300:0", "400:0", 31, 41)


def pair_key(pair: tuple[str, str, int, int]) -> str:
    """The emitter's canonical collider pair key.

    PhysicalCaptureModels.PairKey is ``EntityIdA|ColliderIdA:EntityIdB|ColliderIdB``,
    so the collider ids are part of the id a real shot carries; omitting them
    would mint ids the emitter can never produce.
    """
    entity_a, entity_b, collider_a, collider_b = pair
    return f"{entity_a}|{collider_a}:{entity_b}|{collider_b}"


def contact_id(pair: tuple[str, str, int, int], fixed_step: int, point_index: int) -> str:
    """The emitter's contact id: ``contact:<FixedStep>:<PairKey>:<PointIndex>``."""
    return f"contact:{fixed_step}:{pair_key(pair)}:{point_index}"


def contact(pair: tuple[str, str, int, int], fixed_step: int, point_index: int) -> JsonObject:
    entity_a, entity_b, collider_a, collider_b = pair
    return {
        "contact_id": contact_id(pair, fixed_step, point_index),
        "entity_a_id": entity_a,
        "entity_b_id": entity_b,
        "collider_a_id": collider_a,
        "collider_b_id": collider_b,
        "point": {"x": 0.5 + 0.25 * point_index, "y": 0.25},
        "normal_a_to_b": {"x": 0.0, "y": 1.0},
        "separation": -0.01,
        "relative_velocity_a_to_b": {"x": 1.5, "y": 0.0},
        "normal_impulse": 1.0,
        "tangent_impulse": 0.5,
        "is_trigger": False,
    }


def _parser_contact_key(c: JsonObject) -> tuple[Any, ...]:
    point = c["point"]
    return (
        c["entity_a_id"], c["entity_b_id"], c["collider_a_id"], c["collider_b_id"],
        point["x"], point["y"], c["contact_id"],
    )


def retained_contacts() -> list[JsonObject]:
    """What the emitter carries after F7 retention: fixed steps 2 and 3 for both
    pairs, plus one collision-cited step-1 row, all globally sorted under the
    parser's own key (contact_id included)."""
    rows: list[JsonObject] = []
    for fixed_step in (2, 3):
        for point_index, pair in enumerate((PAIR_LOW, PAIR_HIGH)):
            rows.append(contact(pair, fixed_step, point_index))
    rows.append(contact(PAIR_LOW, 1, 0))
    return sorted(rows, key=_parser_contact_key)


def emitter_shaped_unsorted(steps: int) -> list[JsonObject]:
    """The pre-fix emitter shape: per-step samples concatenated step-major."""
    rows: list[JsonObject] = []
    for fixed_step in range(1, steps + 1):
        for point_index, pair in enumerate((PAIR_LOW, PAIR_HIGH)):
            rows.append(contact(pair, fixed_step, point_index))
    return rows


def support_edge(supporter: str, supported: str, pair: tuple[str, str, int, int], point_index: int) -> JsonObject:
    # A support edge cites its pair's previous and current step contacts
    # (PhysicsShotRecorder.UpdateSupport records prior.ContactId and
    # contact.ContactId), so the evidence ids are the exact retained contact
    # ids for fixed steps 2 and 3 -- the same shape `contact_id` mints.
    return {
        "support_id": f"support:{supporter}->{supported}",
        "rule_version": "support_v1",
        "supporter_id": supporter,
        "supported_id": supported,
        "evidence_contact_ids": [contact_id(pair, 2, point_index), contact_id(pair, 3, point_index)],
        "evidence_fixed_steps": [2, 3],
    }


def emitter_ordered_supports() -> list[JsonObject]:
    """The F2 geometry in append (contact-pair) order, which is not supporter
    order. Each edge cites the retained step-2/step-3 contacts of its own pair:
    PAIR_LOW's retained rows carry point 0 and PAIR_HIGH's carry point 1, and
    for both pairs B is the lower body, so the supporter is the B entity. The
    pairs are appended high-pair first, while the contract order sorts by
    (supporter_id, supported_id, support_id) with 200:0 first."""
    return [
        support_edge("400:0", "300:0", PAIR_HIGH, 1),
        support_edge("200:0", "100:0", PAIR_LOW, 0),
    ]


def finalized_supports() -> list[JsonObject]:
    """The same edges under the contract's (supporter_id, supported_id,
    support_id) order, which is what the finalized snapshot emits."""
    return sorted(emitter_ordered_supports(), key=lambda s: (s["supporter_id"], s["supported_id"], s["support_id"]))


def state_record(
    raw_contacts: list[JsonObject],
    support_edges: list[JsonObject],
) -> JsonObject:
    """A minimal state record valid in every respect except, possibly, the order
    of raw_contacts or support_edges."""
    return {
        "record_type": "state",
        "schema_version": SCHEMA_VERSION,
        "capture_id": "retention-order-test",
        "shot_id": "shot_001",
        "sequence": 0,
        "render_frame": 7,
        "render_time": 0.116,
        "fixed_step": 3,
        "fixed_time": 0.06,
        "coordinates": {
            field: getattr(EXPECTED_COORDINATES, field)
            for field in EXPECTED_COORDINATES.__dataclass_fields__
        },
        "rgb_frame": {
            "relative_path": "frames/0007.png",
            "render_frame": 7,
            "width_pixels": 640,
            "height_pixels": 480,
            "source": "synchronized_endpoint",
        },
        "nodes": [],
        "raw_contacts": raw_contacts,
        "support_edges": support_edges,
    }


class PhysicsCaptureRetentionOrderTests(unittest.TestCase):
    def test_retention_shaped_globally_sorted_data_parses(self) -> None:
        frame = _parse_state(state_record(retained_contacts(), finalized_supports()), 0)
        self.assertEqual(len(frame.raw_contacts), 5)
        self.assertEqual(len(frame.support_edges), 2)

    def test_emitter_shaped_unsorted_cumulative_contacts_are_still_rejected(self) -> None:
        with self.assertRaises(PhysicsContractError) as raised:
            _parse_state(state_record(emitter_shaped_unsorted(3), []), 0)
        self.assertEqual(raised.exception.code, ContractErrorCode.DETERMINISTIC_ORDER)

    def test_unsorted_support_edges_are_still_rejected(self) -> None:
        # The support edges in append order are not the contract's
        # (supporter_id, supported_id, support_id) order.
        with self.assertRaises(PhysicsContractError) as raised:
            _parse_state(state_record(retained_contacts(), emitter_ordered_supports()), 0)
        self.assertEqual(raised.exception.code, ContractErrorCode.DETERMINISTIC_ORDER)


if __name__ == "__main__":
    unittest.main()
