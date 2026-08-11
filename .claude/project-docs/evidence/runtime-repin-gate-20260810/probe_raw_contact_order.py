#!/usr/bin/env python3
"""Prove or refute F1 without spending the single non-retryable smoke.

F1, as reported by the code reviewer on the session-3 diff: the Unity emitter
serialises the recorder's *cumulative* raw-contact list, which is sorted only
within each fixed step, while scripts/physics_capture_parsing.py:281 requires the
whole array to be globally sorted by a key that does not include fixed_step.

This probe reconstructs the emitter's exact output shape in Python and feeds it to
the real parser entry point, scripts.physics_capture_parsing._parse_state. Nothing
here re-implements the ordering predicate: the verdict comes from the production
parser raising, or not raising, PhysicsCaptureContractError.

Emitter shape, read out of the C# before this file was written:

  PhysicsShotRecorder.RecordContacts        :406-435  builds stepContacts,
      sorts them with CompareContacts (entity_a, entity_b, collider_a, collider_b,
      point.x, point.y — the parser's key minus contact_id), then
      rawContacts.AddRange(stepContacts). rawContacts is never cleared.
  PhysicsShotRecorder.CreateFinalizedSnapshot :642   passes the cumulative field.
  PhysicalShotRecorderSnapshot ctor          :316-319 copies the whole list.
  PhysicsCaptureProtocol.BuildStateJson      :141    writes recorder.RawContacts
      in list order; BuildContactsJson :148-155 iterates, and never sorts.

So one state record per shot carries every contact from every fixed step,
concatenated step-major.

Run with PYTHONDONTWRITEBYTECODE=1 — package_physics_player.py:105 aborts a build
if scripts/__pycache__/ exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.physics_capture_contract import (  # noqa: E402
    EXPECTED_COORDINATES,
    SCHEMA_VERSION,
    PhysicsContractError,
)
from scripts.physics_capture_parsing import _parse_state  # noqa: E402

# Two pairs that a real shot has at once: the bird resting on a block, and two
# blocks resting on each other. Entity ids follow ENTITY_ID_PATTERN (`<id>:<gen>`)
# and are canonical (entity_a_id < entity_b_id), as _parse_contact:244 requires.
PAIR_LOW = ("100:0", "200:0", 11, 21)
PAIR_HIGH = ("300:0", "400:0", 31, 41)

CONTACT_SORT_KEY = ("entity_a_id", "entity_b_id", "collider_a_id", "collider_b_id")


def contact(pair: tuple[str, str, int, int], fixed_step: int, point_index: int) -> dict[str, Any]:
    entity_a, entity_b, collider_a, collider_b = pair
    pair_key = f"{entity_a}|{entity_b}"
    return {
        "contact_id": f"contact:{fixed_step}:{pair_key}:{point_index}",
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


def step_sample(fixed_step: int) -> list[dict[str, Any]]:
    """One full FixedUpdate sample: both resting pairs, sorted as CompareContacts
    sorts them — which is the parser's key minus contact_id, so a single step in
    isolation is always correctly ordered."""
    sample = [contact(PAIR_LOW, fixed_step, 0), contact(PAIR_HIGH, fixed_step, 1)]
    return sorted(sample, key=lambda c: tuple(c[field] for field in CONTACT_SORT_KEY))


def emitter_shaped(steps: int) -> list[dict[str, Any]]:
    """What the wire actually carries: per-step samples concatenated in step order."""
    out: list[dict[str, Any]] = []
    for fixed_step in range(1, steps + 1):
        out.extend(step_sample(fixed_step))
    return out


def globally_sorted(contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same set under the parser's own key, contact_id included."""
    return sorted(
        contacts,
        key=lambda c: (
            c["entity_a_id"], c["entity_b_id"], c["collider_a_id"], c["collider_b_id"],
            c["point"]["x"], c["point"]["y"], c["contact_id"],
        ),
    )


# --- F2: the same defect class on support_edges -------------------------------
#
# UpdateSupport (PhysicsShotRecorder.cs:651-683) appends each edge in the
# iteration order of stepContacts — that is, ordered by the *contact's* entity
# pair. But the edge's supporter_id is whichever of the pair sits lower in y,
# which may be either member. The parser (:283) requires the array sorted by
# (supporter_id, supported_id, support_id). Whether the emitted order happens to
# satisfy that is therefore a function of the scene's geometry, not an invariant.
#
# The case below is the one the geometry can produce: a low-numbered pair whose
# *upper* member is entity A (so the supporter is the high id), sampled before a
# high-numbered pair whose *lower* member is entity A (supporter is the low id).

def support_edge(supporter: str, supported: str, pair_key: str) -> dict[str, Any]:
    # _parse_support:260 requires exactly this id shape, so it is derived rather
    # than invented — an invented one fails on format before the order is checked.
    return {
        "support_id": f"support:{supporter}->{supported}",
        "rule_version": "support_v1",
        "supporter_id": supporter,
        "supported_id": supported,
        "evidence_contact_ids": [f"contact:1:{pair_key}:0", f"contact:2:{pair_key}:0"],
        "evidence_fixed_steps": [1, 2],
    }


def emitter_shaped_supports() -> list[dict[str, Any]]:
    """Append order is by contact pair; the resulting supporter ids are not."""
    return [
        # pair 100:0|900:0 — B is the lower body, so the supporter is 900:0
        support_edge("900:0", "100:0", "100:0|900:0"),
        # pair 200:0|300:0 — A is the lower body, so the supporter is 200:0
        support_edge("200:0", "300:0", "200:0|300:0"),
    ]


def globally_sorted_supports(supports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        supports,
        key=lambda s: (s["supporter_id"], s["supported_id"], s["support_id"]),
    )



def state_record(
    raw_contacts: list[dict[str, Any]],
    support_edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A minimal state record valid in every respect except, possibly, the order
    of raw_contacts or support_edges. Zero nodes keep the only failure this probe
    can produce the one it is testing for."""
    return {
        "record_type": "state",
        "schema_version": SCHEMA_VERSION,
        "capture_id": "probe-capture",
        "shot_id": "shot_001",
        "sequence": 0,
        "render_frame": 7,
        "render_time": 0.116,
        "fixed_step": 4,
        "fixed_time": 0.08,
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
        "support_edges": support_edges or [],
    }


def attempt(
    label: str,
    raw_contacts: list[dict[str, Any]],
    support_edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        frame = _parse_state(state_record(raw_contacts, support_edges), 0)
    except PhysicsContractError as error:
        return {
            "case": label,
            "contact_count": len(raw_contacts),
            "support_count": len(support_edges or []),
            "parsed": False,
            "error_code": error.code.value,
            "error": str(error),
        }
    return {
        "case": label,
        "contact_count": len(raw_contacts),
        "support_count": len(support_edges or []),
        "parsed": True,
        "parsed_contact_count": len(frame.raw_contacts),
        "parsed_support_count": len(frame.support_edges),
    }


def main() -> int:
    single_step = step_sample(1)
    cumulative = emitter_shaped(3)
    ordered = globally_sorted(cumulative)
    supports = emitter_shaped_supports()

    results = [
        attempt("single_step_only", single_step),
        attempt("emitter_shaped_cumulative_3_steps", cumulative),
        attempt("globally_sorted_same_set", ordered),
        attempt("f2_emitter_shaped_support_edges", ordered, supports),
        attempt("f2_globally_sorted_support_edges", ordered, globally_sorted_supports(supports)),
    ]

    by_case = {item["case"]: item for item in results}
    f1_confirmed = (
        by_case["single_step_only"]["parsed"]
        and not by_case["emitter_shaped_cumulative_3_steps"]["parsed"]
        and by_case["globally_sorted_same_set"]["parsed"]
    )
    f2_confirmed = (
        not by_case["f2_emitter_shaped_support_edges"]["parsed"]
        and by_case["f2_globally_sorted_support_edges"]["parsed"]
    )

    report = {
        "probe": "sidecar_array_order",
        "findings": ["F1", "F2"],
        "question": (
            "Do the emitter's raw_contacts and support_edges arrays satisfy the "
            "global ordering contract at scripts/physics_capture_parsing.py:281 and :283?"
        ),
        "parser_entry_point": "scripts.physics_capture_parsing._parse_state",
        "results": results,
        "f1_verdict": "f1_confirmed" if f1_confirmed else "f1_not_reproduced",
        "f2_verdict": "f2_confirmed" if f2_confirmed else "f2_not_reproduced",
        "interpretation": {
            "f1": (
                "A single fixed step parses; the same contacts accumulated across "
                "three steps do not; globally sorting that identical set makes it "
                "parse again. The defect is the order the emitter writes, not the "
                "contacts themselves. rawContacts is never cleared, and "
                "PhysicalSnapshotRuntime.FixedUpdate:109 samples every collider "
                "every fixed step, so any shot longer than one step with more than "
                "one contacting pair emits a violating array."
            ),
            "f2": (
                "support_edges is pruned to the current step, so it is not "
                "cumulative — but its append order follows the contact pair while "
                "the contract sorts by supporter_id, which is whichever body is "
                "lower in y. Whether a given scene violates the order is therefore "
                "a property of its geometry, not an invariant. The case here is one "
                "the geometry can produce."
            ),
        },
    }
    print(json.dumps(report, indent=2))
    return 0 if (f1_confirmed and f2_confirmed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
