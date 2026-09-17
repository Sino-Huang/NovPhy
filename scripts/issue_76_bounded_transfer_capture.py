"""Additive paired-input capture over the frozen #76 shared-history collector.

The frozen v4 collector remains unchanged.  This wrapper intercepts its first
paused request-72 observation, obtains the retained request-70 screen capture
first, and seals both single-frame views with the already-recorded three-frame
world-camera history before the shot is allowed to proceed.
"""
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import issue_76_shared_history_capture as shared
from world_model.training.native_history_data import VISUAL_DIM, visual_carriers


PAIR_MANIFEST = "paired-inputs.json"
SINGLE_VIEWS = frozenset(("legacy-single", "corrected-single"))
HISTORY_VIEW = "corrected-history"


def _json_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _fixed_time(record):
    if "fixed_time_seconds" in record:
        return record["fixed_time_seconds"]
    return record["fixed_time"]


def _require_predecision(step, decision_fixed_step):
    if step > decision_fixed_step:
        raise ValueError("post-action frame cannot be assembled as predecision input")


def _frame(relative_path, png, metadata, decision_fixed_step, source):
    step = metadata["fixed_step"]
    _require_predecision(step, decision_fixed_step)
    return {
        "relative_path": relative_path,
        "png_sha256": sha256(png).hexdigest(),
        "png_bytes": len(png),
        "fixed_step": step,
        "fixed_time_seconds": _fixed_time(metadata),
        "relative_native_step": step - decision_fixed_step,
        "capture_id": metadata["capture_id"],
        "sequence": metadata["sequence"],
        "render_frame": metadata["render_frame"],
        "source": source,
        "predecision": True,
    }


def build_paired_input_manifest(member, decision_fixed_step, legacy, corrected, history_rows):
    """Validate and describe three views of one sealed native decision state."""
    legacy_state = shared._plain_json(legacy.state)
    corrected_metadata = shared._plain_json(corrected.metadata)
    history_rows = [shared._plain_json(row) for row in history_rows]
    if len(history_rows) != 3:
        raise ValueError("corrected history must contain exactly three actual frames")

    expected_steps = [decision_fixed_step - 100, decision_fixed_step - 50,
                      decision_fixed_step]
    actual_steps = [row["fixed_step"] for row in history_rows]
    for step in actual_steps:
        _require_predecision(step, decision_fixed_step)
    if actual_steps != expected_steps:
        raise ValueError("corrected history must use native steps -100/-50/0")
    if any(abs(_fixed_time(b) - _fixed_time(a) - .02) > 1e-5
           for a, b in zip(history_rows, history_rows[1:])):
        raise ValueError("corrected history must use actual twenty-millisecond spacing")

    for metadata in (legacy_state, corrected_metadata):
        _require_predecision(metadata["fixed_step"], decision_fixed_step)
        if metadata["fixed_step"] != decision_fixed_step:
            raise ValueError("single-frame view is not the sealed decision frame")
    if (legacy_state["capture_id"] != corrected_metadata["capture_id"]
            or _fixed_time(legacy_state) != _fixed_time(corrected_metadata)):
        raise ValueError("legacy and corrected singles do not share one native scenario state")
    if _fixed_time(corrected_metadata) != _fixed_time(history_rows[-1]):
        raise ValueError("single-frame views and corrected history have different decision clocks")
    if corrected.canonical_png != history_rows[-1]["canonical_png"]:
        raise ValueError("corrected single differs from the latest corrected history frame")

    legacy_frame = _frame("legacy-single/frame_000001.png", legacy.png,
                          legacy_state, decision_fixed_step,
                          "request_70_screen_capture_request_72_v1_equivalent")
    corrected_frame = _frame("corrected-single/frame_000001.png",
                             corrected.canonical_png, corrected_metadata,
                             decision_fixed_step,
                             "request_72_render_canonical_rgb")
    history_frames = [
        _frame(f"corrected-history/frame_{ordinal:06}.png", row["canonical_png"],
               row, decision_fixed_step, "sealed_aligned_render_canonical_rgb")
        for ordinal, row in enumerate(history_rows, 1)
    ]
    scenario = shared._plain_json(member["scenario"])
    scenario_binding = {
        "member_identity": member["identity"],
        "base_cluster": member["base_cluster"],
        "engine_seed": member["engine_seed"],
        "scenario_sha256": _json_digest(scenario),
    }
    state_binding = {
        "runtime_capture_id": corrected_metadata["capture_id"],
        "history_capture_id": history_rows[-1]["capture_id"],
        "decision_fixed_step": decision_fixed_step,
        "decision_fixed_time_seconds": _fixed_time(corrected_metadata),
        "physics_barrier_paused": True,
        "action_executed_before_capture": False,
    }
    pair_id = "paired-input-v1:" + _json_digest({
        "scenario": scenario_binding,
        "state": state_binding,
    })
    return {
        "schema": "issue_76_bounded_transfer_paired_inputs_v1",
        "identity": pair_id,
        "scenario_binding": scenario_binding,
        "decision_state_binding": state_binding,
        "same_native_decision_state": True,
        "post_action_frames_used": False,
        "views": {
            "legacy-single": {
                "input_kind": "single_frame",
                "hud_inclusive": True,
                "frames": [legacy_frame],
                "temporal_assembly": {
                    "actual_frame_count": 1,
                    "repeated_or_padded_frames": False,
                    "prior_observation_available": [0.0],
                    "elapsed_observation_seconds": [0.0],
                    "motion_available": [False],
                },
            },
            "corrected-single": {
                "input_kind": "single_frame",
                "hud_inclusive": False,
                "frames": [corrected_frame],
                "temporal_assembly": {
                    "actual_frame_count": 1,
                    "repeated_or_padded_frames": False,
                    "prior_observation_available": [0.0],
                    "elapsed_observation_seconds": [0.0],
                    "motion_available": [False],
                },
            },
            "corrected-history": {
                "input_kind": "three_actual_predecision_frames",
                "hud_inclusive": False,
                "frames": history_frames,
                "temporal_assembly": {
                    "actual_frame_count": 3,
                    "repeated_or_padded_frames": False,
                    "relative_native_steps": [-100, -50, 0],
                    "prior_observation_available": [0.0, 1.0, 1.0],
                    "elapsed_observation_seconds": [0.0, .02, .02],
                },
            },
        },
    }


def record_paired_inputs(destination, member, decision_fixed_step,
                         legacy, corrected, history_rows):
    """Seal PNGs and pairing metadata before intervention."""
    destination = Path(destination)
    manifest = build_paired_input_manifest(
        member, decision_fixed_step, legacy, corrected, history_rows)
    destination.mkdir(parents=True)
    pngs = {
        "legacy-single/frame_000001.png": legacy.png,
        "corrected-single/frame_000001.png": corrected.canonical_png,
    }
    pngs.update({f"corrected-history/frame_{ordinal:06}.png": row["canonical_png"]
                 for ordinal, row in enumerate(history_rows, 1)})
    for relative_path, png in pngs.items():
        path = destination / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    shared.files.write(destination / PAIR_MANIFEST, manifest)
    return manifest


def assemble_visual_carriers(view_name, parser_output, timestamps):
    """Apply the frozen carrier transform without inventing temporal frames."""
    expected = 1 if view_name in SINGLE_VIEWS else 3 if view_name == HISTORY_VIEW else None
    if expected is None:
        raise ValueError("unknown bounded-transfer input view")
    timestamps = torch.as_tensor(timestamps, dtype=torch.float64)
    if timestamps.ndim != 1 or len(timestamps) != expected:
        detail = "exactly one actual frame" if expected == 1 else "exactly three actual frames"
        raise ValueError(f"{view_name} requires {detail}")
    if any(value.shape[0] != expected for value in parser_output.values()):
        raise ValueError("parser outputs do not match the view's actual frame count")
    carriers = visual_carriers(parser_output, timestamps)
    if carriers.shape != (expected, VISUAL_DIM):
        raise ValueError("visual carrier assembly returned the wrong shape")
    if expected == 1:
        slot_features = carriers[:, 2:].reshape(1, -1, 13)
        if bool((carriers[:, :2] != 0).any()) or bool((slot_features[:, :, 8:11] != 0).any()):
            raise ValueError("single-frame input fabricated prior or motion channels")
    return carriers


class _PairedEndpoint:
    def __init__(self, delegate, recorder):
        self.delegate = delegate
        self.recorder = recorder
        self.captured = False

    def get_observation_capture(self):
        if self.captured:
            return self.delegate.get_observation_capture()
        legacy = self.delegate.get_physics_capture_v1()
        corrected = self.delegate.get_observation_capture()
        self.recorder(legacy, corrected)
        self.captured = True
        return corrected

    def disconnect(self):
        self.delegate.disconnect()


def capture_one(output, member, limits):
    """Run the frozen v4 branch collector while additively sealing three views."""
    output = Path(output)
    attempt = output / "attempts" / member["identity"]
    paired_manifest = None

    def seal(legacy, corrected):
        nonlocal paired_manifest
        _, rows = shared.read_history(attempt / "aligned", limits["decision_fixed_step"])
        paired_manifest = record_paired_inputs(
            attempt / "paired-inputs", member, limits["decision_fixed_step"],
            legacy, corrected, rows)

    original_endpoint = shared.ScienceBirdsBridge
    def endpoint_factory(*args, **kwargs):
        endpoint = _PairedEndpoint(original_endpoint(*args, **kwargs), seal)
        return endpoint

    with patch.object(shared, "ScienceBirdsBridge", endpoint_factory):
        result = shared.capture_one(output, member, limits)
    result["schema"] = "issue_76_bounded_transfer_capture_v1"
    result["paired_input_manifest"] = None if paired_manifest is None else paired_manifest["identity"]
    result["paired_input_views"] = [] if paired_manifest is None else list(paired_manifest["views"])
    if result["complete"] and paired_manifest is None:
        result["failure"] = "ValueError: paired input capture was not sealed before the shot"
        result["complete"] = False
    shared.files.write(output / "results" / (member["identity"] + ".json"), result)
    print(f"Bounded-transfer {member['identity']} complete={result['complete']} "
          f"failure={result['failure']}", flush=True)
    return result
