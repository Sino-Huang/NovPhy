"""Bounded reader for the lossless native-step trace, not the legacy 64MiB sidecar."""
import gzip
import json
import math
from pathlib import Path

MAX_CHUNK_BYTES = 16 * 2**20
CHUNK_SAMPLES = 250
MAX_STEPS = 30000
OBSERVATION_STRIDE = 50
FIXED_DELTA_SECONDS = .0004


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite native trace value")
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def validate_sample(sample):
    if sample.get("complete_raw_non_trigger_contacts") is not True:
        raise ValueError("native step lacks complete raw contact enumeration")
    entities = sample["entities"]
    ids = {e["entity_id"] for e in entities}
    if not ids or len(ids) != len(entities):
        raise ValueError("native entity inventory is empty or duplicated")
    for entity in entities:
        if entity["entity_id"] != "runtime:" + entity["scenario_object_id"]:
            raise ValueError("native entity lost its authored identity")
        if entity["lifecycle"] not in ("active", "inactive", "destroyed"):
            raise ValueError("unknown native entity lifecycle")
        if bool(entity["body_present"]) != (entity["body"] is not None):
            raise ValueError("native body availability differs")
    colliders = {c["collider_id"]: c for c in sample["colliders"]}
    if len(colliders) != len(sample["colliders"]) or any(c["entity_id"] not in ids for c in colliders.values()):
        raise ValueError("native collider inventory is duplicated or unresolved")
    contacts = sample["contacts"]
    if len({c["contact_id"] for c in contacts}) != len(contacts):
        raise ValueError("native contact IDs are duplicated")
    for contact in contacts:
        for side in ("a", "b"):
            entity = contact[f"entity_{side}_id"]
            collider = contact[f"collider_{side}_id"]
            if entity not in ids or collider not in colliders or colliders[collider]["entity_id"] != entity:
                raise ValueError("native contact has an unresolved participant")
    finite_tree(sample)


class NativeTrace:
    """Iterate one <=16MiB chunk at a time; no whole-shot physics object graph."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "native-manifest.json").read_text())
        value = self.manifest
        if value.get("schema") != "canonical_native_trace_v1":
            raise ValueError("unsupported native trace schema")
        if value.get("fixed_delta_seconds") != FIXED_DELTA_SECONDS or value.get("observation_stride") != OBSERVATION_STRIDE:
            raise ValueError("native time or observation cadence differs from the contract")
        if value.get("status") not in ("complete", "failed"):
            raise ValueError("native manifest has no declared completion status")
        if value["last_fixed_step"] - value["first_fixed_step"] > MAX_STEPS:
            raise ValueError("native trace exceeded its physical-time window")

    def chunks(self):
        for ordinal, descriptor in enumerate(self.manifest["chunks"], 1):
            expected = f"native/chunk-{ordinal:06}.json.gz"
            if descriptor["path"] != expected:
                raise ValueError("native chunk membership/order differs")
            path = self.root / expected
            if path.stat().st_size != descriptor["compressed_bytes"]:
                raise ValueError("native compressed chunk is incomplete")
            with gzip.open(path, "rb") as stream:
                data = stream.read(MAX_CHUNK_BYTES + 1)
            if len(data) > MAX_CHUNK_BYTES or len(data) != descriptor["uncompressed_bytes"]:
                raise ValueError("native chunk exceeds its decompressed bound or is incomplete")
            chunk = json.loads(data)
            if chunk.get("schema") != "canonical_native_chunk_v1" or chunk.get("capture_id") != self.manifest["capture_id"] or chunk.get("shot_id") != self.manifest["shot_id"]:
                raise ValueError("native chunk belongs to a different capture")
            samples = chunk["fixed_step_samples"]
            if not 1 <= len(samples) <= CHUNK_SAMPLES or len(samples) != descriptor["sample_count"]:
                raise ValueError("native chunk sample count differs")
            if samples[0]["fixed_step"] != descriptor["first_fixed_step"] or samples[-1]["fixed_step"] != descriptor["last_fixed_step"]:
                raise ValueError("native chunk endpoints differ")
            if len(chunk["events"]) != descriptor["event_count"]:
                raise ValueError("native chunk event count differs")
            if "terminal_evidence" in chunk:
                raise ValueError("a native file boundary must not masquerade as a gameplay terminal")
            yield chunk

    def validate(self):
        value = self.manifest
        expected = value["first_fixed_step"]
        frames = value["frame_records"]
        frame_steps = [f["fixed_step"] for f in frames]
        if not frame_steps or frame_steps[0] != expected or len(frame_steps) != len(set(frame_steps)):
            raise ValueError("native observation-frame inventory differs")
        scheduled = [f["fixed_step"] for f in frames if not f["forced_terminal"]]
        if any(b - a != OBSERVATION_STRIDE for a, b in zip(scheduled, scheduled[1:])):
            raise ValueError("native observation stride has a gap")
        if any(f["forced_terminal"] for f in frames[:-1]):
            raise ValueError("only the genuine final observation may be off-grid")
        if frames[-1]["forced_terminal"] and not 0 < frame_steps[-1] - scheduled[-1] < OBSERVATION_STRIDE:
            raise ValueError("native off-grid terminal is not between adjacent scheduled observations")
        event_ids, entities, event_types = set(), set(), {}
        terminal_match = False
        for chunk in self.chunks():
            for sample in chunk["fixed_step_samples"]:
                if sample["fixed_step"] != expected:
                    raise ValueError("native microstep coverage has a gap or overlap")
                validate_sample(sample)
                entities.update(e["entity_id"] for e in sample["entities"])
                expected += 1
            for event in chunk["events"]:
                if event["event_id"] in event_ids or not set(event["participants"]).issubset(entities):
                    raise ValueError("native event is duplicated or unresolved")
                if not value["first_fixed_step"] <= event["fixed_step"] <= value["last_fixed_step"]:
                    raise ValueError("native event lies outside the captured interval")
                finite_tree(event)
                event_ids.add(event["event_id"])
                name = event["event_type"]
                event_types[name] = event_types.get(name, 0) + 1
                terminal = value["terminal_evidence"]
                if terminal and event["event_id"] == terminal["event_id"]:
                    terminal_match = event["fixed_step"] == terminal["fixed_step"] and name == terminal["reason"]
        count = expected - value["first_fixed_step"]
        if count != value["sample_count"]:
            raise ValueError("native manifest sample total differs")
        complete = value["status"] == "complete"
        if complete:
            if value["failure"] is not None or value["complete_every_native_step"] is not True:
                raise ValueError("native complete status contradicts failure/coverage")
            if expected - 1 != value["last_fixed_step"] or frame_steps[-1] != value["last_fixed_step"] or not terminal_match:
                raise ValueError("native complete capture lacks its genuine terminal coverage")
        elif value["terminal_evidence"] is not None or not value["failure"]:
            raise ValueError("failed native capture invents a terminal or omits its failure")
        return {"complete": complete, "sample_count": count, "frame_count": len(frames),
                "event_counts": event_types, "first_fixed_step": value["first_fixed_step"],
                "last_fixed_step": value["last_fixed_step"],
                "physical_seconds": (value["last_fixed_step"] - value["first_fixed_step"]) * FIXED_DELTA_SECONDS,
                "failure": value["failure"]}

    def observed_samples(self):
        if self.manifest["status"] != "complete":
            raise ValueError("failed native traces are not complete training examples")
        wanted = {f["fixed_step"] for f in self.manifest["frame_records"]}
        for chunk in self.chunks():
            for sample in chunk["fixed_step_samples"]:
                if sample["fixed_step"] in wanted:
                    yield sample
