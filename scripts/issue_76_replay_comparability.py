"""Prospective operational comparability checks, not hidden-state identity claims."""
import numpy as np

LIMITS = {
    "camera_matrix_absolute_tolerance": 1e-6,
    "rgb_mean_absolute_difference_255": 1.,
    "rgb_large_difference_threshold_255": 8,
    "rgb_large_difference_pixel_fraction": .01,
    "body_position_absolute_tolerance": .01,
    "body_velocity_absolute_tolerance": .01,
    "body_rotation_degrees_tolerance": 1.,
    "body_angular_velocity_degrees_tolerance": 1.,
}


def compare(reference, candidate, limits=LIMITS):
    """Inputs contain persisted evidence only; no task outcomes or model scores.

    All captured bodies, including birds, participate. Runtime-specific contact
    identifiers and absolute clock origins are not physical reset identities.
    Timing gaps are reported, not silently replaced by a zero-duration gap.
    """
    failures = []
    for key in ("base_cluster", "scenario", "generation_seed", "engine_seed", "exposure_role",
                "observation_configuration", "viewport"):
        if reference[key] != candidate[key]:
            failures.append("source_" + key)
    gaps = []
    for value in (reference, candidate):
        times = np.asarray([value["decision_time"], value["capture_time"]], dtype=float)
        if not np.isfinite(times).all() or times[0] > times[1]:
            failures.append("noncausal_decision_capture_time")
            gaps.append(None)
        else:
            gaps.append(float(times[1] - times[0]))
    left, right = reference["transform"], candidate["transform"]
    if set(left) != set(right):
        failures.append("camera_transform_fields")
    else:
        for key in left:
            if key.endswith("_matrix"):
                a, b = np.asarray(left[key]), np.asarray(right[key])
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0, atol=limits["camera_matrix_absolute_tolerance"]):
                    failures.append("camera_" + key)
            elif left[key] != right[key]:
                failures.append("camera_" + key)
    images = [np.asarray(value["rgb"]) for value in (reference, candidate)]
    image_mean = large_fraction = None
    if any(image.shape != (480, 640, 3) or image.dtype != np.uint8 or np.ptp(image) == 0 for image in images):
        failures.append("invalid_or_constant_decision_rgb")
    else:
        difference = np.abs(images[0].astype(np.int16) - images[1].astype(np.int16))
        image_mean = float(difference.mean())
        large_fraction = float((difference.max(-1) > limits["rgb_large_difference_threshold_255"]).mean())
        if image_mean > limits["rgb_mean_absolute_difference_255"] or large_fraction > limits["rgb_large_difference_pixel_fraction"]:
            failures.append("decision_rgb_difference")
    if reference["initial"]["world"] != candidate["initial"]["world"]:
        failures.append("native_world_configuration")
    entities = []
    for value in (reference, candidate):
        rows = value["initial"]["entities"]
        mapping = {row["scenario_object_id"]: row for row in rows}
        if len(mapping) != len(rows):
            failures.append("duplicate_initial_entity")
        entities.append(mapping)
    if set(entities[0]) != set(entities[1]):
        failures.append("initial_entity_inventory")
    maxima = {key: 0. for key in ("position", "velocity", "rotation_degrees", "angular_velocity_degrees_per_second")}
    for identity in sorted(set(entities[0]) & set(entities[1])):
        a, b = entities[0][identity], entities[1][identity]
        if a["lifecycle"] != b["lifecycle"] or a["body_present"] != b["body_present"]:
            failures.append("initial_lifecycle_or_body:" + identity)
        if a["body"] is None or b["body"] is None:
            if a["body"] != b["body"]:
                failures.append("initial_body_availability:" + identity)
            continue
        for key in ("body_type", "simulated", "gravity_scale", "gravity_applicable"):
            if a["body"][key] != b["body"][key]:
                failures.append("initial_body_configuration:" + identity + ":" + key)
        for key, tolerance in (("position", "body_position_absolute_tolerance"),
                               ("velocity", "body_velocity_absolute_tolerance"),
                               ("rotation_degrees", "body_rotation_degrees_tolerance"),
                               ("angular_velocity_degrees_per_second", "body_angular_velocity_degrees_tolerance")):
            av, bv = np.asarray(a["body"][key]), np.asarray(b["body"][key])
            if av.shape != bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all():
                failures.append("invalid_initial_body_value:" + identity + ":" + key)
                continue
            difference = np.abs(av - bv)
            if key == "rotation_degrees":
                difference = np.abs((av - bv + 180.) % 360. - 180.)
            maximum = float(difference.max())
            maxima[key] = max(maxima[key], maximum)
            if maximum > limits[tolerance]:
                failures.append("initial_body_difference:" + identity + ":" + key)
    return {"comparable": not failures, "failures": failures,
            "decision_to_capture_seconds": gaps, "rgb_mean_absolute_difference_255": image_mean,
            "rgb_large_difference_pixel_fraction": large_fraction, "maximum_body_differences": maxima,
            "all_birds_included": True, "hidden_state_identity_proven": False,
            "task_outcomes_or_model_scores_used": False}
