"""Issue #113 item 1: the frozen observable macro predicates."""
import gzip
import json

from world_model.data import observable_macro as om


def synthetic(moving, support_index, removed_step=300, onset_step=400):
    return {"schema": om.SCHEMA_SUMMARY, "frame_steps": [100 + 50 * f for f in range(len(moving))],
            "moving": moving, "launch_step": 150, "bird_removed_step": removed_step, "support_index": support_index,
            "support_sets": [[f"s{i}"] for i in range(max(support_index) + 1)], "contact_onset_step": onset_step}


def test_settle_needs_rest_over_the_window_after_the_bird_is_removed():
    summary = synthetic([0] * 5 + [1] * 10 + [0] * 25, [0] * 40)
    values, available = om.settle_labels(summary, 3, (1, 5))
    assert values[:, 0].tolist() == [False] * 17 + [True] * 23
    assert values[:, 1].tolist() == [False] * 19 + [True] * 21        # D = max(3, 5) = 5
    assert not available[:3, 0].any() and available[3:, 0].all()
    assert not available[:5, 1].any() and available[5:, 1].all()


def test_rest_before_the_bird_is_removed_is_not_settled():
    summary = synthetic([0] * 30, [0] * 30, removed_step=100 + 50 * 20)
    values, _ = om.settle_labels(summary, 3, (1,))
    assert values[:23, 0].sum() == 0 and values[23:, 0].all()
    assert not om.settle_labels(synthetic([0] * 30, [0] * 30, removed_step=None), 3, (1,))[0].any()


def test_structure_change_compares_endpoints():
    summary = synthetic([0] * 40, [0] * 20 + [1] * 5 + [0] * 15)
    values, available = om.change_labels(summary, (1, 5))
    assert values[:, 0].nonzero()[0].tolist() == [20, 25]
    assert values[:, 1].nonzero()[0].tolist() == list(range(20, 30))
    assert not available[:5, 1].any()


def test_phases_and_labels():
    summary = synthetic([0] * 5 + [1] * 10 + [0] * 25, [0] * 20 + [1] * 20)
    labels = om.shot_labels(summary, 3, (1, 5))
    assert labels["phase"].tolist() == [0] * 6 + [1] * 11 + [2] * 23
    assert labels["values"].shape == (40, 2, 2)
    assert not (labels["values"] & ~labels["available"]).any()


def test_debounce_rule_picks_smallest_k_under_the_bar():
    relapsing = synthetic([0] * 5 + [1] * 5 + [0] * 3 + [1] + [0] * 40, [0] * 54)
    chosen, table = om.select_debounce([relapsing], candidates=(2, 5), maximum=0.05)
    assert table[0]["relapsed"] > 0 and table[1]["relapsed"] == 0
    assert chosen == 5


def test_trace_summary_debounces_support_spikes(tmp_path):
    def sample(step, supports):
        bird = {"body_type": "dynamic", "simulated": True, "velocity": [1.0 if step < 40 else 0.0, 0.0],
                "angular_velocity_degrees_per_second": 0.0}
        block = {"body_type": "dynamic", "simulated": True, "velocity": [0.0, 0.0],
                 "angular_velocity_degrees_per_second": 5.0 if step == 90 else 0.0}
        return {"fixed_step": step, "contacts": [],
                "entities": [{"entity_id": "runtime:bird:0000", "body_present": True, "body": bird},
                             {"entity_id": "runtime:block:0000", "body_present": True, "body": block}],
                "supports": [{"supporter_entity_id": a, "supported_entity_id": b} for a, b in supports]}
    steps = list(range(0, 101))
    pair = ("runtime:block:0000", "runtime:pig:0000")
    samples = [sample(s, [pair] if (s >= 60 or s == 30) else []) for s in steps]
    chunk = {"fixed_step_samples": samples,
             "events": [{"fixed_step": 10, "event_type": "bird_launched", "participants": ["runtime:bird:0000"]},
                        {"fixed_step": 70, "event_type": "entity_destroyed", "participants": ["runtime:bird:0000"]}]}
    (tmp_path / "native").mkdir()
    with gzip.open(tmp_path / "native" / "c.json.gz", "wb") as stream:
        stream.write(json.dumps(chunk).encode())
    manifest = {"complete_every_native_step": True, "status": "complete", "first_fixed_step": 0,
                "frame_records": [{"fixed_step": s} for s in (0, 50, 100)], "chunks": [{"path": "native/c.json.gz"}]}
    (tmp_path / "native-manifest.json").write_text(json.dumps(manifest))
    summary = om.trace_summary(tmp_path)
    assert summary["support_index"] == [0, 0, 1]      # the step-30 spike is filtered; 60..100 holds the pair
    assert summary["moving"] == [1, 1, 1] and summary["moving_nonbird"] == [0, 0, 1]
    assert summary["launch_step"] == 10 and summary["bird_removed_step"] == 70
    assert summary["destroyed"] == [[70, ["runtime:bird:0000"]]]
