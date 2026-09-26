from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from scripts import capture_pipeline_v2 as pipeline


def png(array):
    stream = BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def state(*nodes):
    return {"coordinates": {"screen_origin": "top_left"},
            "nodes": [{"object_class": cls, "screen_polygon": [{"x": x0, "y": y0}, {"x": x1, "y": y1}]}
                      for cls, x0, y0, x1, y1 in nodes]}


def test_port_blocks_never_overlap_across_slots_or_sequences():
    seen = {}
    for slot in range(8):
        for sequence in range(pipeline.PORT_SUBBLOCKS):
            for port in pipeline.slot_ports(slot, sequence):
                assert port not in seen, (slot, sequence, seen.get(port))
                seen[port] = (slot, sequence)
    assert max(seen) < 32768  # stays below the Linux ephemeral range


def test_character_boxes_cover_birds_and_pigs_only():
    boxes = pipeline.character_boxes(state(("Bird", 94, 249, 102, 256), ("PigSmall", 363, 222, 371, 229),
                                           ("Rect", 10, 10, 20, 20), ("Slingshot", 92, 248, 104, 286)))
    pad = pipeline.CHARACTER_BOX_PAD_PIXELS
    assert boxes == [[94 - pad, 249 - pad, 102 + pad, 256 + pad], [363 - pad, 222 - pad, 371 + pad, 229 + pad]]


def test_character_boxes_refuse_mismatched_screen_origin():
    with pytest.raises(ValueError):
        pipeline.character_boxes({"coordinates": {"screen_origin": "bottom_left"}, "nodes": []})


def test_render_invariance_accepts_blink_inside_character_box():
    reference = np.full((480, 640, 3), 200)
    blink = reference.copy()
    blink[250:255, 96:101] = 0  # eye pixels inside the bird box
    evidence = pipeline.render_invariance(png(reference), png(blink), [[91, 246, 105, 259]])
    assert evidence == {"byte_equal": False, "differing_pixels": 25, "outside_pixels": 0}


def test_render_invariance_rejects_any_pixel_outside_boxes():
    reference = np.full((480, 640, 3), 200)
    moved = reference.copy()
    moved[250:255, 96:101] = 0
    moved[300, 300] = 0  # a block pixel changed: real state change
    with pytest.raises(ValueError, match="outside animated character sprites"):
        pipeline.render_invariance(png(reference), png(moved), [[91, 246, 105, 259]])


def test_box_edges_are_inclusive():
    reference = np.full((20, 20, 3), 200)
    edge = reference.copy()
    edge[15, 15] = 0
    assert pipeline.render_invariance(png(reference), png(edge), [[5, 5, 15, 15]])["outside_pixels"] == 0
    edge[16, 15] = 0
    with pytest.raises(ValueError):
        pipeline.render_invariance(png(reference), png(edge), [[5, 5, 15, 15]])


@pytest.mark.parametrize("failure,expected", [
    (None, None),
    ("PortBindConflict: engine log sciencebirds_29102.log: Address already in use", "port_bind_conflict"),
    ("TimeoutError: timed out", "socket_timeout"),
    ("TimeoutError: Science Birds did not reach PLAYING before timeout", "menu_readiness_timeout"),
    ("TimeoutError: native shot progress stalled for 120s at chunks=3 RGB=10; partial capture retained",
     "shot_progress_stall"),
    ("ValueError: decision RGB differs outside animated character sprites (4 pixels)",
     "render_invariance_violation"),
    ("attempt_wall_limit", "attempt_wall_limit"),
    ("ValueError: something new", "other"),
])
def test_failure_taxonomy(failure, expected):
    assert pipeline.failure_class(failure) == expected
