"""CPU tests for the #132 E1 runner: level lists, request parsing, shot recording/transfer, verdict parsing."""
import hashlib
import re
import struct

import pytest

from scripts import run_issue_132_aibirds_agents as e1
from scripts.slingshot_readiness import _anchored_action_and_command

# Bytes a live Datalab session sent through the relay (stage-A probe, fit level 001), in order.
DATALAB_REQUESTS = ["0100005b3501", "0200000032", "0c", "44", "0c", "44", "0c", "0c", "35", "45", "0c", "40",
                    "22", "40", "1fffffff18000001b70000000000000000", "0c"]
DRAG_LINE = ("DragDetails:20261024202355: levelName :9001_Data/StreamingAssets/Levels/novelty_level_0/type010103/"
             "Levels/issue-109-sc-fit-type010103-001.xml: trial :0: dragx :97.82841: dragy :227.2: dragDX :-232: "
             "DragDY :439: TapTime :0: Score :0,end")
EVALUATION_CSV = ("EvaluationIndex,LevelIndex,Score,LevelStatus,birdsRemaining,pigsRemaining,birdsAtStart,pigsAtStart,"
                  " trial,levelName,playStatus,informed,novelty\n"
                  "1,1,110,Fail,0,1,1,1, 0,9001_Data/StreamingAssets/Levels/x.xml,training,uninformed,basic\n")


@pytest.fixture(scope="module")
def plan():
    return e1.cohort_plan()


def test_n1_is_the_400_single_bird_evaluation_levels(plan):
    levels = e1.n1_levels(plan)
    names = [level["identity"] for level in levels]
    assert len(names) == len(set(names)) == 400
    pattern = re.compile(r"issue-109-sc-evaluation-(type010103|type010105)-(\d{3})")
    families = {}
    for name in names:
        match = pattern.fullmatch(name)
        assert match, name
        families.setdefault(match.group(1), set()).add(int(match.group(2)))
    assert families == {"type010103": set(range(1, 201)), "type010105": set(range(1, 201))}
    assert all(len(e1.bird_slots(level)) == 1 for level in levels)
    # pinned level files: hash only (metadata), never parsed here
    for level in levels:
        assert hashlib.sha256((e1.COHORT / level["xml_path"]).read_bytes()).hexdigest() == level["xml_sha256"]


def test_smoke_levels_are_fit_only(plan):
    levels = e1.smoke_levels(plan)
    assert len(levels) == 20
    assert all(level["split"] == "fit" for level in levels)
    assert {level["generator_family"] for level in levels} == {"type010103", "type010105"}
    assert sorted(level["ordinal"] for level in levels) == sorted(list(range(1, 11)) * 2)
    evaluation = {level["identity"] for level in e1.n1_levels(plan)}
    assert not evaluation & {level["identity"] for level in levels}
    assert all(len(e1.bird_slots(level)) == 1 for level in levels)


def test_parse_requests_splits_a_recorded_datalab_stream():
    stream = b"".join(bytes.fromhex(item) for item in DATALAB_REQUESTS)
    parsed, rest = e1.parse_requests(stream)
    assert rest == b""
    assert [bytes([code]) + payload for code, payload in parsed] == [bytes.fromhex(i) for i in DATALAB_REQUESTS]
    # a request split across reads is kept until complete
    head, tail = stream[:3], stream[3:]
    first, rest = e1.parse_requests(head)
    assert first == [] and rest == head
    parsed, rest = e1.parse_requests(rest + tail)
    assert len(parsed) == len(DATALAB_REQUESTS) and rest == b""
    # variable-length novelty reports take the rest of the write
    parsed, rest = e1.parse_requests(bytes([66]) + b"\x00" * 23)
    assert parsed == [(66, b"\x00" * 23)] and rest == b""


def test_decode_shot_records_the_wire_values():
    code, payload = e1.parse_requests(bytes.fromhex("1fffffff18000001b70000000000000000"))[0][0]
    shot = e1.decode_shot(code, payload)
    assert (shot["name"], shot["fx"], shot["fy"], shot["t1"], shot["t2"], shot["polar"]) == \
        ("cshoot", -232, 439, 0, 0, False)
    fast = e1.decode_shot(41, struct.pack("!iiii", 120, 300, 0, 650))
    assert (fast["name"], fast["fx"], fast["fy"], fast["t2"]) == ("cfastshoot", 120, 300, 650)
    assert e1.decode_shot(32, struct.pack("!iiii", 45, 90, 0, 0))["polar"]


@pytest.mark.parametrize("anchor", [
    {"gameX": 97.83, "gameY": 252.8, "canvasX": 97.83, "canvasY": 227.2, "pixelsPerWorldUnit": 9.1},
    {"gameX": 180.0, "gameY": 300.49, "canvasX": 180.0, "canvasY": 179.51, "pixelsPerWorldUnit": 7.0},
])
@pytest.mark.parametrize("wire", [(-232, 439, 0, 0), (40, 300, 0, 650), (97, 228, 1000, 0), (-600, 560, 0, 0)])
def test_shot_transfer_reproduces_the_wire_release_point(anchor, wire):
    shot = {"code": 31, "polar": False, "fx": wire[0], "fy": wire[1], "t1": wire[2], "t2": wire[3]}
    action = e1.shot_action(shot, anchor)
    requested = {"drag_release": [action["drag_x"], action["drag_y"]], "coordinate_frame": "slingshot_relative",
                 "tap_time": action["tap_time_ms"], "release_time": action["release_time_ms"]}
    _, command = _anchored_action_and_command(requested, anchor, e1.FRAME_HEIGHT)
    assert (command["x"], command["y"], command["tapTime"]) == (wire[0], wire[1], wire[3])
    assert e1.execution_suffix(action) == e1.execution_suffix(dict(action))


def test_polar_shots_are_not_transferred():
    with pytest.raises(ValueError):
        e1.shot_action({"polar": True, "fx": 1, "fy": 2, "t1": 0, "t2": 0}, {"gameX": 1, "gameY": 1})


def test_engine_verdict_and_drag_parsing(tmp_path):
    (tmp_path / "sciencebirds_39401.log").write_text("noise\n" + DRAG_LINE + "\nmore\n")
    details = e1.drag_details(tmp_path)
    assert details == [{"dragx": 97.82841, "dragy": 227.2, "dragDX": -232.0, "DragDY": 439.0, "TapTime": 0.0,
                        "Score": 0.0}]
    evaluation = tmp_path / "Agent_noname_Port_39401_Evaluation_1"
    evaluation.mkdir()
    (evaluation / "Agent_noname_Trial_0_uninformed_1_EvaluationData.csv").write_text(EVALUATION_CSV)
    rows = e1.evaluation_rows(tmp_path)
    assert len(rows) == 1
    assert (rows[0]["LevelStatus"], rows[0]["pigsRemaining"], rows[0]["trial"]) == ("Fail", "1", "0")


def test_ground_truth_decoding_ignores_the_trailer():
    body = b'[{"features": [{"properties": {"label": "Slingshot"}, "geometry": {"type": "Polygon", ' \
           b'"coordinates": [[[1, 2], [3, 2], [3, 9], [1, 9]]]}}]}]\r\nEND'
    truth = e1.decode_ground_truth(body)
    assert e1.slingshot_vertices(truth) == [[[1, 2], [3, 2], [3, 9], [1, 9]]]


def test_phyq_config_has_phyq_attributes_and_the_repeated_level():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(e1.phyq_config("9001_Data/StreamingAssets/Levels/x.xml"))
    level_set = root.find("trials/trial/game_level_set")
    assert level_set.attrib == e1.PHYQ_SET
    assert root.find("trials/trial").attrib == e1.PHYQ_TRIAL
    assert [g.attrib["level_path"] for g in level_set] == ["9001_Data/StreamingAssets/Levels/x.xml"] * 2


def test_determinism_and_projection():
    def rec(agent, level, repeat, fx):
        return {"agent": agent, "level": level, "repeat": repeat,
                "shot": None if fx is None else {"code": 31, "fx": fx, "fy": 1, "t1": 0, "t2": 0}}
    records = [rec("a", "l1", 0, 5), rec("a", "l1", 1, 5), rec("a", "l2", 0, 7), rec("a", "l2", 1, 7),
               rec("b", "l1", 0, 5), rec("b", "l1", 1, 6), rec("b", "l2", 0, None), rec("b", "l2", 1, 3)]
    det = e1.determinism(records)
    assert det["a"] == {"levels_compared": 2, "identical": 2, "deterministic": True}
    assert det["b"] == {"levels_compared": 1, "identical": 0, "deterministic": False}
    proj = e1.projection({"a": 30.0, "b": 60.0}, {"a": 180.0, "b": 180.0}, {"a": True, "b": False})
    assert proj["agents"]["a"]["executions"] == 400 and proj["agents"]["b"]["executions"] == 1200
    expected = (400 * 210 + 1200 * 240) / 3600
    assert proj["worker_hours"] == pytest.approx(expected, abs=0.01)
    assert proj["remainder_for_a3_worker_hours"] == pytest.approx(190 - expected, abs=0.01)
    assert proj["within_allowance"] is (expected <= 190)
