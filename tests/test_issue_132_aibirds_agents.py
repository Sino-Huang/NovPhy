"""CPU tests for the #132 E1 runner: level lists, request parsing, shot recording/transfer, verdict parsing."""
import hashlib
import json
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


def test_engine_window_is_daytime_melbourne_only():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(e1.ENGINE_WINDOW_TZ)
    assert e1.engine_minutes_left(datetime(2026, 10, 25, 7, 4, tzinfo=tz)) == 0.0
    assert e1.engine_minutes_left(datetime(2026, 10, 25, 7, 5, tzinfo=tz)) == pytest.approx(625.0)
    assert e1.engine_minutes_left(datetime(2026, 10, 25, 17, 0, tzinfo=tz)) == pytest.approx(30.0)
    assert e1.engine_minutes_left(datetime(2026, 10, 25, 17, 30, tzinfo=tz)) == 0.0
    assert e1.engine_minutes_left(datetime(2026, 10, 25, 23, 0, tzinfo=tz)) == 0.0


# --- publication (synthetic N1 records; never real N1 content) ---------------------------------------------------

SYN_LEVELS = [f"lvl-{i}" for i in range(6)]
SYN_ATTEMPTS = {"datalab": 3, "eaglewing": 3, "pigshooter": 1}
# outcome per (agent, level index) -> list of per-attempt verdicts: True pass, False fail, None no shot, "x" failed exec
SYN_OUTCOMES = {
    "datalab": [[True, False, False], [False, True, False], [False, False, False], [None, True, True],
                [False, False, True], ["x", True, True]],
    "eaglewing": [[False, False, False], [True, True, True], [False, False, True], [False, True, False],
                  [False, False, False], [False, False, False]],
    "pigshooter": [[True], [False], [True], [None], [False], [True]],
}


def make_run(root, stop=None):
    plan = {"schema": "issue_132_n1_run_plan_v1", "attempts_per_level": SYN_ATTEMPTS, "chunk_levels": 2,
            "level_order": SYN_LEVELS, "level_order_sha256": "sha256:x", "allowance_worker_hours": 190.0}
    e1.write_json(root / "n1" / "run-plan.json", plan)
    if stop:
        e1.write_json(root / "n1" / "budget-stop.json", stop)
    anchor = {"gameX": 100.0, "gameY": 200.0}
    for agent, levels in SYN_OUTCOMES.items():
        for index, outcomes in enumerate(levels):
            for attempt, outcome in enumerate(outcomes):
                level = SYN_LEVELS[index]
                shot = None if outcome is None else {"code": 31, "polar": False, "fx": 10 * attempt + index,
                                                     "fy": 300 + len(agent), "t1": 0, "t2": 0}
                record = {"agent": agent, "level": level, "repeat": attempt, "status": "complete", "shot": shot,
                          "anchor": anchor, "wall_seconds": 20.0, "decision_seconds": 5.0,
                          "native": None if outcome is None else {"LevelStatus": "Pass" if outcome is True else "Fail"}}
                e1.write_json(root / "n1" / "stage-a" / agent / f"{level}-r{attempt}" / "record.json", record)
                execution = e1.execution_of(record)
                if execution is not None:
                    failed = outcome == "x"
                    e1.write_json(root / "n1" / "verdicts" / f"{execution}.json",
                                  {"execution": execution, "status": "failed" if failed else "complete",
                                   "pig_removed": None if failed else outcome is True, "wall_seconds": 100.0,
                                   "wire": shot})
    return plan


def expected_block(values_by_level):
    from scripts import run_launch_power_probe as p94
    levels = sorted(values_by_level)
    return p94.bootstrap([values_by_level[level] for level in levels], levels)


def test_pass_at_k_follows_the_102_episode_rule(tmp_path):
    plan = make_run(tmp_path)
    _, stop, records, verdicts = e1.load_run(tmp_path)
    assert stop is None
    summary = e1.agent_summary(plan, records, verdicts, "datalab")
    # level 3: no shot at attempt 1 ends the episode (typed failure); level 5: failed execution ends it
    pass1 = {"lvl-0": 1.0, "lvl-1": 0.0, "lvl-2": 0.0, "lvl-3": 0.0, "lvl-4": 0.0, "lvl-5": 0.0}
    pass3 = {"lvl-0": 1.0, "lvl-1": 1.0, "lvl-2": 0.0, "lvl-3": 0.0, "lvl-4": 1.0, "lvl-5": 0.0}
    assert summary["pass1"] == expected_block(pass1)
    assert summary["pass3"] == expected_block(pass3)
    assert summary["pass1"]["clusters"] == 6 and summary["pass1"]["interval"][0] <= summary["pass1"]["mean"]
    assert summary["outcomes"] == {"failed": 1, "passed": 3, "typed_failure": 2}
    assert summary["typed_failures"] == {"no_shot": 1, "stage_b_failed": 1}
    deterministic = e1.agent_summary(plan, records, verdicts, "pigshooter")
    assert deterministic["pass3"] is None and deterministic["levels"]["pass3"] == 0
    assert deterministic["pass1"] == expected_block({"lvl-0": 1.0, "lvl-1": 0.0, "lvl-2": 1.0, "lvl-3": 0.0,
                                                     "lvl-4": 0.0, "lvl-5": 1.0})


def test_allowance_stop_restricts_pass3_to_levels_with_all_attempts(tmp_path):
    plan = make_run(tmp_path, stop={"attempt": 3, "chunk_start": 2})
    for level in SYN_LEVELS[2:]:   # attempt 3 never ran past the first chunk
        for agent in ("datalab", "eaglewing"):
            record_path = tmp_path / "n1" / "stage-a" / agent / f"{level}-r2" / "record.json"
            record_path.unlink()
    _, stop, records, verdicts = e1.load_run(tmp_path)
    summary = e1.agent_summary(plan, records, verdicts, "eaglewing")
    assert summary["levels"] == {"pass1": 6, "pass3": 2}
    assert summary["pass3"] == expected_block({"lvl-0": 0.0, "lvl-1": 1.0})
    assert summary["pass1"] == expected_block({"lvl-0": 0.0, "lvl-1": 1.0, "lvl-2": 0.0, "lvl-3": 0.0,
                                               "lvl-4": 0.0, "lvl-5": 0.0})
    assert len(e1.planned_jobs(plan, stop)) == 6 * 3 + 6 * 2 + 2 * 2


def test_publish_refuses_an_incomplete_run(tmp_path):
    make_run(tmp_path)
    (tmp_path / "n1" / "stage-a" / "eaglewing" / "lvl-4-r2" / "record.json").unlink()
    with pytest.raises(e1.IncompleteRun):
        e1.load_run(tmp_path)
    make_run(tmp_path)
    next(iter(sorted((tmp_path / "n1" / "verdicts").glob("*.json")))).unlink()
    with pytest.raises(e1.IncompleteRun):
        e1.publish(tmp_path, a3_draft=tmp_path / "draft.md")
    assert not (tmp_path / "summary.json").exists()


def test_publish_then_validate_round_trip(tmp_path):
    make_run(tmp_path)
    texts = e1.publish(tmp_path, a3_draft=tmp_path / "draft.md")
    summary = json.loads(texts["summary.json"])
    assert summary["schema"] == "issue_132_e1_summary_v1"
    assert set(summary["agents"]) == set(e1.AGENTS) and summary["agents"]["datalab"]["pass3"]["clusters"] == 6
    stage_a = len(list((tmp_path / "n1" / "stage-a").glob("*/*/record.json")))
    assert summary["engine_worker_hours"] == round(stage_a * 20.0 / 3600, 3)
    assert not any(word in texts["findings.md"].lower() for word in ("beats", "outperform", "sota"))
    assert "Relational-Symbolic's native-task fidelity run" in (tmp_path / "draft.md").read_text()
    assert e1.validate(tmp_path, recheck=0) == 0
    record = tmp_path / "n1" / "stage-a" / "datalab" / "lvl-0-r0" / "record.json"
    record.write_text(record.read_text().replace('"wall_seconds": 20.0', '"wall_seconds": 21.0'))
    assert e1.validate(tmp_path, recheck=0) == 1


def test_a3_funding_draft_threshold():
    assert "is funded" in e1.a3_funding_draft(185.0)
    assert "tokens-only, typed `engine_budget`" in e1.a3_funding_draft(185.01)
