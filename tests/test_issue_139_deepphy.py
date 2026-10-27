"""CPU tests of #139 T7 DeepPHY-VLM (CUDA_VISIBLE_DEVICES=""): prompt / parse G-port against the official code, the
launch rendering, the score definition (logits processing, ties), the gates, the zero-shot modes on stubs, the not-run
publication and the projection math. No evaluation data is read (fit / policy stand-ins only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import external_baselines as eb  # noqa: E402
from scripts import run_issue_139_deepphy as r  # noqa: E402
from world_model.baselines.deepphy import novphy as dn  # noqa: E402

GRID_ANSWERS = ["[shoot(angle=7, power=1.0)]", "[shoot(angle=37, power=1.0)]", "[shoot(angle=54, power=1.0)]",
                "[shoot(angle=63, power=1.0)]", "[shoot(angle=4, power=1.0)]", "[shoot(angle=21, power=1.0)]",
                "[shoot(angle=35, power=1.0)]", "[shoot(angle=45, power=1.0)]", "[shoot(angle=2, power=1.0)]",
                "[shoot(angle=14, power=1.0)]", "[shoot(angle=25, power=1.0)]", "[shoot(angle=34, power=1.0)]",
                "[shoot(angle=2, power=1.0)]", "[shoot(angle=11, power=1.0)]", "[shoot(angle=19, power=1.0)]",
                "[shoot(angle=27, power=1.0)]"]


def reference():
    spec = importlib.util.spec_from_file_location("b139_port_reference", ROOT / "scripts/b127_deepphy_port_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------------------
# G-port: prompts, messages, parser (official code executed from the vendored source)
# ---------------------------------------------------------------------------------------------------------------

def test_prompts_messages_and_parser_match_the_official_code():
    record = reference().prompt_parse_gport(eb.grid_items())
    assert record["pass"]
    assert all(p["system_equal"] and p["user_equal"] and p["images_equal"] and p["messages_equal"]
               for p in record["prompts"])
    assert all(p["equal"] for p in record["parses"])
    assert record["prompts"][0]["official_call_kwargs"] == {"max_new_tokens": 200, "do_sample": True,
                                                            "temperature": 0.1, "return_full_text": False}
    assert record["official_crop_calls"] and record["official_crop_calls"][0]["crop_top_cm"] == 1.10


def test_first_shot_prompt_content():
    system, user = dn.prompts()
    assert system == dn.prompt_module().system_prompt_angry_birds
    assert "*   **Current Bird on Slingshot:** `red`" in user
    assert "*   **Remaining Birds in Queue:** `None`" in user
    assert "This is the first shot of the level. No actions have been taken yet." in user
    msgs = dn.messages(dn.image_descriptions(["/x.png"]), system, user)
    assert msgs[1]["content"] == [{"type": "text", "text": dn.CURRENT_LABEL}, {"type": "image", "url": "/x.png"},
                                  {"type": "text", "text": user}]


def test_launch_rendering_of_the_grid():
    items = eb.grid_items()
    assert [dn.answer_text(it["action"]) for it in items] == GRID_ANSWERS
    angle, power = dn.launch_angle_power({"drag_x": -40, "drag_y": 30})
    assert math.isclose(angle, math.degrees(math.atan2(30, 40))) and power == 1.0
    angle, power = dn.launch_angle_power({"drag_x": -9, "drag_y": 0})
    assert angle == 0.0 and math.isclose(power, 9 / dn.CLAMP_RADIUS_PX)
    assert dn.action_code({"drag_x": -9, "drag_y": 0}) == f"shoot(angle=0, power={round(9 / dn.CLAMP_RADIUS_PX, 2)})"
    for it, answer in zip(items, GRID_ANSWERS):
        parsed = dn.parse_response(answer)
        a, p = dn.launch_angle_power(it["action"])
        assert parsed["success"] is False and (parsed["angle"], parsed["power"]) == (round(a), round(p, 2))


def test_screenshot_is_2x_pixel_replication():
    frame = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    shot = dn.screenshot(frame)
    assert shot.shape == (8, 10, 3)
    assert np.array_equal(shot[::2, ::2], frame.transpose(1, 2, 0)) and np.array_equal(shot[1::2, 1::2], shot[::2, ::2])


# ---------------------------------------------------------------------------------------------------------------
# score definition
# ---------------------------------------------------------------------------------------------------------------

def test_logits_processing_matches_hf_processors():
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper
    g = torch.Generator().manual_seed(0)
    vocab, start = 40, 9
    ids = torch.randint(0, vocab, (13,), generator=g)
    rows = torch.randn(len(ids) - start, vocab, generator=g) * 3
    ours = dn.processed_rows(rows, ids, start, temperature=0.1, repetition_penalty=1.05)
    for j in range(len(rows)):
        prefix = ids[None, :start + j]
        ref = TemperatureLogitsWarper(0.1)(prefix, RepetitionPenaltyLogitsProcessor(1.05)(prefix, rows[j:j + 1].clone()))
        assert torch.allclose(ours[j], ref[0], atol=1e-5)
    plain = dn.answer_token_logprobs(rows, ids, start)
    expect = torch.log_softmax(rows, -1)[torch.arange(len(rows)), ids[start:]]
    assert torch.allclose(plain, expect)
    assert dn.answer_score(rows, ids, start) == float(expect.double().sum())
    assert dn.answer_score(rows, ids, start) < 0


class StubScorer(dn.Scorer):
    """Scorer without a model: answer_rows gives each answer string a fixed pseudo-random logit row set."""

    def __init__(self):
        self.calls = []

    def inputs(self, msgs):
        return {"input_ids": torch.zeros(1, 5, dtype=torch.long)}

    def answer_rows(self, model_inputs, answers):
        self.calls.append(list(answers))
        out = []
        for a in answers:
            g = torch.Generator().manual_seed(int(hashlib.sha256(a.encode()).hexdigest()[:12], 16))
            ids = torch.cat([torch.zeros(5, dtype=torch.long), torch.randint(0, 20, (4,), generator=g)])
            out.append((ids, torch.randn(4, 20, generator=g)))
        return out


def test_equal_renderings_tie_exactly_and_rows_rank_by_likelihood():
    scorer = StubScorer()
    items = eb.grid_items()
    scores, info = scorer.score("/x.png", [it["action"] for it in items],
                                {"q7b": {"temperature": 1.0, "repetition_penalty": None},
                                 "q7b-t01": {"temperature": 0.1, "repetition_penalty": 1.05}})
    assert len(scorer.calls) == 1 and len(scorer.calls[0]) == 15  # each distinct answer scored once
    for name in ("q7b", "q7b-t01"):
        assert scores[name][8] == scores[name][12]
        assert len(set(scores[name])) == 15
    rows = eb.score_rows(items, scores["q7b"], r.KEY)
    assert [row[r.KEY] for row in rows] == [-s for s in scores["q7b"]]
    assert rows[8][r.KEY] == rows[12][r.KEY]
    costs = np.asarray([[row[r.KEY] for row in rows]] * 3)
    rank = eb.g_rank(costs)
    assert rank["pass"] and math.isclose(rank["mean_tied_pair_share"], 1 / 120)


def test_nonfinite_score_is_an_excluded_row():
    rows = eb.score_rows(eb.grid_items()[:2], [float("-inf"), -3.0], r.KEY)
    assert rows[0]["excluded"] and rows[0][r.KEY] is None and rows[1][r.KEY] == 3.0


# ---------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------

def _no_worker(*args, **kwargs):
    raise AssertionError("the scoring worker must not run")


def test_gpu_modes_refuse_without_funding_and_g_fid(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "score_items", _no_worker)
    for mode in (r.selection, r.score, r.readouts, r.score_novelty, r.train):
        assert mode(tmp_path) == 1
    eb.write_json(tmp_path / "funded.json", {"funded": False, "decision": "x"})
    assert not r.funded(tmp_path) and r.selection(tmp_path) == 1
    eb.write_json(tmp_path / "funded.json", {"funded": True, "decision": "x"})
    assert r.funded(tmp_path)
    assert r.selection(tmp_path) == 1 and r.score(tmp_path) == 1  # G-port / G-fid missing
    eb.write_json(tmp_path / "gport/gport.json", {"pass": True})
    assert r.selection(tmp_path) == 1  # G-fid missing
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": False})
    assert r.selection(tmp_path) == 1 and r.score(tmp_path) == 1
    eb.write_json(tmp_path / "fidelity/gfid.json", {"pass": True})
    assert r.ladder(tmp_path)["pass"]


def _fake_response(items, names, device="cpu"):
    scores, seconds = {}, {}
    for k, (i, _, actions) in enumerate(items):
        base = [-(10.0 + ((k * 7 + j * 3) % 11)) for j in range(len(actions))]
        scores[i] = {n: [b * (1 + 0.1 * m) for b in base] for m, n in enumerate(names)}
        seconds[i] = 1.0 + 0.01 * k
    return {"scores": scores, "seconds": seconds, "prompt_tokens": {i: 1202 for i, _, _ in items},
            "load_seconds": 30.0, "seconds_total": 60.0, "device": device, "device_name": device,
            "model_dir": "stub", "dtype": "float16", "batch": 16, "torch": "x", "transformers": "4.52.4",
            "policy": dict(r.harness.SCORING_POLICY), "flops": {"decision_flops": 2.0e14, "candidates": 16}}


def test_timing_needs_no_funding(tmp_path, monkeypatch):
    def fake(output, tag, model, names, items, device=None, **kwargs):
        return _fake_response(items, names)

    monkeypatch.setattr(r, "score_items", fake)
    assert not (tmp_path / "funded.json").exists()
    assert r.timing(tmp_path, decisions=3) == 0
    record = eb.read_json(tmp_path / "timing.json")
    assert set(record["models"]) == set(dn.MODELS) and record["decisions"] == 3
    assert record["models"]["Qwen2.5-VL-7B-Instruct"]["configs"] == ["q7b", "q7b-t01"]
    assert record["models"]["Qwen2.5-VL-3B-Instruct"]["seconds_per_decision_median"] == pytest.approx(1.015)


# ---------------------------------------------------------------------------------------------------------------
# zero-shot contract modes on stubs (fit / policy data only)
# ---------------------------------------------------------------------------------------------------------------

def test_selection_readouts_apply_the_selection_rule(tmp_path):
    cells = eb.policy_cells("grid")
    perfect = {m: [1.0 if cells[m]["verdicts"][e["ordinal"]] else -1.0 for e in cells[m]["inventory"]] for m in cells}
    flat = {m: [0.0] * len(cells[m]["inventory"]) for m in cells}
    for name, scores in (("q7b", perfect), ("q7b-t01", flat), ("q3b", perfect)):
        eb.write_json(r.selection_dir(tmp_path, name) / "scores.json", scores)
    assert r.selection_readouts(tmp_path) == 0
    record = eb.read_json(tmp_path / "selection.json")
    assert record["results"]["q7b"]["auc"] == pytest.approx(1.0) and record["results"]["q7b-t01"]["auc"] == 0.5
    assert record["selected"] == "q3b"  # tie at 4 d.p. -> fewer parameters


def test_score_pass_builds_primary_and_wrong_anchor_rows(tmp_path, monkeypatch):
    cells_policy = eb.policy_cells("grid")
    members = sorted(cells_policy)[:3]
    frames = eb.policy_anchor_frames()
    fit = dict(list(sorted(eb.fit_anchor_frames().items()))[:2])
    cells = {inv: {"mixed": {m: cells_policy[m] for m in members}} for inv in eb.INVENTORIES}
    monkeypatch.setattr(eb, "fit_anchor_frames", lambda: fit)

    def fake(output, tag, model, names, items, device=None, **kwargs):
        return _fake_response(items, names)

    monkeypatch.setattr(r, "score_items", fake)
    body, response = r.score_pass(tmp_path, {"selected": "q7b"}, "stub", {"frames": frames}, cells)
    assert set(body["inventories"]) == set(eb.INVENTORIES)
    rows = body["inventories"]["grid"][members[0]]["rows"]
    assert [row[r.KEY] for row in rows] == [-s for s in response["scores"][f"grid/{members[0]}"]["q7b"]]
    assert set(body["wrong_anchor"]) == set(fit) and len(body["wrong_anchor"][sorted(fit)[0]]["rows"]) == 16
    assert body["wall_per_decision"] == pytest.approx(np.median([response["seconds"][f"grid/{m}"] for m in members]))


def test_score_novelty_refuses_before_primary_records_then_scores_like_the_primary_path(tmp_path, monkeypatch):
    eb.write_json(tmp_path / "funded.json", {"funded": True})
    monkeypatch.setattr(r, "load_plan", lambda output: {"selected": "q7b"})
    monkeypatch.setattr(r, "load_manifest", lambda output: {"frozen_at": "t"})
    monkeypatch.setattr(r, "score_items", _no_worker)
    assert r.score_novelty(tmp_path) == 1  # no primary records
    policy = eb.policy_cells("grid")
    levels = sorted(policy)[:2]
    frames = eb.policy_anchor_frames()
    cells = {"levels": {lv: policy[lv] for lv in levels}, "mixed": levels}
    for arm, seed in r.arm_seeds():
        eb.write_json(r.record_path(tmp_path, arm, seed), {"scored_at": "t"})
    monkeypatch.setattr(r, "load_records", lambda output: {})
    monkeypatch.setattr(r, "arm_guards", lambda output, records: {"T7": {"G_rank": True, "pass": True}})
    monkeypatch.setattr(r, "guard_extras", lambda output: {"G_port": True, "G_fid": True})
    monkeypatch.setattr(eb, "read_novelty_anchors", lambda output, policy, cells: {
        "frames": {lv: frames[lv] for lv in levels}, "carriers": {}, "read_at": "t2"})
    monkeypatch.setattr(eb, "novelty_reference_records", lambda *a, **k: {})
    seen = {}

    def fake(output, tag, model, names, items, device=None, **kwargs):
        seen["items"] = items
        return _fake_response(items, names)

    monkeypatch.setattr(r, "score_items", fake)
    assert r.score_novelty(tmp_path, cells=cells) == 0
    response = _fake_response(seen["items"], ["q7b"])
    for seed in eb.SEEDS:
        record = eb.read_json(r.novelty_record_path(tmp_path, "T7", seed))
        assert record["schema"] == r.SCHEMA_NOVELTY_RECORD
        for lv in levels:
            assert record["inventories"]["grid"][lv]["rows"] == eb.score_rows(
                policy[lv]["inventory"], response["scores"][lv]["q7b"], r.KEY)


def test_train_is_a_zero_shot_no_op(tmp_path, monkeypatch):
    eb.write_json(tmp_path / "funded.json", {"funded": True})
    monkeypatch.setattr(r, "load_plan", lambda output: {"selected": "q7b"})
    assert r.train(tmp_path) == 0
    for arm, seed in r.arm_seeds():
        report = eb.read_json(r.model_dir(tmp_path, arm, seed) / "report.json")
        assert report["zero_shot"] is True and report["gpu_seconds"] == 0.0 and report["steps"] == 0


# ---------------------------------------------------------------------------------------------------------------
# not-run publication, projection
# ---------------------------------------------------------------------------------------------------------------

def test_publish_not_run(tmp_path):
    with pytest.raises(SystemExit):
        r.publish(tmp_path)
    url = "https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-0"
    eb.write_json(tmp_path / "not-run.json", {"typed_reason": "cap", "decision": url})
    assert r.project(tmp_path) == 0
    assert r.publish(tmp_path) == 0
    summary = eb.load_child_summary(tmp_path / "summary.json")
    assert summary["arms"] == {} and summary["extra"]["pending_arms"] == {"T7": "cap"}
    assert summary["extra"]["not_run_marker"]["decision"] == url
    assert summary["gfid"]["T7"]["pass"] is False and summary["extra"]["projection"]["primary"] == "projection_upper"
    assert "`cap`" in (tmp_path / "findings.md").read_text()


def test_projection_math():
    rates = {"Qwen2.5-VL-7B-Instruct": 3.0, "Qwen2.5-VL-3B-Instruct": 1.5}
    loads = {"Qwen2.5-VL-7B-Instruct": 100.0, "Qwen2.5-VL-3B-Instruct": 50.0}
    p = r.child_projection(rates, loads, "Qwen2.5-VL-7B-Instruct", 2.0)
    selection = 94 * 3.0 + 100 + 94 * 1.5 + 50
    after = (480 + 992 + 464 + 992) * 3.0 + 4 * 100
    assert p["selection_gpu_h"] == pytest.approx(selection / 3600)
    assert p["total_gpu_h"] == pytest.approx((selection + after) / 3600 + 2.0 + 1.0)
    assert p["fits_remainder"] and p["within_caps"]
    big = r.child_projection({m: 200.0 for m in rates}, loads, "Qwen2.5-VL-7B-Instruct", 2.0)
    assert not big["fits_remainder"]


def test_forward_flops_of_the_7b():
    config = json.loads((dn.model_dir("Qwen2.5-VL-7B-Instruct") / "config.json").read_text())
    f = r.forward_flops(config)
    d, inter, layers, vocab, seq = 3584, 18944, 28, 152064, 1216
    per_layer = 2 * d * d + 2 * d * 4 * 128 + 3 * d * inter
    assert f["language"] == 2 * seq * (layers * per_layer + d * vocab)
    assert 1.5e13 < f["total"] < 2.5e13
    assert r.decision_flops("Qwen2.5-VL-7B-Instruct")["decision_flops"] == 16 * f["total"]


# ---------------------------------------------------------------------------------------------------------------
# the worker end to end (env b127-deepphy, tiny random-init Qwen2.5-VL on the CPU)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not dn.WORKER_PYTHON.is_file(), reason="env b127-deepphy absent")
def test_worker_scores_fit_frames_with_a_tiny_model(tmp_path):
    tiny = tmp_path / "tiny"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
    subprocess.run([str(dn.WORKER_PYTHON), "-c", "import sys; sys.path.insert(0, 'scripts'); "
                    "import b127_deepphy_port_reference as ref; ref.tiny_model(sys.argv[1])", str(tiny)],
                   cwd=ROOT, env=env, check=True)
    frames = eb.fit_anchor_frames()
    members = sorted(frames)[:2]
    items, actions = r.grid_actions()
    response = r.score_items(tmp_path, "tiny", "Qwen2.5-VL-7B-Instruct", ["q7b", "q7b-t01"],
                             [(m, frames[m], actions) for m in members], device="cpu", model_path=tiny,
                             dtype="float32", batch=4)
    assert response["transformers"] == "4.52.4" and response["device"] == "cpu"
    for m in members:
        for name in ("q7b", "q7b-t01"):
            s = response["scores"][m][name]
            assert len(s) == 16 and all(math.isfinite(v) and v < 0 for v in s) and s[8] == s[12]
        assert response["answers"][m] == GRID_ANSWERS and response["prompt_tokens"][m] == r.PROMPT_TOKENS
    again = r.score_items(tmp_path, "tiny", "Qwen2.5-VL-7B-Instruct", ["q7b"], [], device="cpu")
    assert again["scores"] == response["scores"]  # a finished response is reused
