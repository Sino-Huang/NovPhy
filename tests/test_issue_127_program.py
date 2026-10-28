"""Issue #127 program summary runner (scripts/run_issue_127_program.py): synthetic child summaries (eb.child_summary),
a synthetic scheduler queue and sample decision comments; no evaluation data. CPU only."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts import external_baselines as eb
from scripts import run_issue_127_program as rp

ARMS = {128: {"A1": "primary", "A2": "primary", "A4": "primary", "A3": "secondary"}, 129: {"B1": "primary"},
        130: {"C1-dino": "primary", "C1-car": "secondary"}, 131: {"D1": "primary"}, 133: {"T1": "secondary"},
        134: {"T3": "secondary"}, 135: {}, 136: {"T4": "secondary"}, 137: {"T5": "secondary"},
        138: {"T6": "secondary"}, 139: {"T7": "secondary"}, 140: {"U-native": "control"}}
# a Tier-C child dropped before training publishes no arm block, only its pending arm (as #135 did)
EXTRAS = {135: {"pending_arms": {"T2": "fidelity_not_reproduced"}, "tier_c_arm": True}}


def est(mean, lo, hi):
    return {"mean": mean, "interval": [lo, hi], "clusters": 311, "units": 933, "label": "DESCRIPTIVE"}


ESTIMATES = {1: est(0.05, 0.03, 0.07), 2: est(-0.05, -0.07, -0.03), 3: est(0.0, -0.01, 0.01), 4: est(0.01, -0.02, 0.04)}


def contrast(view, second, estimate, guarded):
    token = eb.c2.token(estimate) if guarded else eb.STOP_TOKEN
    reverse = eb.c2.token(eb.rc.reverse(estimate)) if guarded else eb.STOP_TOKEN
    return {"first": view, "second": second, "cells": 933, "members": 311, "estimate": estimate, "token": token,
            "reverse_token": reverse, "guards_pass": guarded}


def block(view, role="primary", disp=2, guards=True, dropped=None, xc=None, xjdec=None, fires=False, extra_guards=None):
    guarded = bool(guards)
    contrasts = {"X-U": contrast(view, "U", ESTIMATES[disp], guarded),
                 "X-C*": contrast(view, "C*", xc or est(-0.05, -0.07, -0.03), guarded),
                 "X-P": contrast(view, "P", est(0.0, -0.02, 0.02), guarded),
                 "X-J_dec": contrast(view, "J_dec", xjdec or est(-0.03, -0.05, -0.01), guarded)}
    return {"view": view, "role": role, "program_set": view in eb.PROGRAM_SET, "primary_cost_key": "tie_free",
            "guards": {"G_rank": guarded, **(extra_guards or {}), "pass": guarded}, "global_guards_pass": True,
            "guards_pass": guarded, "typed_failures": {}, "nonfinite_share": 0.0,
            "ranker": {"auc": est(0.7, 0.68, 0.72)}, "contrasts": contrasts,
            "disposition": eb.disposition(contrasts["X-U"], guarded), "budget_safeguard": {"fires": fires},
            "compute": {"trainable_parameters": 1000, "training_gpu_seconds": {str(s): 3600.0 for s in eb.SEEDS},
                        "macs_per_candidate": 1e6, "wall_per_decision": 0.01}, "dropped": dropped}


def write_child(artifacts, issue, overrides=None, extra=None, gfid=None):
    overrides = overrides or {}
    blocks = {view: overrides.get(view) or block(view, role) for view, role in ARMS[issue].items()}
    summary = eb.child_summary(issue, f"issue-{issue}-test-v1", "2026-10-25T00:00:00Z", blocks,
                               {"within": {"label": "x - y", **contrast("x", "y", est(0.01, -0.01, 0.03), True)}},
                               {"G_records": {"pass": True}}, gfid or {v: {"pass": True} for v in blocks}, extra or {})
    base = rp.child_dir(artifacts, issue)
    eb.write_json(base / "summary.json", summary)
    eb.write_json(base / "records-manifest.json", {"records": []})


def setup(tmp_path, overrides=None, extras=None, skip=()):
    artifacts, output = tmp_path / "artifacts", tmp_path / "out"
    overrides, extras = overrides or {}, extras or {}
    for issue in ARMS:
        if issue not in skip:
            write_child(artifacts, issue, overrides.get(issue), extras.get(issue, EXTRAS.get(issue)))
    eb.write_json(rp.child_dir(artifacts, 132) / "summary.json",
                  {"agents": {"datalab": {"pass1": 0.25, "pass3": None}}, "engine_worker_hours": 50.0})
    queue = tmp_path / "queue.json"
    eb.write_json(queue, {"jobs": []})
    eb.write_json(output / "budget.json", rp.budget(queue, artifacts, datetime(2026, 10, 25, tzinfo=timezone.utc)))
    eb.write_json(output / "decision-log.json", {"fetched_at": "2026-10-25T00:00:00Z", "issues": [127, *rp.CHILDREN],
                                                 "decisions": []})
    sources = {"issue_123_compute": tmp_path / "c123.json", "issue_102_summary": tmp_path / "s102.json"}
    eb.write_json(sources["issue_123_compute"], {"inventories": {"grid": {"rankers": {
        n: {"auc": est(0.74, 0.72, 0.76), "top1_rate": 0.2} for n in rp.REFERENCE_RANKERS}}}})
    eb.write_json(sources["issue_102_summary"], {"sets": {"N1": {"random": {"pass1": est(0.07, 0.05, 0.08),
                                                                          "pass3": est(0.198, 0.17, 0.22)}}}})
    return artifacts, output, sources


def summary_of(tmp_path, **kwargs):
    allow = kwargs.pop("allow_partial", False)
    artifacts, output, sources = setup(tmp_path, **kwargs)
    return rp.build(output, artifacts, allow, sources)


def test_claim_holds_and_c_star_version(tmp_path):
    summary, findings = summary_of(tmp_path)
    assert summary["program_claim"]["U"]["holds"] and summary["program_claim"]["C*"]["holds"]
    assert {r["view"] for r in summary["program_table"]} == {v for arms in ARMS.values() for v in arms}
    assert not any(t["fires"] for t in summary["tier_c"].values() if t["fires"] is not None)
    assert "**Licensed:**" in findings
    assert not any(word.lower() in findings.lower() for word in rp.BANNED_WORDS)


def test_veto_dropped_member(tmp_path):
    dropped = block("C1-dino", guards=False, disp=4, dropped="readiness_ceiling_below_threshold")
    summary, findings = summary_of(tmp_path, overrides={130: {"C1-dino": dropped}})
    claim = summary["program_claim"]["U"]
    assert not claim["holds"] and {"arm": "C1-dino", "reason": "dropped: readiness_ceiling_below_threshold"} in claim["vetoes"]
    assert {"child": 130, "arm": "C1-dino", "status": "dropped", "reason": "readiness_ceiling_below_threshold"} in summary["not_run"]
    assert "**Not licensed.**" in findings


def test_veto_guard_failure(tmp_path):
    summary, _ = summary_of(tmp_path, overrides={131: {"D1": block("D1", guards=False, disp=4)}})
    assert {"arm": "D1", "reason": "guard_failed"} in summary["program_claim"]["U"]["vetoes"]
    assert summary["tier_c"]["Diffusion Forcing / gamma-models"]["fires"] is False  # stop tokens never fire it


def test_disposition_one_triggers(tmp_path):
    summary, findings = summary_of(tmp_path, overrides={129: {"B1": block("B1", disp=1)}})
    t = summary["tier_c"]
    assert t["DreamerV3 / DIAMOND / V-JEPA 2-AC"] == {**t["DreamerV3 / DIAMOND / V-JEPA 2-AC"], "fires": True, "arms": ["B1"]}
    assert t["P-follow-up"]["fires"] and t["P-follow-up"]["arms"] == ["B1"]
    assert summary["program_claim"]["U"]["disposition_1_arms"] == ["B1"] and not summary["program_claim"]["U"]["holds"]
    assert "B1 ranks launches better than U on the sealed NovPhy grid (scoped)" in findings
    a1 = summary_of(tmp_path / "a1", overrides={128: {"A1": block("A1", disp=1)}})[0]["tier_c"]
    assert a1["P-follow-up"]["arms"] == ["A1"] and not a1["DreamerV3 / DIAMOND / V-JEPA 2-AC"]["fires"]


def test_budget_safeguard_u_native(tmp_path):
    summary, _ = summary_of(tmp_path, overrides={128: {"A4": block("A4", disp=1, fires=True)}})
    assert summary["tier_c"]["U-native"]["fires"] and summary["tier_c"]["U-native"]["arms"] == ["A4"]


def test_c_star_version_derived_from_x_minus_c_star(tmp_path):
    summary, _ = summary_of(tmp_path, overrides={128: {"A1": block("A1", xc=est(0.04, 0.01, 0.07))}})
    assert summary["program_claim"]["U"]["holds"]
    c = summary["program_claim"]["C*"]
    assert not c["holds"] and c["vetoes"] == [{"arm": "A1", "reason": "disposition_1"}] and c["disposition_1_arms"] == ["A1"]


def test_d1_and_ddlp_triggers(tmp_path):
    summary, _ = summary_of(tmp_path, overrides={131: {"D1": block("D1", xjdec=est(0.04, 0.01, 0.07))}},
                            extras={129: {"positions": {"B1": {"pred_225": est(0.06, 0.05, 0.07)},
                                                        "U": {"pred_225": est(0.05, 0.04, 0.06)}}}})
    t = summary["tier_c"]
    assert t["Diffusion Forcing / gamma-models"]["fires"] is True
    assert t["DDLP"]["fires"] is True and t["DDLP"]["B1_fails_G_readout"] is None
    summary, _ = summary_of(tmp_path / "b", overrides={129: {"B1": block("B1", extra_guards={"G_readout": True})}})
    assert summary["tier_c"]["DDLP"]["fires"] is False and summary["tier_c"]["DDLP"]["B1_position_exceeds_U"] is None


def test_partial_and_native_action(tmp_path):
    with pytest.raises(ValueError):
        summary_of(tmp_path / "strict", skip=(131,))
    summary, findings = summary_of(tmp_path, skip=(131,), allow_partial=True)
    assert summary["partial"] and summary["children"]["131"]["status"] == "pending"
    assert {"child": 131, "arm": None, "status": "pending", "reason": "child summary not published"} in summary["not_run"]
    assert {"arm": "D1", "reason": "not_run"} in summary["program_claim"]["U"]["vetoes"]
    assert summary["native_action"]["agents"] == {"datalab": {"pass1": 0.25, "pass3": None}}
    assert summary["native_action"]["issue_102_n1"]["random"]["pass3"]["mean"] == 0.198
    assert "| E1 datalab | 0.2500 | n/a |" in findings and "**Partial summary**" in findings
    assert summary["compute"]["engine_worker_hours"]["132"] == 50.0
    # a Tier-C child dropped before training is listed as not run with its typed reason; coverage rows stay secondary
    assert {"child": 135, "arm": "T2", "status": "not_run", "reason": "fidelity_not_reproduced"} in summary["not_run"]
    assert all(a["arm"] not in ("T1", "T3", "T4") for a in summary["program_claim"]["U"]["vetoes"])


def test_summary_validate_roundtrip(tmp_path):
    artifacts, output, sources = setup(tmp_path)
    assert rp.run_summary(output, artifacts, False, sources) == 0
    assert rp.validate(output, artifacts, sources) == 0
    assert eb.read_json(output / "validation.json")["pass"]
    write_child(artifacts, 129, {"B1": block("B1", disp=4)})
    assert rp.validate(output, artifacts, sources) == 1
    receipt = eb.read_json(output / "validation.json")
    assert not receipt["pass"] and any(a.endswith("issue-129-slotformer-v1/summary.json") for a in receipt["input_drift"])


def iso(h, m=0):
    return f"2026-10-25T{h:02d}:{m:02d}:00+00:00"


def test_budget_interval_union(tmp_path):
    queue = tmp_path / "queue.json"
    eb.write_json(queue, {"jobs": [
        {"id": "128-a", "gpu": 0, "runs": [{"tag": "128-a#1", "launched_at": iso(0), "ended_at": iso(1)}]},
        {"id": "129-b", "gpu": 0, "runs": [{"tag": "129-b#1", "launched_at": iso(0, 30), "ended_at": iso(2)},
                                           {"tag": "129-b#2", "launched_at": None}]},
        {"id": "130-c", "gpu": 1, "runs": [{"tag": "130-c#1", "launched_at": iso(1)}]},
        {"id": "131-d", "gpu": None, "runs": [{"tag": "131-d#1", "launched_at": iso(0), "ended_at": iso(5)}]},
        {"id": "123-x", "gpu": 0, "runs": [{"tag": "123-x#1", "launched_at": iso(0), "ended_at": iso(9)}]},
        {"id": "128-e", "gpu": 2, "runs": [{"tag": "128-e#1", "launched_at": iso(2), "window_killed_at": iso(2, 30)}]},
    ]})
    b = rp.budget(queue, tmp_path, datetime(2026, 10, 25, 3, tzinfo=timezone.utc))
    p = b["program"]
    assert p["per_gpu_device_seconds"] == {"0": 2 * 3600.0, "1": 2 * 3600.0, "2": 1800.0}
    assert p["device_gpu_hours"] == pytest.approx(4.5) and p["process_gpu_hours"] == pytest.approx(5.0)
    assert p["running"] == ["130-c#1"] and p["cpu_runs"] == 1
    assert b["children"]["128"]["device_gpu_hours"] == pytest.approx(1.5)
    assert b["children"]["129"]["device_gpu_hours"] == pytest.approx(1.5)
    assert b["sum_child_device_gpu_hours"] == pytest.approx(5.0) and b["sum_child_device_gpu_hours"] > p["device_gpu_hours"]
    assert b["remaining_gpu_hours"] == pytest.approx(395.5)
    assert b["children"]["128"]["projected_remaining_gpu_h"] is None
    assert "|" in rp.budget_lines(b)[0]


def test_budget_moved_job_keeps_each_runs_gpu(tmp_path):
    """A job moved to another GPU between runs: its earlier run stays on the GPU it ran on (run-level "gpu")."""
    queue = tmp_path / "queue.json"
    eb.write_json(queue, {"jobs": [
        {"id": "137-a", "gpu": 2, "runs": [{"tag": "137-a#1", "gpu": 0, "launched_at": iso(0), "ended_at": iso(2)},
                                           {"tag": "137-a#2", "launched_at": iso(2), "ended_at": iso(3)}]},
        {"id": "128-b", "gpu": 0, "runs": [{"tag": "128-b#1", "launched_at": iso(1), "ended_at": iso(2)}]},
    ]})
    p = rp.budget(queue, tmp_path, datetime(2026, 10, 25, 3, tzinfo=timezone.utc))["program"]
    assert p["per_gpu_device_seconds"] == {"0": 2 * 3600.0, "2": 3600.0}


def test_projection_mapping(tmp_path):
    eb.write_json(rp.child_dir(tmp_path, 128) / "smoke.json", {"projection": {"remaining_gpu_hours": 23.5}})
    eb.write_json(rp.child_dir(tmp_path, 129) / "smoke.json", {"projection": {"total_gpu_h": 33.0, "selection_gpu_h": 8.0}})
    assert rp.projected_remaining(tmp_path, 128)[0] == 23.5 and rp.projected_remaining(tmp_path, 129)[0] == 25.0
    assert rp.projected_remaining(tmp_path, 130)[0] is None


@pytest.mark.parametrize("body,expected", [
    ("## Decision P-budget (2026-10-25): how the 400 GPU-h program budget is counted\n\nbody",
     {"rule": "P-budget", "date": "2026-10-25", "title": "how the 400 GPU-h program budget is counted", "addendum": False}),
    ("## Decision P-crash (2026-10-25): addendum to issuecomment-6386786676 — cause identified",
     {"rule": "P-crash", "addendum": True}),
    ("Addendum to Decision P-other 2026-10-25 (https://github.com/x/issues/128#issuecomment-1): NN-px rows",
     {"rule": "P-other", "date": "2026-10-25", "title": "addendum: NN-px rows", "addendum": True,
      "refers_to": "https://github.com/x/issues/128#issuecomment-1"}),
    ("## Decision P-fid (2026-10-24): band [82.8, 87.2]", {"rule": "P-fid", "title": "band [82.8, 87.2]"}),
    ("Decision taken offline", None), ("## Decision Pending (2026-10-25): x", None), ("", None),
])
def test_parse_heading(body, expected):
    got = rp.parse_heading(body)
    if expected is None:
        assert got is None
    else:
        assert {k: got[k] for k in expected} == expected


def test_decision_log_order_and_markdown():
    comments = {128: [{"html_url": "u2", "created_at": "2026-10-25T01:00:00Z", "body": "## Decision P-cap (2026-10-25): b [x]"},
                      {"html_url": "u3", "created_at": "2026-10-25T00:00:00Z", "body": "plain comment"}],
                127: [{"html_url": "u1", "created_at": "2026-10-25T01:00:00Z", "body": "## Decision P-budget (2026-10-25): a"}]}
    log_ = rp.decision_log(comments)
    assert [d["url"] for d in log_] == ["u1", "u2"]
    md = rp.decision_md({"decisions": log_, "issues": [127, 128], "fetched_at": "t"})
    assert "[b \\[x\\]](u2)" in md and "## #127 (program): 1" in md
