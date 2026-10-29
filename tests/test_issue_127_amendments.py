"""#127 program: post-freeze amendments of child-pinned files are registered against exactly the frozen bytes, and
the child's load_plan accepts the amended file (CPU)."""
from __future__ import annotations

import hashlib
import json

from scripts import post_freeze_amendments as pfa


def _frozen(plan_path):
    """artifact -> sha256 of every pin in a frozen child plan (pins may sit at any depth)."""
    out = {}

    def walk(node):
        if isinstance(node, dict):
            if "artifact" in node and "sha256" in node:
                out[node["artifact"]] = node["sha256"]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.load(open(plan_path)))
    return out


def test_issue_130_fidelity_runner_amendment():
    """#130 A1 + A2: the plot-path changes replace exactly the bytes frozen in the #130 plan (and the A1 bytes)."""
    path = "scripts/b127_dino_wm_fidelity.py"
    frozen = _frozen(".local-artifacts/issue-130-dino-wm-v1/plan.json")[path]
    a1 = "sha256:e5c861457459336bce37ec747adbf42176a350eb3176f004c3def4f5be10d3b9"
    current = "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()
    entry = pfa.AMENDMENTS[path]
    assert entry["issue"] == 130 and entry["replaces"] == (frozen, a1) and entry["sha256"] == current
    assert pfa.matches(path, frozen, current) and not pfa.matches(path, frozen, a1)
    from scripts import run_issue_130_dino_wm as r130
    assert r130.load_plan(r130.OUTPUT)["plan_identity"] == r130.IDENTITY


def test_issue_128_runner_amendment():
    """#128 A1 (drop marker), A2 (A3-sym repair re-selection), A3 (engine seconds): the runner and its tests replace
    exactly the bytes frozen in the #128 plan and the A1 / A2 bytes, and the #128 load_plan accepts them."""
    frozen = _frozen(".local-artifacts/issue-128-direct-predictors-v1/plan.json")
    a1 = {"scripts/run_issue_128_direct_predictors.py":
          "sha256:5b99b0c78a997bfd1a29587f522d86bda148cc1e99320970bce6a484a6fb9ecc",
          "tests/test_issue_128_direct_predictors.py":
          "sha256:396daef3782fa4ace30f0ba73b4762823b85a670da79ed31c7570c5e4ca2bf88"}
    a2 = {"scripts/run_issue_128_direct_predictors.py":
          "sha256:3d6cecc760cf49732bf841ae1eac94e88d92d91c1a43dbe80333bd70275692a8",
          "tests/test_issue_128_direct_predictors.py":
          "sha256:7efb78198bce1d44a8bb3e2fc53a9ab0dc2a63b7c3c2a20276c227857a0fda64"}
    for path, a1_sha in a1.items():
        current = "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()
        entry = pfa.AMENDMENTS[path]
        assert entry["issue"] == 128 and entry["replaces"] == (frozen[path], a1_sha, a2[path])
        assert entry["sha256"] == current and pfa.matches(path, frozen[path], current)
        assert not pfa.matches(path, frozen[path], a1_sha) and not pfa.matches(path, frozen[path], a2[path])
    from scripts import run_issue_128_direct_predictors as r128
    assert r128.load_plan(r128.OUTPUT)["identity"] == r128.IDENTITY


def test_issue_128_phyq_repair1_amendment():
    """#128 A3-sym repair 1 (P-fid): the adapter, both symbolic stores, the A3 G-port record and the A3 tests replace
    exactly the bytes frozen in the #128 plan, and the #128 load_plan accepts them."""
    out = ".local-artifacts/issue-128-direct-predictors-v1"
    frozen = _frozen(f"{out}/plan.json")
    for path in ("world_model/baselines/phyq/novphy.py", f"{out}/a3/data/symbolic-fit.pt",
                 f"{out}/a3/data/symbolic-policy.pt", f"{out}/gport/a3/gport.json", "tests/test_issue_128_phyq.py"):
        current = "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()
        entry = pfa.AMENDMENTS[path]
        assert entry["issue"] == 128 and entry["replaces"] == (frozen[path],) and entry["sha256"] == current
        assert pfa.matches(path, frozen[path], current)
    from scripts import run_issue_128_direct_predictors as r128
    assert r128.load_plan(r128.OUTPUT)["identity"] == r128.IDENTITY


def test_issue_137_runner_amendment():
    """#137 A1 (reporting): the runner and its tests replace exactly the bytes frozen in the #137 plan, the #137 load_plan accepts
    it, and the published parameter count is the trained model's."""
    frozen = _frozen(".local-artifacts/issue-137-diamond-v1/plan.json")
    for path in ("scripts/run_issue_137_diamond.py", "tests/test_issue_137_diamond.py"):
        current = "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()
        entry = pfa.AMENDMENTS[path]
        assert entry["issue"] == 137 and entry["replaces"] == (frozen[path],) and entry["sha256"] == current
        assert pfa.matches(path, frozen[path], current)
    from scripts import run_issue_137_diamond as r137
    plan = r137.load_plan(r137.OUTPUT)
    assert plan["plan_identity"] == r137.IDENTITY
    assert r137.arm_compute(r137.OUTPUT, plan, "T5")["trainable_parameters"] == 4_406_019
