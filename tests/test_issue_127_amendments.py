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
    """#130 A1: the plot-decode chunk change replaces exactly the bytes frozen in the #130 plan."""
    path = "scripts/b127_dino_wm_fidelity.py"
    frozen = _frozen(".local-artifacts/issue-130-dino-wm-v1/plan.json")[path]
    current = "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()
    entry = pfa.AMENDMENTS[path]
    assert entry["issue"] == 130 and entry["replaces"] == (frozen,) and entry["sha256"] == current
    assert pfa.matches(path, frozen, current)
    from scripts import run_issue_130_dino_wm as r130
    assert r130.load_plan(r130.OUTPUT)["plan_identity"] == r130.IDENTITY
