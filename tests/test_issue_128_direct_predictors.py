"""#128 runner: CPU checks of the uncertain code (arm registry, resumable training, typed failure, readouts)."""
import json
import math

import numpy as np
import pytest
import torch

from scripts import external_baselines as eb
from scripts import run_issue_128_direct_predictors as r128

torch.set_num_threads(4)  # the shared host is oversubscribed; small CPU forwards are far faster at 1-4 threads


def test_arm_registry():
    for arm, spec in r128.ARMS.items():
        assert len(spec["configurations"]) <= 3
        if spec["program_set"]:
            assert arm in eb.PROGRAM_SET and spec["role"] == "primary"
        if spec["family"] == "phyq":
            assert not spec["program_set"] and spec["deferred_scoring"] and spec["role"] == "secondary"
    # the native / official configuration is declared first (ties in eb.select_configuration fall to it last)
    assert next(iter(r128.ARMS["A1"]["configurations"])) == "native"
    assert r128.ARMS["A1"]["configurations"]["native"] == {"updates": 100_000, "learning_rate": 3e-4}
    assert {c["updates"] for c in r128.ARMS["A2"]["configurations"].values()} == {9_000}
    assert r128.ARMS["A4"]["batch"] == 512


@pytest.fixture(scope="module")
def context():
    return r128.Context("cpu")


def test_balanced_batches_from_batch_rng(context):
    data = context.trainset("A2")
    assert len(data.is_solved) == 13_919 and data.observations.shape == (480, 236)
    family = r128.family_of("A2")
    first = family.batch(data, r128.ARMS["A2"], eb.batch_rng(20260908, 0))
    again = family.batch(data, r128.ARMS["A2"], eb.batch_rng(20260908, 0))
    assert np.array_equal(first, again) and int(data.is_solved[first].sum()) == 32
    assert not np.array_equal(first, family.batch(data, r128.ARMS["A2"], eb.batch_rng(20260908, 1)))


def test_resume_reproduces_uninterrupted_run(tmp_path, context, monkeypatch):
    config = {"updates": 6, "learning_rate": 3e-4}
    full = r128.train_run("A2", config, 20260908, tmp_path / "full", context)
    assert full["complete"] and not (tmp_path / "full" / "resume.pt").exists()
    part = r128.train_run("A2", config, 20260908, tmp_path / "split", context, max_steps=3)
    assert part["partial"] and (tmp_path / "split" / "resume.pt").is_file()
    done = r128.train_run("A2", config, 20260908, tmp_path / "split", context)
    assert done["complete"]
    a = torch.load(tmp_path / "full" / "model.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "split" / "model.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    # a stop signal writes resume.pt after the current update and raises TrainingPaused
    monkeypatch.setattr(eb, "STOP_REQUESTED", [15])
    with pytest.raises(eb.TrainingPaused):
        r128.train_run("A2", config, 20260908, tmp_path / "stopped", context)
    assert torch.load(tmp_path / "stopped" / "resume.pt", weights_only=False)["step"] == 1


def test_nonfinite_loss_is_a_typed_failure(tmp_path, context, monkeypatch):
    family = r128.family_of("A2")
    monkeypatch.setattr(family, "step", lambda *a, **k: {"loss": torch.tensor(float("nan"))})
    report = r128.train_run("A2", {"updates": 4, "learning_rate": 3e-4}, 20260908, tmp_path, context)
    payload = torch.load(tmp_path / "model.pt", weights_only=False)
    assert not report["complete"] and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    assert payload["state_dict"] is None and payload["typed_failure"]["step"] == 0


def test_readouts_and_selection(tmp_path, context, monkeypatch):
    monkeypatch.setattr(r128, "selection_dir", lambda output, arm, config: tmp_path / arm / config)
    for config in r128.ARMS["A2"]["configurations"]:
        r128.train_run("A2", {"updates": 2, "learning_rate": 3e-4}, 20260908, tmp_path / "A2" / config, context)
    assert r128.selection_readouts(tmp_path, 0, 1, ["A2"], context) == 0
    readout = eb.read_json(tmp_path / "A2" / "lr3e-4" / "readouts.json")
    assert readout["G_rank"]["levels"] == 480 and 0.0 <= readout["policy_grid_auc"] <= 1.0
    assert readout["macs_per_candidate"] > 0 and readout["parameters"] == 1_838_781
    chosen = r128.selection_result(tmp_path, ["A2"])["A2"]
    assert chosen["complete"] and chosen["selected"] in r128.ARMS["A2"]["configurations"]


def test_score_rows_sign_and_launch_matrix():
    items = eb.grid_items()[:3]
    rows = eb.score_rows(items, torch.tensor([2.0, float("nan"), -1.0]))
    assert rows[0][r128.SCORE_KEY] == -2.0 and rows[1]["excluded"] and rows[2][r128.SCORE_KEY] == 1.0
    assert r128.launch_matrix(items).shape == (3, 4)


def test_a1_partial_update_on_cpu(tmp_path, context):
    out = r128.train_run("A1", {"updates": 10, "learning_rate": 3e-4}, 20260908, tmp_path, context, max_steps=1)
    assert out["partial"] and math.isfinite(out["seconds"])
    assert torch.load(tmp_path / "resume.pt", weights_only=False)["step"] == 1


def test_a4_family_wiring_on_cpu(tmp_path, context, monkeypatch):
    """A4 through the runner (runner_trainset / runner_batch / runner_train_step / score_actions), 2 levels x 8."""
    family = r128.family_of("A4")
    monkeypatch.setattr(family, "batch", lambda data, spec, rng: family.da.runner_batch(data, rng, n_tasks_per_batch=2))
    out = r128.train_run("A4", {"updates": 3, "learning_rate": 3e-4}, 20260908, tmp_path, context, max_steps=1)
    assert out["partial"] and torch.load(tmp_path / "resume.pt", weights_only=False)["step"] == 1
    anchors = context.anchors("fit", "frame")
    model = family.build(r128.ARMS["A4"])
    scores = r128.scores_for("A4", model, anchors[sorted(anchors)[0]], eb.grid_items(), "cpu")
    assert scores.shape == (16,) and torch.isfinite(scores).all()


@pytest.mark.parametrize("arm", ["A3-sym", "A3-img"])
def test_a3_family_wiring_on_cpu(arm, tmp_path, context):
    """A3 through the runner: symbolic / frame inputs, AdamW + outer-step resets in resume.pt, Q scores."""
    assert r128.available(arm) and arm in r128.ready_arms()
    configs = r128.ARMS[arm]["configurations"]
    assert 1 <= len(configs) <= 3 and all(c["resets"] and c["learning_rate"] == 3e-4 for c in configs.values())
    config = {"updates": 3, "learning_rate": 3e-4, "resets": [1]}
    out = r128.train_run(arm, config, 20260908, tmp_path, context, max_steps=2)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert out["partial"] and state["step"] == 2 and state["scheduler"]["update"] == 2
    anchors = context.anchors("policy", r128.ARMS[arm]["input"])
    model = r128.family_of(arm).build(r128.ARMS[arm])
    scores = r128.scores_for(arm, model, anchors[sorted(anchors)[0]], eb.grid_items(), "cpu")
    assert scores.shape == (16,) and torch.isfinite(scores).all()


def test_lpt_queues():
    queues, loads = r128.lpt_queues([("a/1", 3.0), ("b/1", 2.0), ("c/1", 2.0), ("d/1", 1.0)], gpus=2)
    assert sorted(loads) == [4.0, 4.0] and sum(len(q) for q in queues) == 4


def test_freeze_score_and_spot_check_on_policy_stand_in(context, monkeypatch):
    """Freeze groups, scoring records and the validation re-score, with the POLICY split standing in for the
    evaluation split (the evaluation split is never read by tests)."""
    import shutil
    import uuid

    output = eb.ROOT / ".local-artifacts" / f"issue-128-test-{uuid.uuid4().hex[:8]}"
    try:
        config = {"updates": 2, "learning_rate": 3e-4}
        plan = {"frozen_at": "2000-01-01T00:00:00+00:00", "scoring_groups": {"test": ["A2"]},
                "arms": {"A2": {"status": "ready", "selected_configuration": "lr3e-4", "config": config,
                                "parameters": 1_838_781}}}
        monkeypatch.setattr(r128, "load_plan", lambda out: plan)
        for seed in eb.SEEDS:
            run = r128.model_dir(output, plan, "A2", seed)
            r128.train_run("A2", config, seed, run, context)
            r128.readout_run("A2", run, context)
        assert r128.model_dir(output, plan, "A2", eb.SEEDS[0]) == output / "selection" / "A2" / "lr3e-4" / "seed-20260908"
        assert r128.freeze_readouts(output) == 0  # the default is the next whole scoring group
        with pytest.raises(ValueError):
            r128.freeze_readouts(output, ["A2"])
        cells = {inv: {"mixed": eb.policy_cells(inv)} for inv in eb.INVENTORIES}
        monkeypatch.setattr(eb, "evaluation_cells", lambda: cells)

        def stand_in(out, policy):
            payload = {"carriers": eb.policy_anchors()["carriers"], "frames": eb.policy_anchor_frames(),
                       "read_at": eb.utc_now(), "scoring_policy": policy}
            eb.atomic_torch(eb.evaluation_anchors_path(out), payload)
            return payload

        monkeypatch.setattr(eb, "read_evaluation_anchors", stand_in)
        assert r128.score(output) == 0
        record = eb.read_json(r128.record_path(output, "A2", eb.SEEDS[1]))
        assert set(record["inventories"]) == set(eb.INVENTORIES) and record["wall_per_decision"] > 0
        assert len(record["inventories"]["grid"]) == len(cells["grid"]["mixed"])
        assert eb.nonfinite_share([record], r128.SCORE_KEY) == 0.0
        nn = eb.read_json(r128.nn_reference_path(output, "NN-px"))  # written once in the first scoring pass
        assert set(nn["inventories"]["grid"]) == set(cells["grid"]["mixed"]) and set(r128.nn_views(output)) == {
            "NN-px", "NN-car"}
        spot = r128.spot_check(output, plan)
        assert spot["pass"] and spot["arm"] == "A2"
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_deferred_selection_and_training_queue(tmp_path, monkeypatch):
    """A4: no selection before the freeze or before its G-fid; after --select-deferred its main seeds join the
    planned GPU queues and seed 20260908 is the selected run."""
    assert not r128.deferred_gate(tmp_path, "A4")
    (tmp_path / "plan.json").write_text("{}")
    monkeypatch.setattr(r128, "gfid_pass", lambda output, arm: None)
    assert not r128.deferred_gate(tmp_path, "A4")
    monkeypatch.setattr(r128, "gfid_pass", lambda output, arm: True)
    assert r128.deferred_gate(tmp_path, "A4") and r128.deferred_gate(tmp_path, "A1")
    configs = r128.ARMS["A4"]["configurations"]
    plan = {"training": {"queues": [["A1/20260909"], [], [], []]},
            "arms": {"A4": {"status": "selection_after_gfid", "selected_configuration": None,
                            "configurations": configs, "deferred_training_gpus": {"20260909": 0, "20260910": 1}}}}
    assert r128.arm_selection(tmp_path, plan, "A4") == (None, None)
    assert r128.train_names(tmp_path, plan, 0) == ["A1/20260909"]
    eb.write_json(r128.deferred_selection_path(tmp_path), {"arms": {"A4": {"selected": "tenth"}}})
    assert r128.arm_selection(tmp_path, plan, "A4") == ("tenth", configs["tenth"])
    assert r128.train_names(tmp_path, plan, 0) == ["A1/20260909", "A4/20260909"]
    assert r128.train_names(tmp_path, plan, 1) == ["A4/20260910"]
    assert r128.model_dir(tmp_path, plan, "A4", 20260908) == r128.selection_dir(tmp_path, "A4", "tenth")


def test_projection_with_deferred_a4_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(r128, "selection_dir", lambda output, arm, config: tmp_path / arm / config)
    eb.write_json(tmp_path / "A1" / "tenth" / "report.json", {"seconds_per_update": 0.027})
    project = r128.projection(tmp_path, ["A1"], {"A1": {"selected": "tenth"}}, deferred=["A4"])
    assert project["per_seed"]["A4"]["source"] == "estimate"
    assert math.isclose(project["deferred_gpu_hours"]["A4"], 10.0 * 1.11 + 2 * 10.0)
    assert math.isclose(project["per_seed"]["A1"]["gpu_hours_per_seed"], 0.027 * 10_000 / 3600)


def test_score_novelty_on_policy_stand_in(context, monkeypatch):
    """--score-novelty refuses before the primary records exist; then, with the POLICY split standing in for both
    the evaluation and the novelty cells (no evaluation data is read), its records have the specified shape and the
    same rows as the primary scoring path on the same anchors."""
    import shutil
    import uuid

    output = eb.ROOT / ".local-artifacts" / f"issue-128-test-{uuid.uuid4().hex[:8]}"
    try:
        config = {"updates": 2, "learning_rate": 3e-4}
        plan = {"frozen_at": "2000-01-01T00:00:00+00:00", "scoring_groups": {"test": ["A2"]},
                "arms": {"A2": {"status": "ready", "selected_configuration": "lr3e-4", "config": config,
                                "parameters": 1_838_781}}}
        monkeypatch.setattr(r128, "load_plan", lambda out: plan)
        monkeypatch.setattr(r128, "gport_pass", lambda out, arm: True)
        monkeypatch.setattr(r128, "gfid_pass", lambda out, arm: True)
        for seed in eb.SEEDS:
            run = r128.model_dir(output, plan, "A2", seed)
            r128.train_run("A2", config, seed, run, context)
            r128.readout_run("A2", run, context)
        with pytest.raises(ValueError):  # no model freeze yet
            r128.score_novelty(output)
        assert r128.freeze_readouts(output) == 0
        with pytest.raises(ValueError):  # no primary records yet
            r128.score_novelty(output)
        policy_cells = eb.policy_cells("grid")
        carriers = eb.policy_anchors()["carriers"]
        monkeypatch.setattr(eb, "evaluation_cells", lambda: {inv: {"mixed": policy_cells} for inv in eb.INVENTORIES})

        def eval_stand_in(out, policy):
            payload = {"carriers": carriers, "frames": eb.policy_anchor_frames(), "read_at": eb.utc_now(),
                       "scoring_policy": policy}
            eb.atomic_torch(eb.evaluation_anchors_path(out), payload)
            return payload

        monkeypatch.setattr(eb, "read_evaluation_anchors", eval_stand_in)
        assert r128.score(output) == 0
        levels = {m: {"inventory": e["inventory"], "verdicts": e["verdicts"]} for m, e in policy_cells.items()}
        cells = {"levels": levels, "mixed": sorted(levels), "in_scope_cells": []}
        monkeypatch.setattr(eb, "novelty_cells", lambda: cells)

        def novelty_stand_in(out, policy, cells=None):
            payload = {"carriers": carriers, "frames": {}, "physics_paths": {}, "read_at": eb.utc_now(),
                       "scoring_policy": policy}
            eb.atomic_torch(eb.novelty_anchors_path(out), payload)
            return payload

        monkeypatch.setattr(eb, "read_novelty_anchors", novelty_stand_in)
        monkeypatch.setattr(eb, "novelty_reference_records", lambda *a, **k: {})
        assert r128.score_novelty(output) == 0
        for seed in eb.SEEDS:
            record = eb.read_json(r128.novelty_record_path(output, "A2", seed))
            primary = eb.read_json(r128.record_path(output, "A2", seed))
            assert record["schema"] == "issue_128_novelty_record_v1" and record["arm"] == "A2"
            assert record["seed"] == seed and record["model_sha256"] == primary["model_sha256"]
            assert set(record["inventories"]["grid"]) == set(levels) and record["scored_at"] >= primary["scored_at"]
            assert record["inventories"]["grid"] == primary["inventories"]["grid"]
        passes = eb.read_json(output / "novelty" / "passes.json")["passes"]
        assert passes[0]["eligible"] == ["A2"] and passes[0]["skipped"] == {}
        with pytest.raises(ValueError):  # a second default pass has no arm left
            r128.score_novelty(output)
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_frame_noise_counts_pixels_and_nn_rule():
    a = torch.zeros(3, 4, 4, dtype=torch.uint8)
    b = a.clone()
    b[:, 0, 0] = 1  # one pixel, three channels
    assert r128.frame_noise(a, b) == (1, 3 / 48)
    refs = {"f2": torch.full((3, 2, 2), 10, dtype=torch.uint8), "f1": torch.full((3, 2, 2), 10, dtype=torch.uint8),
            "f3": torch.full((3, 2, 2), 0, dtype=torch.uint8)}
    order = r128.pixel_orders({"q": torch.full((3, 2, 2), 9, dtype=torch.uint8)}, refs)["q"]
    assert [m for _, m in order] == ["f1", "f2", "f3"] and order[0][0] == 1.0  # tie -> lowest member id
    grid = eb.grid_items()[:2]
    k0, k1 = (r128.launch_key(i["action"]) for i in grid)
    verdicts = {("f1", k0): True, ("f2", k0): False, ("f2", k1): True}  # f1 never executed launch 1
    rows = r128.nn_rows(order, grid, verdicts)
    assert [row["score"] for row in rows] == [1.0, 1.0] and rows[0][r128.SCORE_KEY] == -1.0


def test_phyq_score_passes_cpu_launches(monkeypatch):
    """A3's adapter takes CPU observations (PIL / numpy image transform) and CPU launches (CPU action lookup table)
    and moves them itself; the runner hands it CPU tensors even when scoring on a GPU (128-selro-a3, 2026-10-25)."""
    family = r128.family_of("A3-img")
    seen = []
    monkeypatch.setattr(family.pq, "score_actions", lambda model, obs, launches: seen.append(
        (obs.device, launches.device)) or torch.zeros(len(launches)))
    family.score(None, torch.zeros(3, 240, 320, dtype=torch.uint8), r128.launch_matrix(eb.grid_items()))
    assert seen == [(torch.device("cpu"), torch.device("cpu"))]


# ---------------------------------------------------------------------------------------------------------------
# post-freeze amendment #128 A1: a scoring-group arm dropped with a typed reason (drop marker)
# ---------------------------------------------------------------------------------------------------------------

REAL_PLAN = eb.ROOT / ".local-artifacts/issue-128-direct-predictors-v1/plan.json"
FREEZE_COMMIT = "a3decdc6"  # the #128 freeze commit (plan.json frozen 2026-10-25T09:26:24Z)
DECISION = "https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6426445961"


def _fake_scores(arm, model, observation, items, device):
    """Cheap deterministic scores (no network forward): the amendment is about bookkeeping, not scoring."""
    return r128.launch_matrix(items).sum(1) * (1.0 + float(observation.float().mean()) % 1.0)


def _stand_ins(monkeypatch, output, modules, group_arm="A2"):
    """Policy split standing in for the evaluation split and the reference records (no evaluation data read)."""
    grid = eb.policy_cells("grid")
    cells = {inv: {"mixed": grid if inv in ("offset", "power") else eb.policy_cells(inv)} for inv in eb.INVENTORIES}
    monkeypatch.setattr(eb, "evaluation_cells", lambda: cells)
    clock = iter(f"2026-10-27T00:{m:02d}:{s:02d}+00:00" for m in range(60) for s in range(60))
    monkeypatch.setattr(eb, "utc_now", lambda: next(clock))
    monkeypatch.setattr(eb, "mark", lambda *a, **k: None)  # timing only (kept out of the byte comparison)
    monkeypatch.setattr(eb, "wall_per_decision", lambda fn, decisions: 0.001)

    def anchors(out, policy):
        payload = {"carriers": eb.policy_anchors()["carriers"], "frames": eb.policy_anchor_frames(),
                   "keys": dict(eb.policy_anchors()["keys"]), "read_at": "2026-10-27T00:00:30+00:00",
                   "scoring_policy": policy}
        eb.atomic_torch(eb.evaluation_anchors_path(out), payload)
        return payload

    monkeypatch.setattr(eb, "read_evaluation_anchors", anchors)
    fit = eb.read_json(eb.rt.verdicts_path(eb.RETRAIN))["branches"]
    prior = {inv: eb.gb.prior_stats(fit, inv) for inv in eb.INVENTORIES}

    def as_reference(record):  # the arm's rows under the references' tie-free cost key
        return {**record, "inventories": {inv: {m: {"rows": [{**row, eb.PRIMARY_COST: row[r128.SCORE_KEY]}
                                                             for row in cell["rows"]]}
                                                for m, cell in cells_.items()}
                                          for inv, cells_ in record["inventories"].items()}}

    def references():
        stand = {s: as_reference(eb.read_json(r128.record_path(output, group_arm, s))) for s in eb.SEEDS}
        return {"U": stand, "C*": stand, "J": stand, "J_dec": stand, "prior": prior}

    monkeypatch.setattr(eb, "references", references)
    for module in modules:
        monkeypatch.setattr(module, "scores_for", _fake_scores)
        monkeypatch.setattr(module, "score_nn_references", lambda *a: None)
        monkeypatch.setattr(module, "global_guards", lambda out, plan, records: {
            name: {"pass": True} for name in ("G_isolation", "G_chronology", "G_scoring_policy", "G_records")})


def _train_group(output, plan, arms, context):
    """model.pt / report.json from one CPU update per arm-seed; readouts.json binding each model (synthetic G-rank)."""
    for arm in arms:
        for seed in eb.SEEDS:
            run = r128.model_dir(output, plan, arm, seed)
            r128.train_run(arm, {"updates": 1, "learning_rate": 3e-4}, seed, run, context)
            eb.write_json(run / "readouts.json", {
                "schema": r128.SCHEMA_READOUTS, "arm": arm, "seed": seed, "typed_failure": None,
                "model_sha256": eb.sha256_of(run / "model.pt"), "parameters": 1, "policy_grid_auc": 0.5,
                "G_rank": eb.g_rank(np.random.default_rng(seed).normal(size=(480, 16))), "macs_per_candidate": 1.0})


def _synthetic_output():
    import uuid
    return eb.ROOT / ".local-artifacts" / f"issue-128-test-{uuid.uuid4().hex[:8]}"


def test_dropped_arm_freeze_score_and_publication(context, monkeypatch):
    """With the real frozen plan: the primary group (A1, A2, A2-native, A4) is refused while A4 has neither models
    nor a drop marker; with a valid marker it freezes (A4 recorded as dropped, no entries), --score writes no A4
    record, and the publication carries A4's block: dropped = typed reason, no estimates, guards failing (G_fid
    False), every token the stop token, disposition 4, a program-claim veto."""
    import shutil

    output = _synthetic_output()
    plan = eb.read_json(REAL_PLAN)
    assert plan["scoring_groups"]["primary"] == ["A1", "A2", "A2-native", "A4"]
    assert plan["arms"]["A4"]["status"] == "selection_after_gfid"
    monkeypatch.setattr(r128, "load_plan", lambda out: plan)
    group = ["A1", "A2", "A2-native", "A4"]
    try:
        _train_group(output, plan, ["A1", "A2", "A2-native"], context)
        _stand_ins(monkeypatch, output, [r128])
        with pytest.raises(ValueError):  # A4: no selection, no models, no marker
            r128.freeze_readouts(output, group)
        assert not r128.freeze_path(output).exists()
        marker = r128.drop_marker_path(output, "A4")
        eb.write_json(marker, {"schema": r128.SCHEMA_DROPPED, "arm": "A4", "typed_reason": "made_up",
                               "decision": DECISION, "written_at": "2026-10-28T00:00:00+00:00"})
        with pytest.raises(ValueError):  # an untyped reason is not a drop
            r128.freeze_readouts(output, group)
        eb.write_json(marker, {"schema": r128.SCHEMA_DROPPED, "arm": "A4", "typed_reason": "fidelity_not_reproduced",
                               "decision": DECISION, "written_at": "2026-10-28T00:00:00+00:00"})
        assert r128.freeze_readouts(output, group) == 0
        frozen = eb.read_json(r128.freeze_path(output))["groups"][0]
        assert frozen["arms"] == group and not any(k.startswith("A4/") for k in frozen["entries"])
        assert len(frozen["entries"]) == 9 and frozen["dropped"]["A4"]["typed_reason"] == "fidelity_not_reproduced"
        assert frozen["dropped"]["A4"]["marker_sha256"] == eb.sha256_of(marker)
        with pytest.raises(ValueError):  # never frozen twice
            r128.freeze_readouts(output, group)
        assert r128.score(output) == 0
        assert not any(r128.record_path(output, "A4", s).exists() for s in eb.SEEDS)
        assert all(r128.record_path(output, a, s).is_file() for a in group[:3] for s in eb.SEEDS)
        monkeypatch.setattr(r128, "novelty_section", lambda *a: None)
        compute = r128.compute_tables(output, plan)
        summary = compute["summary"]
        a4 = summary["arms"]["A4"]
        assert a4["dropped"] == "fidelity_not_reproduced" and a4["guards"]["G_fid"] is False
        assert a4["guards"]["pass"] is False and a4["guards_pass"] is False
        assert a4["disposition"]["disposition"] == 4 and a4["ranker"]["auc"] is None
        assert all(c["estimate"] is None and c["token"] == eb.STOP_TOKEN and c["reverse_token"] == eb.STOP_TOKEN
                   for c in a4["contrasts"].values())
        within = summary["within_family"]["A4-A1"]
        assert within["estimate"] is None and within["token"] == eb.STOP_TOKEN
        assert summary["program_claim_entries"]["A4"]["dropped"] == "fidelity_not_reproduced"
        claim = eb.program_claim(summary["program_claim_entries"])
        assert {"arm": "A4", "reason": "dropped: fidelity_not_reproduced"} in claim["vetoes"]
        assert "A4" not in summary["extra"]["pending_arms"] and summary["extra"]["dropped"]["A4"]["decision"] == DECISION
        assert summary["arms"]["A2"]["dropped"] is None and summary["arms"]["A2"]["ranker"]["auc"] is not None
        text = r128.findings_md(plan, compute)
        assert "Dropped (no records" in text and "A4 (fidelity_not_reproduced" in text
        eb.write_json(marker, {**eb.read_json(marker), "written_at": "2026-10-29T00:00:00+00:00"})
        with pytest.raises(ValueError):  # the marker is bound to the freeze by its sha256
            r128.compute_tables(output, plan)
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_no_marker_path_byte_identical_to_frozen_runner(context, monkeypatch, tmp_path):
    """Without a drop marker the amended runner writes the same readout-freeze manifest, records and publication
    bytes as the runner frozen in the #128 plan (loaded from the freeze commit) on the same synthetic inputs."""
    import hashlib
    import importlib.util
    import shutil
    import subprocess

    source = subprocess.run(["git", "show", f"{FREEZE_COMMIT}:{r128.RUNNER}"], cwd=eb.ROOT, capture_output=True,
                            check=True).stdout
    pinned = {e["artifact"]: e["sha256"] for e in eb.read_json(REAL_PLAN)["inputs"]}
    assert "sha256:" + hashlib.sha256(source).hexdigest() == pinned[r128.RUNNER]
    (tmp_path / "r128_frozen.py").write_bytes(source)
    spec = importlib.util.spec_from_file_location("r128_frozen", tmp_path / "r128_frozen.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    plan = {**eb.read_json(REAL_PLAN), "scoring_groups": {"t": ["A2"]}}
    outputs = {"frozen": _synthetic_output(), "amended": _synthetic_output()}
    try:
        _train_group(outputs["frozen"], plan, ["A2"], context)
        shutil.copytree(outputs["frozen"], outputs["amended"])  # the same model bytes for both runners
        for name, module in (("frozen", frozen), ("amended", r128)):
            output = outputs[name]
            monkeypatch.setattr(module, "load_plan", lambda out: plan)
            monkeypatch.setattr(module, "novelty_section", lambda *a: None)
            _stand_ins(monkeypatch, output, [module])
            assert module.freeze_readouts(output, ["A2"]) == 0
            assert module.score(output, ["A2"]) == 0
            text = module.rendered(plan, module.compute_tables(output, plan))
            (output / "rendered.json").write_text(json.dumps(text, sort_keys=True))
        a, b = outputs["frozen"], outputs["amended"]
        assert (a / "readouts-manifest.json").read_text().replace(a.name, "X") == \
            (b / "readouts-manifest.json").read_text().replace(b.name, "X")
        for seed in eb.SEEDS:
            ra, rb = (eb.read_json(r128.record_path(o, "A2", seed)) for o in (a, b))
            assert ra["inventories"] == rb["inventories"] and ra["typed_failure"] == rb["typed_failure"]
        assert (a / "rendered.json").read_text().replace(a.name, "X") == \
            (b / "rendered.json").read_text().replace(b.name, "X")
    finally:
        for output in outputs.values():
            shutil.rmtree(output, ignore_errors=True)
