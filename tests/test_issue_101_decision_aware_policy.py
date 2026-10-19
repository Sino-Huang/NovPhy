import math

import numpy as np
import torch
from torch.nn import functional as F

from scripts import run_tau_ad_within_checkpoint as t96
from world_model.training import request_policy as rp
from world_model.training.cnn_hybrid import DIM, PAIRS, CNNHybridPredictor
from world_model.training.matched_dynamics import MatchedController, controller_labels, work


def predictor(seed=0):
    torch.manual_seed(seed)
    return CNNHybridPredictor().eval()


def window_batch(seed, size=3, frames=21):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(size, frames, DIM, generator=generator)*0.1
    action = torch.rand(size, 5, generator=generator)
    length = torch.tensor([frames-1, 13, 7][:size])
    return z, action, length


# ---------------------------------------------------------------------------
# replication of the existing (ICLR / #77) label path
# ---------------------------------------------------------------------------

def test_request_policy_is_matched_controller():
    torch.manual_seed(5)
    matched = MatchedController(False)
    torch.manual_seed(5)
    mine = rp.RequestPolicy(PAIRS)
    assert list(matched.state_dict()) == list(mine.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(matched.state_dict().values(), mine.state_dict().values()))
    z, action = torch.randn(7, DIM), torch.rand(7, 5)
    remaining = torch.tensor([1, 3, 5, 14, 15, 60, 225])
    assert torch.equal(matched(z, action, remaining), mine(z, action, remaining))
    assert rp.policy_macs(mine) == (DIM+6)*128+128*9


def test_prediction_error_labels_equal_controller_labels_with_contexts():
    model = predictor()
    torch.manual_seed(1)
    controller = MatchedController(False)
    z, action, length = window_batch(2)
    contexts = z+0.01*torch.randn_like(z)
    reference = controller_labels(model, controller, z, action, length, contexts)
    ours = rp.prediction_error_labels(model, controller, z, action, length, contexts)
    assert torch.equal(reference, ours)
    # lambda is a parameter: a huge weight makes the cheapest request (largest feasible Delta) win
    heavy = rp.prediction_error_labels(model, controller, z, action, length, compute_weight=1e3)
    first = PAIRS[int(heavy[0, 0])]
    assert t96.PAIR_NAMES[int(heavy[0, 0])] in ("5-continuous", "15-continuous"), first


def test_window_rows_equal_issue77_label_lineage_loop():
    model = predictor(2)
    torch.manual_seed(3)
    controller = MatchedController(False)
    z, action, length = window_batch(4)
    windows = [{"z": z[i], "action": action[i], "length": length[i]} for i in range(len(z))]
    for aggregated in (False, True):
        records = []
        for position in range(len(z)):  # the #77 label_lineage body, verbatim
            batch_z, a, n_ = z[position:position+1], action[position:position+1], length[position:position+1]
            contexts = batch_z.clone()
            if aggregated:
                current, t = batch_z[:, 0], 0
                with torch.no_grad():
                    while t < int(n_[0]):
                        pair = controller.pairs[int(controller(current, a, n_-t).argmax(-1))]
                        current = model.carrier(current, a, pair)
                        t += pair.delta
                        contexts[:, t] = current
            labels = controller_labels(model, controller, batch_z, a, n_, contexts)
            n = int(n_[0])
            records.append({"z": contexts[0, :n], "action": a.expand(n, -1), "remaining": torch.arange(n, 0, -1),
                            "labels": labels[0, :n]})
        reference = {k: torch.cat([r[k] for r in records]) for k in records[0]}
        ours = rp.window_rows(model, controller, windows, aggregated=aggregated, device="cpu")
        assert all(torch.equal(reference[k], ours[k]) for k in reference)


def test_first_window_follows_s99_rule():
    carriers = torch.arange(30*DIM, dtype=torch.float32).reshape(30, DIM)
    window = rp.first_window(carriers, torch.zeros(5), end=29, window_length=60)
    assert int(window["length"]) == 29 and window["z"].shape == (61, DIM)
    assert torch.equal(window["z"][29], carriers[29]) and torch.equal(window["z"][60], carriers[29])


def test_distil_round_matches_issue77_loop():
    rows = [{"z": torch.randn(40, DIM), "action": torch.rand(40, 5), "remaining": torch.randint(15, 61, (40,)),
             "labels": torch.randint(0, 9, (40,))} for _ in range(3)]
    torch.manual_seed(11+7700)
    reference = MatchedController(False)
    optim = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(11+7700)
    for step in range(25):
        data = rows[step % 3]
        indices = torch.randint(len(data["z"]), (8,), generator=generator)
        batch = {k: v[indices] for k, v in data.items()}
        loss = F.cross_entropy(reference(batch["z"], batch["action"], batch["remaining"]), batch["labels"])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    policy = rp.new_policy(PAIRS, 11, device="cpu")
    rp.distil_round(policy, rows, 0, seed=11, steps=25, batch_size=8, device="cpu")
    assert all(torch.equal(a, b) for a, b in zip(reference.state_dict().values(), policy.state_dict().values()))


# ---------------------------------------------------------------------------
# decision-valued label construction
# ---------------------------------------------------------------------------

def test_candidate_rank_loss_signs_and_missing_partners():
    costs = torch.tensor([0.0, 1.0, 2.0, float("nan")])
    verdicts = torch.tensor([True, False, False, False])
    sign, partners = rp.partner_terms(verdicts, costs)
    loss = rp.candidate_rank_loss(costs, sign, partners, temperature=1.0)
    softplus = lambda x: math.log1p(math.exp(x))  # noqa: E731
    assert math.isclose(float(loss[0]), (softplus(-1.0)+softplus(-2.0))/2, rel_tol=1e-6)  # nan partner skipped
    assert math.isclose(float(loss[1]), softplus(-1.0), rel_tol=1e-6)  # a failure wants a higher cost
    assert math.isclose(float(loss[2]), softplus(-2.0), rel_tol=1e-6)
    assert math.isinf(float(loss[3]))  # nonfinite own cost


def test_fallback_keeps_alpha_and_largest_feasible_delta():
    table = rp.fallback_table(PAIRS)
    f15_macro = t96.PAIR_NAMES.index("15-macro")
    assert table[f15_macro, 15] == f15_macro
    assert t96.PAIR_NAMES[int(table[f15_macro, 14])] == "5-macro"
    assert t96.PAIR_NAMES[int(table[f15_macro, 4])] == "1-macro"


def tie_free(z_end, z0):
    return z_end[:, 3]-0.1*torch.hypot(z_end[:, 4]-z0[:, 4], z_end[:, 5]-z0[:, 5])


def naive_q(model, policy, z0, action, position, state, request, partners, sign, lam, continuation, endpoint):
    pairs = policy.pairs
    z, t, spent, first = state[None], position, 0.0, True
    with torch.no_grad():
        while t < endpoint:
            remaining = endpoint-t
            if first:
                k, first = request, False
            elif continuation["kind"] == "fixed":
                k = int(rp.fallback_table(pairs, endpoint)[continuation["pair_index"], remaining])
            else:
                k = int(policy(z, action[None], torch.tensor([remaining])).argmax(-1))
            z = model.carrier(z, action[None], pairs[k])
            t += pairs[k].delta
            spent += float(work(model, pairs[k], policy))
        cost = float(tie_free(z, z0[None])[0])
    terms = [math.log1p(math.exp(sign*(cost-p)/rp.RANK_TEMPERATURE)) for p in partners]
    return sum(terms)/len(terms)+lam*spent/work(model, rp.REFERENCE_PAIR)


def test_decision_rows_equal_brute_force_lookahead():
    model = predictor(4)
    endpoint = 20
    generator = torch.Generator().manual_seed(4)
    cells = [{"z0": torch.randn(DIM, generator=generator)*0.1, "actions": torch.rand(3, 5, generator=generator),
              "verdicts": torch.tensor([True, False, False])},
             {"z0": torch.randn(DIM, generator=generator)*0.1, "actions": torch.rand(2, 5, generator=generator),
              "verdicts": torch.tensor([False, True])}]
    torch.manual_seed(9)
    policy = rp.RequestPolicy(PAIRS)
    for continuation in ({"kind": "fixed", "pair_index": t96.PAIR_NAMES.index("15-macro")}, {"kind": "policy"}):
        rows = rp.decision_rows(model, policy, cells, continuation=continuation, cost_fn=tie_free,
                                compute_weight=1e-3, endpoint=endpoint, device="cpu")
        base = rows["continuation_costs"]
        verdicts = torch.cat([c["verdicts"] for c in cells])
        assert len(rows["labels"]) == len(rows["q"]) > 0
        for r in range(len(rows["labels"])):
            g = int(rows["candidate"][r])
            c = int(rows["cell_of_candidate"][g])
            first = 0 if c == 0 else 3
            members = range(first, first+len(cells[c]["actions"]))
            partners = [float(base[j]) for j in members if bool(verdicts[j]) != bool(verdicts[g])]
            sign = 1.0 if bool(verdicts[g]) else -1.0
            position = int(rows["position"][r])
            assert int(rows["remaining"][r]) == endpoint-position
            expected = []
            for request, pair in enumerate(PAIRS):
                if pair.delta > endpoint-position:
                    expected.append(math.inf)
                    continue
                expected.append(naive_q(model, policy, cells[c]["z0"], cells[c]["actions"][g-first], position,
                                        rows["z"][r], request, partners, sign, 1e-3, continuation, endpoint))
            got = rows["q"][r].double()
            for e, q in zip(expected, got.tolist()):
                # batched vs single-row float noise, amplified by T = 0.1; q is stored float32
                assert (math.isinf(e) and math.isinf(q)) or abs(e-q) <= 1e-4*max(1.0, abs(e))
            assert int(rows["labels"][r]) == int(np.argmin(expected)) or abs(
                min(expected)-expected[int(rows["labels"][r])]) <= 1e-4*max(1.0, abs(min(expected)))


def test_decision_rows_drop_states_without_opposite_partner():
    model = predictor(6)
    cells = [{"z0": torch.zeros(DIM), "actions": torch.rand(2, 5), "verdicts": torch.tensor([True, True])}]
    rows = rp.decision_rows(model, rp.RequestPolicy(PAIRS), cells, cost_fn=tie_free, compute_weight=0.0,
                            continuation={"kind": "fixed", "pair_index": 8}, endpoint=15, device="cpu")
    assert len(rows["labels"]) == 0


# ---------------------------------------------------------------------------
# restricted schedules (M, OL, schedule kind)
# ---------------------------------------------------------------------------

def test_factorized_selector_matches_brute_force_marginals():
    logits = torch.randn(64, 9)
    logits[::5, 6:] = -torch.inf  # Delta = 15 infeasible
    chosen = t96.select_pairs("factorized", logits, {})
    for row, k in zip(logits, chosen.tolist()):
        grid = row.reshape(3, 3)
        delta = max(range(3), key=lambda d: (float(torch.logsumexp(grid[d], 0)), -d))
        alpha = max(range(3), key=lambda a: (float(torch.logsumexp(grid[:, a], 0)), -a))
        assert k == delta*3+alpha


def test_open_loop_schedule_is_feasible_and_reproducible():
    weights = [5, 0, 1, 0, 0, 0, 3, 0, 9]
    first = rp.open_loop_schedule(weights, PAIRS, np.random.Generator(np.random.PCG64(7202)))
    again = rp.open_loop_schedule(weights, PAIRS, np.random.Generator(np.random.PCG64(7202)))
    assert first == again
    assert sum(PAIRS[k].delta for k in first) == rp.ENDPOINT
    assert set(first) <= {0, 2, 6, 8}
    position = 0
    for k in first:
        assert PAIRS[k].delta <= rp.ENDPOINT-position
        position += PAIRS[k].delta


def test_schedule_kind_equals_fixed_and_records_schedules():
    model = predictor(7)
    z0 = torch.randn(DIM)*0.1
    actions = torch.rand(4, 5)
    index = t96.PAIR_NAMES.index("5-micro")
    fixed, timing = rp.rollout_endpoints(model, None, z0, actions, {"kind": "fixed", "pair_index": index},
                                         select=t96.select_pairs, endpoint=20)
    schedule, timing2 = rp.rollout_endpoints(model, None, z0, actions, {"kind": "schedule", "schedule": [index]*4},
                                             select=t96.select_pairs, endpoint=20, record_schedules=True)
    assert torch.equal(fixed, schedule) and timing["linear_macs"] == timing2["linear_macs"]
    assert timing2["schedules"] == [[index]*4]*4 and timing2["controller_calls"] == 0
    torch.manual_seed(1)
    policy = rp.RequestPolicy(PAIRS)
    _, joint = rp.rollout_endpoints(model, policy, z0, actions, {"kind": "joint"}, select=t96.select_pairs,
                                    endpoint=20, record_schedules=True)
    assert all(sum(PAIRS[k].delta for k in s) == 20 for s in joint["schedules"])
    from world_model.training.cnn_hybrid import linear_macs
    expected = sum(linear_macs(model, PAIRS[k]) for s in joint["schedules"] for k in s)
    assert joint["linear_macs"] == expected+joint["controller_calls"]*rp.policy_macs(policy)
    assert joint["controller_calls"] == sum(len(s) for s in joint["schedules"])


# ---------------------------------------------------------------------------
# publication: failed-guard propagation, post-freeze amendment registry (phase-A review)
# ---------------------------------------------------------------------------

def tokens_of(obj):
    if isinstance(obj, dict):
        return [v for k, v in obj.items() if k == "token" and not isinstance(v, dict)] + [
            t for v in obj.values() for t in tokens_of(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in tokens_of(v)]
    return []


def test_a_failed_guard_turns_every_token_insufficient():
    from scripts import run_decision_aware_policy as dap
    tables = {"grid": {"contrasts": {"J_dec-J": {"token": "supported", "estimate": {"mean": 0.04}}},
                       "lambda_sweep": {"dec": [{"lambda": 0.1, "minus_F_hindsight": {"mean": -0.016}}]}}}
    wrong = {"right_minus_wrong": {"mean": 0.13}, "token": "supported"}
    out_tables, out_wrong = dap.invalidate(tables), dap.invalidate(wrong)
    assert out_tables["grid"]["contrasts"]["J_dec-J"] == {"token": dap.STOP_TOKEN, "estimate": {"mean": 0.04}}
    assert out_tables["grid"]["lambda_sweep"] == tables["grid"]["lambda_sweep"]
    assert out_wrong == {"right_minus_wrong": {"mean": 0.13}, "token": dap.STOP_TOKEN}


def test_failed_guard_publication_carries_no_other_token():
    """The published #101 compute, as the failed-guard path would leave it: every token in the summary
    and the findings table is readiness_or_precision_insufficient; estimates are unchanged."""
    import json
    from pathlib import Path
    import pytest
    from scripts import run_decision_aware_policy as dap
    path = Path(dap.OUTPUT) / "compute.json"
    if not path.is_file():
        pytest.skip("retained #101 publication not present")
    compute = json.loads(path.read_text())
    failed = {**compute, "inventories": dap.invalidate(compute["inventories"]),
              "wrong_anchor": dap.invalidate(compute["wrong_anchor"]),
              "primary": {**compute["primary"], "token": dap.STOP_TOKEN}}
    plan = {"frozen_at": compute["frozen_at"], "claim_boundary": "test"}
    tokens = tokens_of(dap.summary(plan, failed)) + tokens_of(failed)
    assert tokens and set(tokens) == {dap.STOP_TOKEN}
    assert dap.summary(plan, failed)["contrasts"]["grid"]["J_dec-J"]["estimate"] == \
        compute["inventories"]["grid"]["contrasts"]["J_dec-J"]["estimate"]


def test_post_freeze_amendments_accept_only_frozen_or_listed_bytes():
    import hashlib
    from scripts import post_freeze_amendments as pfa
    for artifact, entry in pfa.AMENDMENTS.items():
        current = "sha256:" + hashlib.sha256(open(artifact, "rb").read()).hexdigest()
        assert current == entry["sha256"], f"{artifact}: amended bytes changed without a registry update"
        for frozen in entry["replaces"]:
            assert pfa.matches(artifact, frozen, entry["sha256"])
            assert pfa.matches(artifact, frozen, frozen)
            assert not pfa.matches(artifact, frozen, "sha256:" + "0" * 64)
        assert not pfa.matches(artifact, "sha256:" + "1" * 64, entry["sha256"])
    assert not pfa.matches("scripts/run_gate_b.py", "sha256:a", "sha256:b")


def test_phase_bc_amended_runners_and_modules_are_registered():
    from scripts import post_freeze_amendments as pfa
    issues = {entry["issue"] for entry in pfa.AMENDMENTS.values()}
    assert {106, 115, 117, 119, 120, 121} <= issues
    for path in ("scripts/run_pixel_benchmarks.py", "world_model/data/pixel_benchmarks.py",
                 "world_model/training/pixel_carrier.py", "scripts/run_predicate_feedback_robustness.py",
                 "scripts/run_request_interference_benchmarks.py", "scripts/run_remaining_frames_input.py",
                 "scripts/run_cost_alignment.py", "scripts/run_request_code_factorisation.py"):
        assert path in pfa.AMENDMENTS, path


def test_issue_123_gpu_window_amendments_are_registered():
    """#123 A1 (owner GPU-window rule): graceful stop / resume of the main training, Tier-T launcher and fidelity
    queue; each replaces exactly the bytes frozen in the #123 plan."""
    import json
    from scripts import post_freeze_amendments as pfa
    plan = json.load(open(".local-artifacts/issue-123-hjepa-comparison-v1/plan.json"))
    frozen = {entry["artifact"]: entry["sha256"] for entry in plan["inputs"]}
    for path in ("scripts/hjepa_train_ckpt.py", "scripts/run_hjepa_fidelity.py"):
        assert pfa.AMENDMENTS[path]["issue"] == 123
        assert pfa.AMENDMENTS[path]["replaces"] == (frozen[path],)


def test_issue_123_prescore_amendments_are_registered():
    """#123 A2 (pre-scoring fidelity review): the runner's second link (frozen -> A1 -> A2), and the G-port reference
    and the #123 tests, each replacing exactly the bytes frozen in the #123 plan; load_plan accepts all three."""
    import hashlib
    import json
    from scripts import post_freeze_amendments as pfa
    plan = json.load(open(".local-artifacts/issue-123-hjepa-comparison-v1/plan.json"))
    frozen = {entry["artifact"]: entry["sha256"] for entry in plan["inputs"]}
    runner = "scripts/run_hjepa_comparison.py"
    assert pfa.AMENDMENTS[runner]["issue"] == 123
    assert pfa.AMENDMENTS[runner]["replaces"] == (
        frozen[runner], "sha256:b3390059a7d83c8433c5dc13fc211c25c6522d466cc11baf56f75268dfe4f6f4")
    for path in ("scripts/hjepa_port_reference.py", "tests/test_issue_123_hjepa_comparison.py"):
        assert pfa.AMENDMENTS[path]["issue"] == 123
        assert pfa.AMENDMENTS[path]["replaces"] == (frozen[path],)
    for path in (runner, "scripts/hjepa_port_reference.py", "tests/test_issue_123_hjepa_comparison.py"):
        current = "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert pfa.matches(path, frozen[path], current)
