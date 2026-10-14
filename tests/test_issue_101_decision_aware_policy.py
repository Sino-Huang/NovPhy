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
