import itertools

import torch

from scripts import run_tau_ad_within_checkpoint as t96
from world_model.training import request_policy as rp
from world_model.training import request_selectors as rs
from world_model.training.cnn_hybrid import DIM, PAIRS, CNNHybridPredictor, linear_macs
from world_model.training.matched_dynamics import CONTINUOUS_PAIRS, work

R_CONT = tuple(i for i, p in enumerate(PAIRS) if p in CONTINUOUS_PAIRS)


def predictor(seed=0):
    torch.manual_seed(seed)
    return CNNHybridPredictor().eval()


def random_logits(seed=0, size=96):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(size, 9, generator=generator)
    logits[::4, 6:] = -torch.inf  # Delta = 15 infeasible
    logits[1::7, 3:] = -torch.inf  # only Delta = 1 feasible
    logits[2::5] = logits[2::5].round()  # ties
    return logits


def no_previous(size):
    return torch.full((size,), -1, dtype=torch.long)


# ---------------------------------------------------------------------------
# static restricted kinds
# ---------------------------------------------------------------------------

def test_restricting_to_all_nine_equals_unrestricted_joint_argmax():
    logits = random_logits()
    spec = rs.restricted_spec(PAIRS, PAIRS)
    assert spec["allowed"] == tuple(range(9))
    assert torch.equal(rs.select_requests(logits, spec, PAIRS, no_previous(len(logits))),
                       t96.select_pairs("joint", logits, {}))


def test_temporal_and_description_equal_t96_selectors():
    logits = random_logits(1)
    previous = no_previous(len(logits))
    for a, alpha in enumerate(t96.ALPHAS):
        spec = rs.temporal_spec(PAIRS, alpha)
        assert spec["allowed"] == (a, 3+a, 6+a)
        assert torch.equal(rs.select_requests(logits, spec, PAIRS, previous),
                           t96.select_pairs("temporal", logits, {"alpha_index": a}))
    for d, delta in enumerate(t96.DELTAS):
        spec = rs.description_spec(PAIRS, delta)
        assert spec["allowed"] == (3*d, 3*d+1, 3*d+2)
        feasible = torch.isfinite(logits[:, 3*d]).nonzero(as_tuple=True)[0]  # D-Delta needs Delta <= remaining
        assert torch.equal(rs.select_requests(logits[feasible], spec, PAIRS, previous[feasible]),
                           t96.select_pairs("description", logits[feasible], {"delta_index": d}))


def test_diagonal_and_anti_diagonal_subsets_and_brute_force():
    assert [t96.PAIR_NAMES[i] for i in rs.diagonal_spec(PAIRS)["allowed"]] == [
        "1-continuous", "5-micro", "15-macro"]
    assert [t96.PAIR_NAMES[i] for i in rs.anti_diagonal_spec(PAIRS)["allowed"]] == [
        "1-macro", "5-micro", "15-continuous"]
    logits = random_logits(2)
    for spec in (rs.diagonal_spec(PAIRS), rs.anti_diagonal_spec(PAIRS)):
        chosen = rs.select_requests(logits, spec, PAIRS, no_previous(len(logits)))
        for row, k in zip(logits, chosen.tolist()):
            best = max(spec["allowed"], key=lambda i: (float(row[i]), -i))
            assert k == best


def test_no_feasible_allowed_request_raises():
    logits = torch.zeros(2, 9)
    logits[:, 3:] = -torch.inf
    spec = rs.restricted_spec(PAIRS, [(15, "macro")])
    try:
        rs.select_requests(logits, spec, PAIRS, no_previous(2))
    except ValueError:
        return
    raise AssertionError("an infeasible restricted argmax must raise")


# ---------------------------------------------------------------------------
# monotone (path-dependent)
# ---------------------------------------------------------------------------

def monotone_allowed(previous, k):
    if previous < 0:
        return True
    dp, ap = divmod(previous, 3)
    d, a = divmod(k, 3)
    delta_p, delta = t96.DELTAS[dp], t96.DELTAS[d]
    return (delta-delta_p)*(a-ap) >= 0


def test_monotone_matches_brute_force_and_always_has_a_feasible_request():
    logits = random_logits(3, size=90)
    previous = torch.tensor([-1, *range(9)]*9)
    chosen = rs.select_requests(logits, rs.monotone_spec(), PAIRS, previous)
    for row, p, k in zip(logits, previous.tolist(), chosen.tolist()):
        options = [i for i in range(9) if monotone_allowed(p, i) and torch.isfinite(row[i])]
        assert k == max(options, key=lambda i: (float(row[i]), -i))
    # with only Delta = 1 feasible, some Delta = 1 request is allowed after any previous request
    for p in range(9):
        assert any(monotone_allowed(p, k) for k in range(3))
    table = rs.monotone_table(PAIRS)
    for p, k in itertools.product(range(9), range(9)):
        assert bool(table[p, k]) == monotone_allowed(p, k)


def test_monotone_rollout_schedules_respect_the_rule():
    model = predictor(4)
    torch.manual_seed(4)
    policy = rp.RequestPolicy(PAIRS)
    z0 = torch.randn(DIM)*0.1
    actions = torch.rand(6, 5)
    _, timing = rs.rollout_endpoints(model, policy, z0, actions, rs.monotone_spec(), endpoint=40,
                                     record_schedules=True)
    for schedule in timing["schedules"]:
        assert sum(PAIRS[k].delta for k in schedule) == 40
        for p, k in zip([-1, *schedule], schedule):
            assert monotone_allowed(p, k)


# ---------------------------------------------------------------------------
# rollout: equality with request_policy.rollout_endpoints and replay
# ---------------------------------------------------------------------------

def test_rollout_restricted_to_all_nine_equals_request_policy_joint_rollout():
    model = predictor(5)
    torch.manual_seed(5)
    policy = rp.RequestPolicy(PAIRS)
    z0 = torch.randn(DIM)*0.1
    actions = torch.rand(5, 5)
    reference, ref_timing = rp.rollout_endpoints(model, policy, z0, actions, {"kind": "joint"},
                                                 select=t96.select_pairs, endpoint=30, record_schedules=True)
    mine, timing = rs.rollout_endpoints(model, policy, z0, actions, rs.restricted_spec(PAIRS, PAIRS),
                                        endpoint=30, record_schedules=True)
    assert torch.equal(reference, mine)
    assert timing["schedules"] == ref_timing["schedules"]
    for key in ("transition_calls", "controller_calls", "linear_macs"):
        assert timing[key] == ref_timing[key]


def test_rollout_equals_request_policy_on_its_kinds():
    model = predictor(6)
    torch.manual_seed(6)
    policy = rp.RequestPolicy(PAIRS)
    z0 = torch.randn(DIM)*0.1
    actions = torch.rand(4, 5)
    index = t96.PAIR_NAMES.index("5-macro")
    cases = [(None, {"kind": "fixed", "pair_index": index}), (None, {"kind": "schedule", "schedule": [index]*6}),
             (policy, {"kind": "joint"}), (policy, {"kind": "factorized"})]
    for owner, spec in cases:
        a, ta = rp.rollout_endpoints(model, owner, z0, actions, spec, select=t96.select_pairs, endpoint=30,
                                     record_schedules=True)
        b, tb = rs.rollout_endpoints(model, owner, z0, actions, spec, select=t96.select_pairs, endpoint=30,
                                     record_schedules=True)
        assert torch.equal(a, b) and ta["schedules"] == tb["schedules"]
        assert ta["linear_macs"] == tb["linear_macs"] and ta["controller_calls"] == tb["controller_calls"]


def test_temporal_rollout_equals_t96_temporal_kind():
    model = predictor(7)
    torch.manual_seed(7)
    policy = rp.RequestPolicy(PAIRS)
    z0 = torch.randn(DIM)*0.1
    actions = torch.rand(4, 5)
    a, ta = rp.rollout_endpoints(model, policy, z0, actions, {"kind": "temporal", "alpha_index": 0},
                                 select=t96.select_pairs, endpoint=30, record_schedules=True)
    b, tb = rs.rollout_endpoints(model, policy, z0, actions, rs.temporal_spec(PAIRS, "continuous"), endpoint=30,
                                 record_schedules=True)
    assert torch.equal(a, b) and ta["schedules"] == tb["schedules"] and ta["linear_macs"] == tb["linear_macs"]
    assert all(PAIRS[k].abstraction.value == "continuous" for s in tb["schedules"] for k in s)


def test_replay_reproduces_the_joint_rollout_without_policy_calls():
    model = predictor(8)
    torch.manual_seed(8)
    policy = rp.RequestPolicy(PAIRS)
    z0 = torch.randn(DIM)*0.1
    actions = torch.rand(5, 5)
    joint, timing = rs.rollout_endpoints(model, policy, z0, actions, {"kind": "joint"}, select=t96.select_pairs,
                                         endpoint=30, record_schedules=True)
    replayed, again = rs.rollout_endpoints(model, None, z0, actions, rs.replay_spec(timing["schedules"]),
                                           endpoint=30, record_schedules=True)
    assert torch.equal(joint, replayed) and again["schedules"] == timing["schedules"]
    assert again["controller_calls"] == 0 and again["transition_calls"] == timing["transition_calls"]
    assert again["linear_macs"] == timing["linear_macs"]-timing["controller_calls"]*rp.policy_macs(policy)
    # alpha replaced by continuous: same Delta sequence and call count, continuous per-call cost
    as_cont = [[3*(k//3) for k in s] for s in timing["schedules"]]
    _, cont = rs.rollout_endpoints(model, None, z0, actions, rs.replay_spec(as_cont), endpoint=30)
    assert cont["transition_calls"] == timing["transition_calls"]
    assert cont["linear_macs"] == timing["transition_calls"]*linear_macs(model, PAIRS[0])


# ---------------------------------------------------------------------------
# training on a restricted request set
# ---------------------------------------------------------------------------

def test_masked_policy_shares_parameters_and_all_allowed_equals_request_policy():
    plain = rp.new_policy(PAIRS, 11, device="cpu")
    masked_all = rs.new_masked_policy(PAIRS, range(9), 11, device="cpu")
    masked_cont = rs.new_masked_policy(PAIRS, R_CONT, 11, device="cpu")
    assert list(plain.state_dict()) == list(masked_cont.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), masked_cont.state_dict().values()))
    z, action = torch.randn(6, DIM), torch.rand(6, 5)
    remaining = torch.tensor([1, 4, 5, 14, 15, 225])
    assert torch.equal(plain(z, action, remaining), masked_all(z, action, remaining))
    logits = masked_cont(z, action, remaining)
    assert torch.isinf(logits[:, [i for i in range(9) if i not in R_CONT]]).all()
    assert torch.equal(logits[:, list(R_CONT)], plain(z, action, remaining)[:, list(R_CONT)])
    assert rp.policy_macs(masked_cont) == rp.policy_macs(plain)


def windows(seed, count=3, frames=21):
    generator = torch.Generator().manual_seed(seed)
    return [{"z": torch.randn(frames, DIM, generator=generator)*0.1, "action": torch.rand(5, generator=generator),
             "length": torch.tensor(length)} for length in (frames-1, 13, 7)[:count]]


def test_restricted_window_rows_with_all_requests_equal_window_rows():
    model = predictor(9)
    policy = rp.new_policy(PAIRS, 9, device="cpu")
    items = windows(9)
    for aggregated in (False, True):
        reference = rp.window_rows(model, policy, items, aggregated=aggregated, device="cpu")
        mine = rs.restricted_window_rows(model, policy, items, range(9), aggregated=aggregated, device="cpu")
        assert all(torch.equal(reference[k], mine[k]) for k in reference)


@torch.no_grad()
def naive_restricted_labels(model, policy, z, action, length, allowed, compute_weight):
    """Independent single-window DP over the allowed requests (ties to the lower index)."""
    frames = z.shape[0]
    reference = work(model, rp.REFERENCE_PAIR)
    value = [0.0]*frames
    labels = [0]*(frames-1)
    for t in range(frames-2, -1, -1):
        if t >= length:
            continue
        best = None
        for i in allowed:
            pair = PAIRS[i]
            end = t+pair.delta
            if end >= frames or end > length:
                continue
            pred = model.carrier(z[t][None], action[None], pair)[0]
            cost = pair.delta*float((pred-z[end]).square().mean())
            cost += compute_weight*work(model, pair, policy)/reference+value[end]
            if best is None or cost < best[0]:
                best = (cost, i)
        value[t], labels[t] = best
    return labels


def test_restricted_window_rows_run_the_dp_over_the_allowed_requests_only():
    model = predictor(10)
    policy = rs.new_masked_policy(PAIRS, R_CONT, 10, device="cpu")
    items = windows(10)
    rows = rs.restricted_window_rows(model, policy, items, R_CONT, aggregated=False, compute_weight=1e-2,
                                     device="cpu")
    assert set(rows["labels"].tolist()) <= set(R_CONT)
    start = 0
    for item in items:
        n = int(item["length"])
        expected = naive_restricted_labels(model, policy, item["z"], item["action"], n, R_CONT, 1e-2)[:n]
        assert rows["labels"][start:start+n].tolist() == expected
        start += n
    aggregated = rs.restricted_window_rows(model, policy, items, R_CONT, aggregated=True, device="cpu")
    assert set(aggregated["labels"].tolist()) <= set(R_CONT)
    # the masked policy trains on the restricted labels with a finite loss
    losses = rp.distil_round(policy, [aggregated], 0, seed=10, steps=5, batch_size=16, device="cpu")
    assert all(torch.isfinite(torch.tensor(losses)))
