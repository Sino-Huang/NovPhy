"""Issue-116 request-set scaling: the R_9 replication guard, request indexing, the endpoint rule, the pruned
and vectorized DP, the policy heads and the disposition rules (CPU, data-free)."""
import math

import pytest
import torch

from scripts import run_request_set_scaling as rss
from world_model.model import Abstraction, PredictionPair
from world_model.training import request_policy as rp
from world_model.training import request_sets as rs
from world_model.training import stable_decision_dynamics as sd
from world_model.training.cnn_hybrid import PAIRS, CNNHybridPredictor

R9, R18, R30, RC = (rs.request_set(n) for n in ("R_9", "R_18", "R_30", "R_cont"))


def batch(rows, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, 236, generator=generator) * 0.3, torch.rand(rows, 5, generator=generator)


# ---------------------------------------------------------------------------
# R_9 replication guard
# ---------------------------------------------------------------------------

def test_r9_is_the_hard_coded_request_set():
    assert R9.pairs == PAIRS
    assert [R9.request_name(k) for k in range(9)] == [f"F-{p.delta}-{p.abstraction}" for p in PAIRS]
    assert rs.schedule(R9, "total") == [step % 9 for step in range(9000)] == rs.schedule(R9, "request")
    assert all(rs.curriculum_step(u, 9000) == u for u in range(0, 9000, 7))
    assert rs.skip_cap(9000, 45) == 45


@pytest.mark.parametrize("name", ["R_9", "R_cont"])
def test_generalised_carrier_equals_cnn_hybrid_bit_for_bit(name):
    torch.manual_seed(3)
    reference = CNNHybridPredictor().eval()
    ours = rs.load_predictor(rs.request_set(name), reference.state_dict(), "cpu")
    z, action = batch(17)
    with torch.no_grad():
        for pair in PAIRS:
            assert torch.equal(reference.carrier(z, action, pair), ours.carrier(z, action, pair))


def test_generalised_carrier_rejects_pairs_outside_its_set_and_accepts_long_delta():
    model = rs.predictor_class(R9)()
    z, action = batch(3)
    with pytest.raises(ValueError):
        model.carrier(z, action, PredictionPair(2, Abstraction.CONTINUOUS))
    wide = rs.load_predictor(RC, model.state_dict(), "cpu")
    with torch.no_grad():
        assert torch.isfinite(wide.carrier(z, action, PredictionPair(75, Abstraction.MACRO))).all()


def tiny_ensemble(request_set_):
    torch.manual_seed(5)
    model = rs.predictor_class(request_set_)()
    return rs.RequestSetEnsemble([model], learning_rate=1e-3, weight_decay=0.0, grad_clip=1.0, compile=False)


def loss_inputs(delta, horizon, seed=1):
    generator = torch.Generator().manual_seed(seed)
    steps = horizon // delta
    long = {"start": torch.randn(1, 3, 236, generator=generator) * 0.3, "action": torch.rand(1, 3, 5, generator=generator),
            "targets": torch.randn(1, 3, steps, 236, generator=generator) * 0.3}
    rank = {"z0": torch.randn(1, 1, 236, generator=generator) * 0.3, "action": torch.rand(1, 1, 3, 5, generator=generator),
            "verdict": torch.tensor([[[1, 0, -1]]])}
    return long, rank


@pytest.mark.parametrize("delta,horizon", [(1, 60), (5, 75), (15, 225)])
def test_generalised_long_and_rank_loss_equals_v3_for_dividing_delta(delta, horizon):
    ensemble = tiny_ensemble(R9)
    pair = PredictionPair(delta, Abstraction.MICRO)
    long, rank = loss_inputs(delta, horizon)
    reference = sd.truncated_long_and_rank_loss(ensemble, pair, long=long, rank=rank, horizon=horizon, endpoint=225,
                                                presence=2, center=(5, 6), temperature=0.1, truncation=45)
    schedule = [R9.pairs[j] for j in rs.endpoint_schedule(R9.pairs, R9.index(pair))]
    ours = rs.long_and_rank_loss(ensemble, pair, long=long, rank=rank, rank_pairs=schedule, horizon=horizon, presence=2,
                                 center=(5, 6), temperature=0.1)
    assert all(torch.equal(a, b) for a, b in zip(reference, ours))


def test_long_and_rank_loss_follows_the_endpoint_schedule_for_non_dividing_delta():
    calls = []

    class Counting:
        size, device = 1, torch.device("cpu")

        def transition(self, pair, z, action):
            calls.append((pair.delta, z.shape[1]))
            return z * 0.999

    pair = PredictionPair(74, Abstraction.CONTINUOUS)
    long, rank = loss_inputs(74, 148)
    schedule = [RC.pairs[j] for j in rs.endpoint_schedule(RC.pairs, RC.index(pair))]
    long_loss, rank_loss = rs.long_and_rank_loss(Counting(), pair, long=long, rank=rank, rank_pairs=schedule,
                                                 horizon=148, presence=2, center=(5, 6), temperature=0.1)
    assert [d for d, _ in calls] == [74, 74, 74, 3]
    assert [rows for _, rows in calls] == [6, 6, 3, 3]  # long rows drop after two transitions
    assert torch.isfinite(long_loss).all() and torch.isfinite(rank_loss).all()


# ---------------------------------------------------------------------------
# request indexing
# ---------------------------------------------------------------------------

def test_request_sets_are_frozen_and_delta_major():
    assert [s.size for s in (R9, R18, R30, RC)] == [9, 18, 30, 225]
    assert R18.deltas == (1, 2, 3, 5, 10, 15) and R30.deltas == (1, 2, 3, 5, 8, 10, 15, 25, 45, 75)
    assert RC.deltas == tuple(range(1, 76))
    for request_set_ in (R9, R18, R30, RC):
        for k, pair in enumerate(request_set_.pairs):
            assert request_set_.index(pair) == k
            assert request_set_.delta_of(k) == pair.delta and request_set_.alpha_of(k) == pair.abstraction
            assert k == request_set_.deltas.index(pair.delta) * 3 + list(Abstraction).index(pair.abstraction)


def test_sub_requests_keep_order_and_cover_alpha():
    subset = rs.sub_requests(RC, (1, 5, 15))
    assert [RC.pairs[i] for i in subset] == list(PAIRS)
    with pytest.raises(ValueError):
        rs.RequestSet("bad", (2, 3))


def test_factorised_index_maps_match_the_set():
    head = rs.FactorisedPolicy(R30, 16)
    assert head.delta_of.tolist() == [R30.deltas.index(R30.delta_of(k)) for k in range(R30.size)]
    assert head.alpha_of.tolist() == [k % 3 for k in range(R30.size)]


# ---------------------------------------------------------------------------
# the endpoint rule and local windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("request_set_", [R9, R18, R30, RC], ids=lambda s: s.name)
def test_endpoint_schedule_ends_exactly_at_225_with_the_same_alpha(request_set_):
    for k, pair in enumerate(request_set_.pairs):
        schedule = rs.endpoint_schedule(request_set_.pairs, k)
        assert sum(request_set_.pairs[j].delta for j in schedule) == 225
        assert all(request_set_.pairs[j].abstraction == pair.abstraction for j in schedule)
        assert schedule[:225 // pair.delta] == [k] * (225 // pair.delta)
        if 225 % pair.delta == 0:
            assert schedule == [k] * (225 // pair.delta)


def test_endpoint_schedule_examples():
    deltas = lambda s, k: [s.pairs[j].delta for j in rs.endpoint_schedule(s.pairs, k)]  # noqa: E731
    assert deltas(R18, R18.index(PredictionPair(2, Abstraction.MACRO))) == [2] * 112 + [1]
    assert deltas(R18, R18.index(PredictionPair(10, Abstraction.MICRO))) == [10] * 22 + [5]
    assert deltas(R30, R30.index(PredictionPair(8, Abstraction.CONTINUOUS))) == [8] * 28 + [1]
    assert deltas(RC, RC.index(PredictionPair(74, Abstraction.CONTINUOUS))) == [74] * 3 + [3]


def test_window_positions_reproduce_the_s99_rule_and_extend_for_long_delta():
    starts, offsets, lasts = torch.tensor([0, 15, 225]), torch.tensor([100, 100, 100]), torch.tensor([285, 285, 285])
    positions, length = rs.window_positions(starts, offsets, lasts, 15)
    assert positions.shape == (3, 61) and length.tolist() == [60, 60, 60]
    assert positions[0].tolist() == [100 + i for i in range(61)]
    positions, length = rs.window_positions(starts, offsets, lasts, 75)
    assert positions.shape == (3, 301) and length.tolist() == [285, 270, 60]
    assert int(positions[2, -1]) == 100 + 225 + 60
    assert rs.local_window(15) == 60 and rs.local_window(16) == 64


# ---------------------------------------------------------------------------
# DP: vectorized and chunked labels equal request_policy; the pruned DP
# ---------------------------------------------------------------------------

def windows(count, frames, seed=2):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(count, frames, 236, generator=generator) * 0.3
    action = torch.rand(count, 5, generator=generator)
    length = torch.randint(5, frames, (count,), generator=generator)
    return z, action, length


def test_window_labels_equal_request_policy_labels():
    torch.manual_seed(7)
    model = rs.predictor_class(R9)().eval()
    policy = rp.RequestPolicy(PAIRS)
    z, action, length = windows(3, 31)
    for i in range(3):
        reference = rp.prediction_error_labels(model, policy, z[i:i + 1], action[i:i + 1], length[i:i + 1])
        ours, _ = rs.window_labels(model, PAIRS, z[i:i + 1], action[i:i + 1], length[i:i + 1], mac_policy=policy, chunk=1)
        assert torch.equal(reference, ours)


def test_backward_dp_equals_the_per_request_loop():
    generator = torch.Generator().manual_seed(4)
    deltas = [1, 2, 3, 5, 8]
    size, frames = 6, 24
    costs = torch.rand(size, frames, len(deltas), generator=generator)
    for i, d in enumerate(deltas):
        costs[:, frames - d:, i] = torch.inf
    length = torch.randint(3, frames, (size,), generator=generator)
    values = torch.zeros(size, frames)
    labels = torch.zeros(size, frames - 1, dtype=torch.long)
    for t in range(frames - 2, -1, -1):
        choices = costs[:, t].clone()
        for i, d in enumerate(deltas):
            if t + d < frames:
                choices[:, i] += values[:, t + d]
                choices[:, i].masked_fill_(length < t + d, torch.inf)
        minimum, labels[:, t] = choices.min(-1)
        values[:, t] = torch.where(length > t, minimum, 0.)
    ours, ours_values = rs.backward_dp(costs, deltas, length)
    assert torch.equal(ours, labels) and torch.equal(ours_values, values)


def test_batched_dagger_equals_request_policy_dagger():
    torch.manual_seed(8)
    model = rs.predictor_class(R9)().eval()
    policy = rp.RequestPolicy(PAIRS)
    z, action, length = windows(3, 25)
    ours = rs.batched_dagger(model, policy, z, action, length, chunk=1)
    for i in range(3):
        reference = rp.dagger_contexts(model, policy, z[i:i + 1], action[i:i + 1], length[i:i + 1])
        assert torch.equal(reference[0], ours[i])


def test_pruned_grid_and_pruned_dp():
    assert rs.pruned_deltas() == (1, 2, 3, 5, 7, 11, 16, 23, 34, 51, 75)
    torch.manual_seed(9)
    model = rs.predictor_class(RC)().eval()
    z, action, length = windows(2, 18)
    exact, exact_values = rs.window_labels(model, RC.pairs, z, action, length, chunk=2)
    subset = rs.sub_requests(RC, rs.pruned_deltas())
    pruned, pruned_values = rs.window_labels(model, RC.pairs, z, action, length, chunk=2, subset=subset)
    mask = torch.arange(17)[None] < length[:, None]
    assert torch.isin(pruned[mask], torch.tensor(subset)).all()
    assert (pruned_values >= exact_values * (1 - 1e-6)).all()
    assert 0.0 <= rs.disagreement(exact, pruned, length) <= 1.0


def test_label_statistics():
    assert rs.entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0)
    assert rs.entropy_bits([5, 0, 0]) == 0.0
    labels = torch.tensor([[0, 1, 2], [2, 2, 0]])
    assert rs.label_histogram(labels, torch.tensor([3, 1]), 4) == [1, 1, 2, 0]
    assert rs.disagreement(labels, torch.tensor([[0, 1, 1], [2, 0, 0]]), torch.tensor([3, 1])) == 0.25


@pytest.mark.parametrize("spec", [{"kind": "fixed", "pair_index": 4}, {"kind": "schedule", "schedule": [8] * 14 + [5] * 3},
                                  {"kind": "joint"}])
def test_generalised_rollout_equals_request_policy_rollout(spec):
    from scripts import run_tau_ad_within_checkpoint as t96
    torch.manual_seed(10)
    model = rs.predictor_class(R9)().eval()
    policy = rp.RequestPolicy(PAIRS)
    z0, _ = batch(1)
    _, actions = batch(7, seed=3)
    holder = policy if spec["kind"] == "joint" else None
    reference, ref_timing = rp.rollout_endpoints(model, holder, z0[0], actions, spec, select=t96.select_pairs,
                                                 record_schedules=True)
    ours, timing = rs.rollout_endpoints(model, holder, z0[0], actions, spec, select=t96.select_pairs, pairs=PAIRS,
                                        record_schedules=True)
    assert torch.equal(reference, ours)
    assert all(timing[k] == ref_timing[k] for k in ("transition_calls", "controller_calls", "linear_macs", "schedules"))


# ---------------------------------------------------------------------------
# policy heads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,request_set_", [("factorised", R18), ("factorised", RC), ("ordinal", RC)])
def test_heads_return_normalized_masked_joint_log_probabilities(kind, request_set_):
    head = rs.new_head(kind, request_set_, 20260908, width=32, device="cpu")
    z, action = batch(40)
    remaining = torch.randint(1, 226, (40,))
    scores = head(z, action, remaining)
    deltas = torch.tensor([p.delta for p in request_set_.pairs])
    feasible = deltas[None] <= remaining[:, None]
    assert torch.isinf(scores[~feasible]).all() and torch.isfinite(scores[feasible]).all()
    assert torch.allclose(torch.logsumexp(scores, -1), torch.zeros(40), atol=1e-5)
    choice = scores.argmax(-1)
    assert (deltas[choice] <= remaining).all()


def test_ordinal_head_starts_uniform_over_delta():
    head = rs.OrdinalPolicy(RC, 16)
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.bias.zero_()
        log_delta = head.delta_log_probs(torch.zeros(1, 16), torch.tensor([225]))
    assert torch.allclose(log_delta.exp(), torch.full((1, 75), 1 / 75), atol=1e-5)


def test_matched_widths_track_the_flat_parameter_count():
    for request_set_ in (R9, RC):
        flat = rs.parameter_count(rp.RequestPolicy(request_set_.pairs))
        for kind in ("factorised", "ordinal") if request_set_ is RC else ("factorised",):
            width = rs.matched_width(kind, request_set_, flat)
            per_unit = rs.head_parameters(kind, request_set_, width + 1) - rs.head_parameters(kind, request_set_, width)
            assert abs(rs.head_parameters(kind, request_set_, width) - flat) <= per_unit


# ---------------------------------------------------------------------------
# disposition rules
# ---------------------------------------------------------------------------

def grid_with(tokens):
    return {"contrasts": {rss.primary_contrast_name(c): {"J-flat": {"token": t}} for c, t in tokens.items()}}


def test_ranking_disposition_needs_both_budgets():
    supported = {c: "not_supported_by_this_experiment" for c in rss.TRAINED_CONFIGS}
    supported.update({"R_30/total": "supported", "R_30/request": "supported"})
    assert rss.disposition_ranking(grid_with(supported), True)["token"] == "supported"
    one = {c: "not_supported_by_this_experiment" for c in rss.TRAINED_CONFIGS}
    one["R_cont/request"] = "supported"
    assert rss.disposition_ranking(grid_with(one), True)["token"] == rss.STOP_TOKEN
    none = {c: "not_supported_by_this_experiment" for c in rss.TRAINED_CONFIGS}
    assert rss.disposition_ranking(grid_with(none), True)["token"] == "not_supported_by_this_experiment"
    assert rss.disposition_ranking(grid_with(supported), False)["token"] == rss.STOP_TOKEN


def test_dp_disposition_thresholds():
    def dp(seconds, unit):
        configs = {c: {"round0_mean": {"seconds_per_evaluation": 1.0}, "per_seed": {}} for c in rss.CONFIGS}
        for c in ("R_cont/total", "R_cont/request"):
            configs[c]["per_seed"] = {"1": {"round0": {"gpu_seconds": seconds, "peak_memory_gb": 1.0,
                                                       "seconds_per_evaluation": unit}}}
        return {"configs": configs}
    assert rss.disposition_dp(dp(100.0, 1.5), True)["token"] == "supported"
    assert rss.disposition_dp(dp(4000.0, 1.0), True)["token"] == "not_supported_by_this_experiment"
    assert rss.disposition_dp(dp(100.0, 2.5), True)["token"] == "not_supported_by_this_experiment"
    assert rss.disposition_dp(dp(100.0, 1.0), False)["token"] == rss.STOP_TOKEN


def test_compact_rows_round_trip():
    rows = [{"ordinal": 3, "excluded": False, rss.PRIMARY_COST: 0.25, rss.harness.SECONDARY_COST: 1.0},
            {"ordinal": 5, "excluded": True, rss.PRIMARY_COST: None, rss.harness.SECONDARY_COST: None}]
    items = [{"ordinal": 3}, {"ordinal": 5}]
    assert rss.expand(items, rss.compact(rows)) == rows
    assert math.isclose(rss.compact(rows)[0][0], 0.25)
