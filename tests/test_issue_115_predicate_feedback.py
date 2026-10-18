"""Issue #115: corruption calibration, oracle alignment and the label-free bottleneck."""
from __future__ import annotations

import math

import torch

from world_model.model import Abstraction, PredictionPair
from world_model.training import decision_dynamics as dd
from world_model.training import predicate_feedback as pf
from world_model.training import request_policy as rp
from world_model.training import request_selectors as rs
from world_model.training.cnn_hybrid import DIM, PAIRS, SLOTS, CNNHybridPredictor

MICRO, MACRO = Abstraction.MICRO, Abstraction.MACRO


def carriers(count, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(count, DIM, generator=g) * 0.5
    z[:, 2::13] = (torch.rand(count, SLOTS, generator=g) > 0.3).float()
    return z


def actions(count, seed=1):
    return torch.rand(count, 5, generator=torch.Generator().manual_seed(seed))


def model_of(cls=CNNHybridPredictor, seed=3):
    torch.manual_seed(seed)
    return cls().eval()


# --- feedback path equals the predictor's own path --------------------------------------------------------

def test_gate_and_carry_reproduce_the_predictor():
    model, z, a = model_of(), carriers(7), actions(7)
    with torch.no_grad():
        for mode in (MICRO, MACRO):
            assert torch.equal(pf.gate(z, mode, *model.symbols(z, mode)), model.symbolic_features(z, mode))
        for pair in PAIRS:
            features = None if pair.abstraction == Abstraction.CONTINUOUS else pf.own_features(model, z, pair)
            assert torch.equal(pf.carry(model, z, a, pair, features), model.carrier(z, a, pair))


def test_default_feedback_rollout_equals_request_selectors():
    model, z0 = model_of(), carriers(1)[0]
    a = actions(6)
    torch.manual_seed(5)
    policy = rp.RequestPolicy(PAIRS).eval()
    for spec in ({"kind": "fixed", "pair_index": 4}, {"kind": "fixed", "pair_index": 8},
                 rs.temporal_spec(PAIRS, "continuous"), rs.description_spec(PAIRS, 5)):
        use = None if spec["kind"] == "fixed" else policy
        first, t1 = rs.rollout_endpoints(model, use, z0, a, spec, record_schedules=True)
        feedback = pf.Feedback("default", MICRO, device="cpu")
        second, t2 = pf.rollout_endpoints(model, use, z0, a, spec, feedback=feedback, record_schedules=True)
        assert torch.equal(first, second) and t1["schedules"] == t2["schedules"]
        assert t1["linear_macs"] == t2["linear_macs"]


# --- oracle alignment -------------------------------------------------------------------------------------

def labelled_truth(rows, frames, seed=9):
    g = torch.Generator().manual_seed(seed)
    relations = torch.rand(rows, frames, SLOTS, SLOTS, 2, generator=g) < 0.05
    upper = torch.triu(relations[..., 0], 1)
    relations[..., 0] = upper | upper.transpose(-1, -2)
    mask = torch.rand(rows, frames, SLOTS, generator=g) < 0.7
    pair_mask = mask[..., :, None] & mask[..., None, :] & ~torch.eye(SLOTS, dtype=torch.bool)
    relation_mask = pair_mask[..., None].expand(-1, -1, -1, -1, 2).clone()
    values = torch.rand(rows, frames, 2, 3, generator=g) < 0.2
    available = torch.ones(rows, frames, 2, 3, dtype=torch.bool)
    return relations & relation_mask, relation_mask, values, available


class Recording(pf.Feedback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def features(self, model, z_all, rows, positions, pair):
        out = super().features(model, z_all, rows, positions, pair)
        if pair.abstraction == self.mode:
            self.seen.append((rows.clone(), positions.clone(), out.clone()))
        return out


def test_oracle_reads_the_branch_label_at_the_matching_frame():
    rows, frames = 5, 60
    relations, relation_mask, values, available = labelled_truth(rows, frames)
    last = torch.tensor([59, 12, 30, 59, 3])
    truth = pf.BlockTruth(last, relations, relation_mask, values, available)
    model = model_of()
    z0, a = carriers(1)[0], actions(rows)
    feedback = Recording("oracle", MICRO, truth=truth, device="cpu")
    pf.rollout_endpoints(model, None, z0, a, {"kind": "fixed", "pair_index": PAIRS.index(PredictionPair(5, MICRO))},
                         feedback=feedback, endpoint=55)
    positions = sorted({int(p) for _, p, _ in feedback.seen for p in p})
    assert positions == list(range(0, 55, 5))
    for r, p, out in feedback.seen:
        t = torch.minimum(p, last[r])
        expected = pf.truth_features(MICRO, relations[r, t], relation_mask[r, t])
        assert torch.equal(out, expected)
    counts = feedback.counts.sum(0)
    assert int(counts[:, 1].sum()) == 0 and int(counts[:, 2].sum()) == 0 and int(counts[:, 0].sum()) > 0
    assert torch.allclose(pf.f1_from(counts), torch.ones(2, dtype=torch.float64))
    assert feedback.clamped == sum(int((p > last[r]).sum()) for r, p, _ in feedback.seen)


def test_macro_oracle_uses_the_request_delta_column():
    rows, frames = 3, 40
    _, _, values, available = labelled_truth(rows, frames)
    truth = pf.BlockTruth(torch.full((rows,), frames - 1), None, None, values, available)
    feedback = pf.Feedback("oracle", MACRO, truth=truth, device="cpu")
    model = model_of()
    z = carriers(rows)
    for delta in pf.DELTAS:
        pos = torch.tensor([0, 7, 39])
        out = feedback.features(model, z, torch.arange(rows), pos, PredictionPair(delta, MACRO))
        d = pf.DELTAS.index(delta)
        assert torch.equal(out[:, :2], (values[torch.arange(rows), pos, :, d]).float())


def test_removed_and_shuffled_feedback():
    model, z = model_of(), carriers(4)
    pair = PredictionPair(5, MICRO)
    removed = pf.Feedback("removed", MICRO, device="cpu")
    assert not bool(removed.features(model, z, torch.arange(4), torch.zeros(4, dtype=torch.long), pair).any())
    partner = torch.tensor([1, 2, 3, 0])
    shuffled = pf.Feedback("shuffled", MICRO, partner=partner, device="cpu")
    out = shuffled.features(model, z, torch.tensor([0, 2]), torch.zeros(2, dtype=torch.long), pair)
    assert torch.equal(out, model.symbolic_features(z[[1, 3]], MICRO))
    other = shuffled.features(model, z, torch.tensor([0]), torch.zeros(1, dtype=torch.long), PredictionPair(5, MACRO))
    assert torch.equal(other, model.symbolic_features(z[[0]], MACRO))


# --- corruption calibration -------------------------------------------------------------------------------

def test_flips_hit_the_rate_keep_contact_symmetric_and_touch_only_labels():
    relations, relation_mask, values, available = labelled_truth(40, 50)
    relations, relation_mask = relations.flatten(0, 1), relation_mask.flatten(0, 1)
    for p in (0.05, 0.1, 0.2, 0.4):
        flipped, tally = pf.flip_relations(relations, relation_mask, p, torch.Generator().manual_seed(1), chunk=333)
        rate = tally["flipped"] / tally["eligible"]
        sd = math.sqrt(p * (1 - p) / tally["eligible"])
        assert abs(rate - p) < 5 * sd
        assert torch.equal(flipped[..., 0], flipped[..., 0].transpose(1, 2))
        assert not bool((flipped ^ relations)[~relation_mask].any())
        changed = (flipped ^ relations)
        upper = pf.upper_triangle()
        assert int((changed[..., 0] & upper).sum()) + int(changed[..., 1].sum()) == tally["flipped"]
    same, tally = pf.flip_relations(relations, relation_mask, 0.0, torch.Generator().manual_seed(1))
    assert torch.equal(same, relations) and tally["flipped"] == 0
    flat_values, flat_available = values.flatten(0, 1), available.flatten(0, 1)
    out, tally = pf.flip_values(flat_values, flat_available, 0.2, torch.Generator().manual_seed(2))
    assert abs(tally["flipped"] / tally["eligible"] - 0.2) < 5 * math.sqrt(0.16 / tally["eligible"])


def test_flip_f1_follows_the_symmetric_noise_formula():
    relations, relation_mask, _, _ = labelled_truth(200, 20)
    relations, relation_mask = relations.flatten(0, 1), relation_mask.flatten(0, 1)
    p = 0.1
    flipped, _ = pf.flip_relations(relations, relation_mask, p, torch.Generator().manual_seed(4))
    counts = pf.micro_counts(flipped.float(), relations, relation_mask)
    f1 = pf.f1_from(counts)[1].item()
    prevalence = relations[..., 1][relation_mask[..., 1]].float().mean().item()
    precision = (1 - p) * prevalence / ((1 - p) * prevalence + p * (1 - prevalence))
    expected = 2 * precision * (1 - p) / (precision + 1 - p)
    assert abs(f1 - expected) < 0.02


def test_bisection_hits_a_target_on_a_decreasing_curve():
    curve = lambda sigma: 0.9 / (1 + sigma)  # noqa: E731
    for target in (0.72, 0.45, 0.2):
        result = pf.bisect_log(curve, target, iterations=20)
        assert result["reached"] and abs(result["f1"] - target) < 1e-3
    assert not pf.bisect_log(curve, 0.95)["reached"]
    assert not pf.bisect_log(curve, 0.001)["reached"]


def test_gaussian_feedback_keeps_contact_symmetric_and_degrades_f1():
    model, z = model_of(), carriers(64)
    pair = PredictionPair(1, MICRO)
    rows, positions = torch.arange(64), torch.zeros(64, dtype=torch.long)
    clean = pf.Feedback("default", MICRO, device="cpu").features(model, z, rows, positions, pair)
    noisy = pf.Feedback("gauss", MICRO, sigma=3.0, generator=torch.Generator().manual_seed(0),
                        device="cpu").features(model, z, rows, positions, pair)
    values = noisy[:, :-2].reshape(64, SLOTS, SLOTS, 2)
    assert torch.allclose(values[..., 0], values[..., 0].transpose(1, 2))
    assert not torch.equal(clean, noisy)
    assert torch.equal(clean[:, -2:], noisy[:, -2:])


# --- the bottleneck and the Delta-indexed macro head ------------------------------------------------------

def parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_variant_capacity():
    base = parameters(CNNHybridPredictor())
    assert parameters(pf.BinaryMicroPredictor()) == base
    assert parameters(pf.DeltaMacroPredictor()) == base + 128 * 8 + 8
    assert parameters(pf.BinaryDeltaMacroPredictor()) == base + 128 * 8 + 8
    model = pf.DeltaMacroPredictor()
    assert model.macro_adapter.in_features == 4 and model.micro_adapter.in_features == pf.MICRO_WIDTH


def test_binary_bottleneck_codes_are_discrete_and_trainable_through_prediction():
    model, z, a = model_of(pf.BinaryMicroPredictor), carriers(9), actions(9)
    features = model.symbolic_features(z, MICRO)
    available = features[:, -2:]
    assert set(available.unique().tolist()) <= {0.0, 1.0}
    values = features[:, :-2].reshape(9, SLOTS, SLOTS, 2)
    presence = z[:, 2::13].clamp(0, 1)
    active = presence[:, :, None] * presence[:, None, :] * (1 - torch.eye(SLOTS))
    codes = values / active[..., None].clamp_min(1e-9)
    on = active[..., None].expand_as(values) > 0
    assert set(codes[on].round().unique().tolist()) <= {0.0, 1.0}
    assert torch.allclose(codes[on], codes[on].round())
    model.train()
    out = model.carrier(z, a, PredictionPair(1, MICRO))
    out.square().sum().backward()
    assert model.micro_head[0].weight.grad is not None and float(model.micro_head[0].weight.grad.abs().sum()) > 0
    macro = model_of(pf.BinaryDeltaMacroPredictor)
    for delta in pf.DELTAS:
        f = macro.symbolic_features(z, MACRO, delta)
        assert set(f.unique().tolist()) <= {0.0, 1.0}


def test_delta_macro_head_selects_the_request_column():
    model, z = model_of(pf.DeltaMacroPredictor), carriers(5)
    raw = model.macro_head(z).unflatten(-1, (3, 4))
    for d, delta in enumerate(pf.DELTAS):
        logits, availability = model.symbols(z, MACRO, delta)
        assert torch.equal(logits, raw[:, d, :2]) and torch.equal(availability, raw[:, d, 2:])


# --- the training loss ------------------------------------------------------------------------------------

def batch_of(members, rows, macro_delta=False, seed=11):
    g = torch.Generator().manual_seed(seed)
    relations, relation_mask, values, available = labelled_truth(members * rows, 61, seed)
    shape = (members, rows, 61)
    batch = {"z": torch.rand(*shape, DIM, generator=g) * 0.3, "action": torch.rand(members, rows, 5, generator=g),
             "length": torch.tensor([[60, 33, 12, 60][:rows]] * members),
             "relations": relations.reshape(*shape, SLOTS, SLOTS, 2),
             "relations_mask": relation_mask.reshape(*shape, SLOTS, SLOTS, 2)}
    if macro_delta:
        batch["macros"] = values.reshape(*shape, 2, 3)
        batch["macros_mask"] = available.reshape(*shape, 2, 3)
    else:
        batch["macros"] = values[..., 0].reshape(*shape, 2)
        batch["macros_mask"] = available[..., 0].reshape(*shape, 2)
    return batch


def ensemble_of(cls, members=2):
    models = []
    for seed in range(members):
        torch.manual_seed(seed)
        models.append(cls())
    return dd.Ensemble(models, learning_rate=1e-4, weight_decay=1e-4, grad_clip=1.0, compile=False)


def test_default_spec_reproduces_decision_dynamics_local_loss():
    ensemble = ensemble_of(CNNHybridPredictor)
    batch = batch_of(2, 4)
    for pair in (PAIRS[1], PAIRS[5], PAIRS[6]):
        assert torch.equal(pf.local_loss(ensemble, batch, pair), dd.local_loss(ensemble, batch, pair))


def test_label_free_and_weighted_specs():
    ensemble = ensemble_of(CNNHybridPredictor)
    batch = batch_of(2, 4)
    pair = PAIRS[4]  # (5, micro)
    free = pf.local_loss(ensemble, batch, pair, pf.SymbolicSpec(micro="none"))
    scrambled = {**batch, "relations": ~batch["relations"], "relations_mask": ~batch["relations_mask"]}
    assert torch.equal(free, pf.local_loss(ensemble, scrambled, pair, pf.SymbolicSpec(micro="none")))
    assert not torch.equal(free, pf.local_loss(ensemble, batch, pair))
    weighted = pf.SymbolicSpec(weighted=frozenset({"micro"}))
    batch["micro_labelled"] = torch.ones(2, 4, 61, dtype=torch.bool)
    assert torch.equal(pf.local_loss(ensemble, batch, pair, weighted), pf.local_loss(ensemble, batch, pair))
    batch["micro_labelled"] = torch.zeros(2, 4, 61, dtype=torch.bool)
    assert torch.equal(pf.local_loss(ensemble, batch, pair, weighted),
                       pf.local_loss(ensemble, batch, pair, pf.SymbolicSpec(micro="none")))


def test_delta_macro_spec_reads_the_delta_column():
    ensemble = ensemble_of(pf.DeltaMacroPredictor)
    batch = batch_of(2, 4, macro_delta=True)
    spec = pf.SymbolicSpec(macro_delta=True)
    for d, delta in enumerate(pf.DELTAS):
        pair = PredictionPair(delta, MACRO)
        value = pf.local_loss(ensemble, batch, pair, spec)
        assert torch.isfinite(value).all()
        other = dict(batch)
        other["macros"] = batch["macros"].clone()
        for e in range(len(pf.DELTAS)):
            if e != d:
                other["macros"][..., e] = ~other["macros"][..., e]
        assert torch.equal(pf.local_loss(ensemble, other, pair, spec), value)
        changed = dict(batch)
        changed["macros"] = batch["macros"].clone()
        changed["macros"][..., d] = ~changed["macros"][..., d]
        assert not torch.equal(pf.local_loss(ensemble, changed, pair, spec), value)


class FakeTemporal(torch.nn.Module):
    history = 4

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, history, delta):
        self.seen.append(history.clone())
        return torch.zeros(len(history), 2)


class Tracing(pf.Feedback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []

    def observe(self, z, index, positions):
        super().observe(z, index, positions)
        self.states.append((positions[index].clone(), z[index].clone()))


def test_temporal_feedback_reads_the_rows_own_stride_history():
    model, z0, a = model_of(pf.DeltaMacroPredictor), carriers(1)[0], actions(3)
    head = FakeTemporal()
    rows = 3
    _, _, values, available = labelled_truth(rows, 60)
    truth = pf.BlockTruth(torch.full((rows,), 59), None, None, values, available)
    feedback = Tracing("temporal", MACRO, truth=truth, temporal=head, rows=rows, endpoint=40, device="cpu")
    pf.rollout_endpoints(model, None, z0, a, {"kind": "fixed", "pair_index": PAIRS.index(PredictionPair(5, MACRO))},
                         feedback=feedback, endpoint=40)
    for call, (positions, _) in zip(head.seen, feedback.states):
        t = int(positions[0])
        for j in range(4):
            q = max(t - (3 - j) * 5, 0)
            expected = next(s for p, s in feedback.states if int(p[0]) == q)
            assert torch.equal(call[:, j], expected)


# --- guard propagation on the published compute.json (phase-B/C review) -------------------------------------

TOKENS = {"supported", "not_supported_by_this_experiment", "readiness_or_precision_insufficient", "unsupported"}


def token_values(obj):
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in token_values(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in token_values(v)]
    return [obj] if isinstance(obj, str) and obj in TOKENS else []


def test_every_failed_guard_invalidates_every_published_token():
    import copy
    import json

    import pytest

    from scripts import run_predicate_feedback_robustness as runner
    if not (runner.OUTPUT / "compute.json").is_file():
        pytest.skip("issue-115 publication not present")
    compute = json.loads((runner.OUTPUT / "compute.json").read_text())
    plan = json.loads((runner.OUTPUT / "plan.json").read_text())
    assert runner.apply_guard_failure(copy.deepcopy(compute)) == compute  # every retained guard passes
    assert "supported" in token_values(compute["dispositions"])
    for guard in compute["guards"]:
        failed = copy.deepcopy(compute)
        failed["guards"][guard]["pass"] = False
        runner.apply_guard_failure(failed)
        assert failed["guard_failure"]["failed_guards"] == [guard]
        tokens = token_values(failed) + token_values(runner.summary(plan, failed))
        assert tokens and set(tokens) == {runner.STOP_TOKEN}, guard
        for d, entry in failed["dispositions"].items():
            assert set(entry["contrast_tokens"].values()) == {runner.STOP_TOKEN}
            assert entry["reference_description_value_J_minus_T"]["token"] == runner.STOP_TOKEN
            assert entry["contrasts"] == compute["dispositions"][d]["contrasts"]
        assert f"Guard failure ({guard})" in runner.findings_md(plan, failed)
