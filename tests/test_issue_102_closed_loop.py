"""Episode mechanics of the #102 closed-loop runner (ranking, exclusion, random arm, recorded replay)."""
from collections import Counter

from scripts import run_closed_loop_evaluation as cl


def rows(costs, excluded=()):
    return [{"ordinal": o, "excluded": o in excluded, cl.PRIMARY_COST: None if o in excluded else c}
            for o, c in enumerate(costs)]


def test_model_ranking_lowest_cost_first_ties_to_lower_ordinal_nonfinite_last():
    assert cl.model_ranking(rows([0.3, 0.1, 0.1, -0.2], excluded=(0,))) == [3, 1, 2, 0]


def test_choose_skips_failed_candidates():
    assert cl.choose([4, 2, 7], [4]) == 2
    assert cl.choose([4, 2], [4, 2]) is None


def test_random_ranking_is_a_seeded_permutation():
    ordinals = list(range(16))
    first = cl.random_ranking(20260908, 3, ordinals)
    assert sorted(first) == ordinals
    assert first == cl.random_ranking(20260908, 3, ordinals)
    assert first != cl.random_ranking(20260909, 3, ordinals)


def test_random_first_pick_is_roughly_uniform():
    counts = Counter(cl.random_ranking(7, level, list(range(16)))[0] for level in range(4000))
    assert set(counts) == set(range(16))
    assert max(counts.values()) < 2 * min(counts.values())


def test_oracle_ranking_lists_recorded_passes_only():
    assert cl.oracle_ranking({"verdicts": {0: False, 5: True, 2: True}}) == [2, 5]
    assert cl.oracle_ranking({"verdicts": {0: False}}) == []


def context(verdicts):
    items = [{"ordinal": o, "suffix": f"c{o:02d}", "key": f"L-c{o:02d}", "action": {}} for o in range(4)]
    return {"levels": {"L": {"inventory": items, "verdicts": verdicts, "set": cl.N1, "stream": "s",
                             "recorded": {}, "recorded_any_pass": any(verdicts.values())}}}


def test_open_levels_excludes_failed_shots_and_closes_on_pass():
    ctx = context({0: False, 1: False, 2: True, 3: False})
    decisions = {(1, "a", 1): {"levels": {"L": {"chosen": 1, "execution": "e1"}}},
                 (2, "a", 1): {"levels": {"L": {"chosen": 0, "execution": "e0"}}}}
    fail = {"status": "complete", cl.PASS_FIELD: False}
    assert cl.open_levels(ctx, "a", 1, 2, decisions, {"e1": fail}) == {"L": [1]}
    assert cl.open_levels(ctx, "a", 1, 3, decisions, {"e1": fail, "e0": fail}) == {"L": [1, 0]}
    assert cl.open_levels(ctx, "a", 1, 2, decisions, {"e1": {"status": "complete", cl.PASS_FIELD: True}}) == {}
    assert cl.open_levels(ctx, "a", 1, 2, decisions, {"e1": {"status": "failed", cl.PASS_FIELD: None}}) == {}


def test_recorded_replay_walks_the_shot_one_ranking():
    ctx = context({0: False, 1: False, 2: True, 3: False})
    decisions = {(1, "a", 1): {"levels": {"L": {"ranking": [1, 0, 2, 3]}}}}
    assert cl.recorded_replay(ctx, decisions, "a", 1, "L") == {"pass_shot": 3}
    decisions = {(1, "a", 1): {"levels": {"L": {"ranking": [1, 0, 3, 2]}}}}
    assert cl.recorded_replay(ctx, decisions, "a", 1, "L") == {"pass_shot": None}


def test_episode_view_pass_at_k():
    shots = [{"shot": 1, "pass": False, "native_clear": False}, {"shot": 2, "pass": True, "native_clear": False}]
    view = cl.episode_view({"shots": shots, "outcome": "passed"}, {"pass_shot": 2})
    assert (view["pass1"], view["pass2"], view["pass3"]) == (0.0, 1.0, 1.0)
    assert view["shots_to_pass"] == 2.0 and view["replay_differs"] == 0.0
