"""Pareto-set and bootstrap frontier-share mechanics of the #103 runner."""
import numpy as np

from scripts import run_engine_truth_frontier as etf
from scripts import run_launch_power_probe as p94


def brute_pareto(x, y):
    out = []
    for i in range(len(x)):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            out.append(False)
            continue
        out.append(not any(np.isfinite(y[j]) and x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i])
                           for j in range(len(x))))
    return out


def rows_of(values):
    """values: {policy: [per-member cell values (None = undefined)]}, one cell per member."""
    members = len(next(iter(values.values())))
    return [{"member": f"m{k:02d}", "seed": 0, "views": {p: {"auc": v[k]} for p, v in values.items()}}
            for k in range(members)]


def test_pareto_known_cases():
    x = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 1.0])
    y = np.array([0.5, 0.7, 0.7, 0.8, 0.6, 0.7])
    assert etf.pareto_mask(x, y).tolist() == [True, True, False, True, False, True]


def test_pareto_undefined_points_are_off_and_dominate_nothing():
    x = np.array([1.0, 2.0, np.nan])
    y = np.array([np.nan, 0.6, 0.9])
    assert etf.pareto_mask(x, y).tolist() == [False, True, False]


def test_pareto_batch_matches_brute_force():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 5, size=12).astype(float)
    ys = rng.integers(0, 4, size=(40, 12)).astype(float)
    ys[rng.random(ys.shape) < 0.05] = np.nan
    batch = etf.pareto_batch(x, ys)
    for d in range(len(ys)):
        assert batch[d].tolist() == brute_pareto(x, ys[d])


def test_cluster_means_match_block_of_point():
    rows = rows_of({"a": [0.9, 0.7, None, 0.5], "b": [0.1, 0.2, 0.3, 0.4]})
    rows += [{"member": "m00", "seed": 1, "views": {"a": {"auc": 0.8}, "b": {"auc": 0.3}}}]
    members, means, defined = etf.cluster_matrix(rows, ["a", "b"], "auc")
    assert members == ["m00", "m01", "m02", "m03"]
    point = means[defined[:, 0], 0].mean()
    expected = p94.bootstrap([0.9, 0.7, 0.5, 0.8], ["m00", "m01", "m03", "m00"])["mean"]
    assert abs(point - expected) < 1e-12
    assert defined[:, 0].tolist() == [True, True, False, True]


def test_shares_dominance_and_draws():
    rows = rows_of({"a": [0.8, 0.82, 0.84, 0.86], "b": [0.6, 0.62, 0.64, 0.66], "c": [0.9, 0.5, 0.9, 0.5]})
    shares = etf.frontier_shares(rows, ["a", "b", "c"], np.array([1.0, 2.0, 3.0]), "auc", draws=400)
    assert shares[0] == 1.0 and shares[1] == 0.0  # a beats b in every member at lower x
    assert 0.0 < shares[2] < 1.0  # c beats a only when its high members dominate the draw


def test_shares_use_the_p94_draw_matrix():
    """With one policy defined everywhere, the share draws are the p94 bootstrap draws: the policy is on
    the frontier iff its draw mean beats a constant competitor at lower x."""
    values = [0.551, 0.652, 0.753, 0.854, 0.955]
    rows = rows_of({"a": values, "k": [0.7537] * 5})
    shares = etf.frontier_shares(rows, ["k", "a"], np.array([1.0, 2.0]), "auc", draws=300)
    generator = np.random.default_rng(np.random.PCG64(p94.BOOTSTRAP_SEED))
    draws = generator.integers(0, 5, size=(300, 5))
    expected = float(np.mean(np.array(values)[draws].mean(axis=1) > 0.7537))
    assert abs(shares[1] - expected) < 1e-12
