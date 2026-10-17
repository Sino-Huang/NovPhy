from world_model.model import Abstraction
from world_model.training.cnn_hybrid import PAIRS

from scripts import run_request_interference_benchmarks as item5


def test_six_requests_follow_the_issue_79_84_cycle_and_arms_cover_them():
    assert item5.PAIRS6 == tuple(p for p in PAIRS if p.abstraction != Abstraction.MACRO)
    assert item5.REQUESTS == ("F-1-continuous", "F-1-micro", "F-5-continuous", "F-5-micro",
                              "F-15-continuous", "F-15-micro")
    assert item5.ARMS["J"] == tuple(range(6))
    assert [item5.ARMS[f"D{d}"] for d in item5.DELTAS] == [(0, 1), (2, 3), (4, 5)]
    assert all(item5.ARMS[item5.specialist_arm(r)] == (i,) for i, r in enumerate(item5.REQUESTS))


def test_budgets_replay_the_joint_updates_or_match_its_count():
    joint = item5.schedule_of("J", "full")
    assert joint == [(k % 6, k) for k in range(9000)]
    for arm, indices in item5.ARMS.items():
        if arm == "J":
            continue
        request = item5.schedule_of(arm, "request")
        assert request == [(p, g) for p, g in joint if p in indices] and len(request) == 1500 * len(indices)
        assert len(item5.schedule_of(arm, "full")) == 9000


def test_long_starts_keep_every_target_inside_the_clip_and_curriculum_caps_at_120():
    assert item5.long_starts(128) == tuple(range(8))
    for frames in (151, 170, 207, 242):
        starts = item5.long_starts(frames)
        assert len(starts) == 8 and starts[0] == 0 and starts[-1] + 120 == frames - 1
    assert item5.horizon_at(0) == 30 and item5.horizon_at(5399) == 105 and item5.horizon_at(5400) == 120
    assert all(item5.horizon_at(g) % 15 == 0 for g in range(0, 9000, 300))


def test_claim_token_reads_the_interval_against_1_5():
    block = {"clusters": 12, "interval": [0.8, 1.49]}
    assert item5.claim_token(block) == "supported"
    assert item5.claim_token({**block, "interval": [1.51, 3.0]}) == "not_supported_by_this_experiment"
    assert item5.claim_token({**block, "interval": [1.2, 1.6]}) == item5.STOP_TOKEN
    assert item5.claim_token({"clusters": 2, "interval": [0.1, 0.2]}) == item5.STOP_TOKEN
    assert item5.claim_token(None) == item5.STOP_TOKEN
