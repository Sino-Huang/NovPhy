import torch

from scripts import run_decision_dynamics_retrain as rt
from scripts import run_slot_encoder_fix as s99


def synthetic_shot(key, member, end, seed):
    generator = torch.Generator().manual_seed(seed)
    n = end + 1
    return {"key": key, "member": member, "end": end, "action": torch.rand(5, generator=generator),
            "carriers": torch.rand(n, 236, generator=generator),
            "relations": torch.rand(n, 18, 18, 2, generator=generator) > 0.5,
            "relation_mask": torch.rand(n, 18, 18, 2, generator=generator) > 0.5,
            "macros": torch.rand(n, 2, generator=generator) > 0.5,
            "macro_mask": torch.rand(n, 2, generator=generator) > 0.5}


def test_indexed_windows_gather_exactly_the_issue_99_windows_including_terminal_padding():
    shots = [synthetic_shot("a", "L1", 7, 0), synthetic_shot("b", "L1", 75, 1), synthetic_shot("c", "L2", 285, 2)]
    payload = rt.build_payload(shots, [])
    expected = []
    for shot in shots:
        labels = {"relations": shot["relations"].float(), "relation_mask": shot["relation_mask"],
                  "macros": shot["macros"].float(), "macro_mask": shot["macro_mask"]}
        expected.extend(s99.windows({"end": shot["end"], "action": shot["action"], "labels": labels}, shot["carriers"]))
    positions = payload["window_positions"]
    assert len(positions) == len(expected) == 1 + 5 + 16
    for w, row in enumerate(expected):
        assert torch.equal(payload["carriers"][positions[w]], row["z"])
        assert torch.equal(payload["relations"][positions[w]], row["relations"])
        assert torch.equal(payload["relation_mask"][positions[w]], row["relations_mask"])
        assert torch.equal(payload["macros"][positions[w]], row["macros"])
        assert torch.equal(payload["macro_mask"][positions[w]], row["macros_mask"])
        assert torch.equal(payload["shot_actions"][payload["window_shot"][w]], row["action"])
        assert int(payload["window_length"][w]) == int(row["length"])
    assert payload["long_windows"].tolist() == [[0, 0], [1, 0], [1, 15], [1, 30], [1, 45], [1, 60],
                                                [2, 0], [2, 15], [2, 30], [2, 45], [2, 60]]


def branch(level, suffix, entries, action):
    return {"key": f"{level}-{suffix}", "member": level,
            "candidate": {"suffix": suffix, "inventory_entries": entries, "action": action}}


def action(x):
    return {"drag_x": x, "drag_y": 10, "release_time_ms": 1000, "tap_time_ms": 0}


def test_ranking_cells_keep_mixed_verdict_bearing_candidates_in_ordinal_order(monkeypatch):
    monkeypatch.setattr(rt, "fit_levels", lambda: [{"identity": "L1"}, {"identity": "L2"}])
    branches = [
        branch("L1", "c00", [{"inventory": "angle", "ordinal": 1}], action(-80)),
        branch("L1", "c01", [{"inventory": "angle", "ordinal": 0}, {"inventory": "grid", "ordinal": 2}], action(-70)),
        branch("L1", "c02", [{"inventory": "grid", "ordinal": 0}], action(-60)),
        branch("L2", "c00", [{"inventory": "angle", "ordinal": 0}], action(-80)),
        branch("L2", "c01", [{"inventory": "angle", "ordinal": 1}], action(-70)),
    ]
    verdicts = {"L1-c00": True, "L1-c01": False, "L1-c02": False, "L2-c00": True, "L2-c01": True}
    anchors = {"L1": torch.ones(236), "L2": torch.zeros(236)}
    cells = rt.ranking_cells(branches, verdicts, anchors)
    # L1 grid holds only failures and L2 angle only removals: neither is a ranking cell
    assert [(c["member"], c["inventory"]) for c in cells] == [("L1", "angle")]
    cell = cells[0]
    assert cell["ordinals"] == [0, 1]
    assert cell["verdict"].tolist() == [0, 1] + [-1] * 18
    assert torch.allclose(cell["action"][:2, 0], torch.tensor([-70 / 480, -80 / 480]))
    assert torch.equal(cell["action"][2:], torch.zeros(18, 5))
    assert torch.equal(cell["z0"], anchors["L1"])


def test_excursion_flags_a_late_rise_of_one_request_against_its_own_reference_window():
    requests = [f"r{i % 3}" for i in range(900)]
    local = torch.full((900,), 0.1, dtype=torch.float64)
    norms = torch.ones(900, dtype=torch.float64)
    flat = rt.excursion(local, norms, requests)
    assert not flat["excursion"] and flat["first_spike_update"] is None
    late = local.clone()
    late[[i for i in range(890, 900) if requests[i] == "r1"]] = 0.6
    norms[880] = 5e3
    risen = rt.excursion(late, norms, requests)
    assert risen["excursion"] and risen["requests_elevated"] == ["r1"]
    assert risen["first_spike_update"] == 880 and risen["first_spike_request"] == requests[880]
    # a rise inside [0.6 n, 0.95 n) moves the reference too, so the end state is not an excursion
    early = local.clone()
    early[600:] = 0.6
    assert not rt.excursion(early, torch.ones(900, dtype=torch.float64), requests)["excursion"]
