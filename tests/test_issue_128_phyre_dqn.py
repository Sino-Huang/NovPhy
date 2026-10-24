"""#128 A1 DQN-px / A2 DQN-car: CPU checks of the vendored PHYRE DQN code and the NovPhy adapter."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from world_model.baselines.phyre_dqn import novphy as nv

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model/baselines/phyre_dqn/vendor"
CLONES = {"https://github.com/facebookresearch/phyre": Path("/mnt/array/sukaih/novphy-baselines-src/phyre@d0765dd"),
          "https://github.com/pytorch/vision": Path("/mnt/array/sukaih/novphy-baselines-src/torchvision@v0.4.0")}


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_vendor_hashes_and_edits():
    doc = json.loads((VENDOR / "VENDOR.json").read_text())
    assert doc["schema"] == "issue_127_vendor_v1" and doc["commit"] == "d0765dd2f6d6cf41b933a3a57173c63d21fbfa6a"
    on_disk = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.name != "VENDOR.json"}
    assert on_disk == set(doc["files"])
    for rel, entry in doc["files"].items():
        assert sha(VENDOR / rel) == entry["vendored_sha256"], rel
        if entry["upstream_path"] and not entry["edits"]:
            assert entry["upstream_sha256"] == entry["vendored_sha256"], rel
        clone = CLONES[entry.get("upstream_repo", doc["upstream"])]
        if entry["upstream_path"] and clone.is_dir():
            src = clone / entry["upstream_path"]
            if not src.is_file():  # the torchvision files were fetched flat
                src = clone / Path(entry["upstream_path"]).name
            assert sha(src) == entry["upstream_sha256"], rel


def test_vendored_edits_are_only_the_listed_import_lines():
    doc = json.loads((VENDOR / "VENDOR.json").read_text())
    clone = CLONES[doc["upstream"]]
    if not clone.is_dir():
        pytest.skip("upstream clone absent")
    for rel in ("nets.py", "neural_agent.py"):
        upstream = (clone / doc["files"][rel]["upstream_path"]).read_text().splitlines()
        vendored = (VENDOR / rel).read_text().splitlines()
        assert len(upstream) == len(vendored)
        changed = [f"{a} -> {b}" for a, b in zip(upstream, vendored) if a != b]
        assert [c.split(" (")[0] for c in doc["files"][rel]["edits"]] == changed


def test_a1_rgb_shapes_and_film_before_layer4():
    torch.manual_seed(0)
    model = nv.A1Net("rgb").eval()
    assert model.stem[0].in_channels == 3 and "embed_weights" not in dict(model.named_buffers())
    film = model._action_network
    assert model.action_networks[:3] == [None, None, None] and model.action_networks[3] is film
    assert film.net.layers[0].in_features == 4 and film.net.layers[0].out_features == 256
    assert film.net.output.out_features == 2 * model.stages[3][0].conv1.in_channels == 512
    frames = torch.randint(0, 256, (2, 3, 240, 320), dtype=torch.uint8)
    with torch.no_grad():
        features = model.preprocess(frames)["features"]
        assert features.shape == (2, 256, 15, 20)  # layer3 output: FiLM is applied before layer4
        actions = torch.rand(2, 4)
        assert model(frames, actions).shape == (2,)
        # one anchor against 16 candidates (the official eval path) equals the batched forward
        grid = torch.rand(16, 4)
        one = nv.score_actions(model, frames[0], grid, batch_size=5)
        batched = model(frames[0:1].expand(16, -1, -1, -1), grid)
    assert one.shape == (16,) and torch.allclose(one, batched, atol=1e-5)
    assert float(one.std()) > 0  # the action reaches the logit


def test_a1_rgb_input_scaling():
    model = nv.A1Net("rgb")
    x = torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)
    assert torch.equal(model._image_colors_to_onehot(x), torch.ones(1, 3, 4, 4))
    official = nv.A1Net("onehot7", action_size=3)
    assert official.stem[0].in_channels == 7 and official._image_colors_to_onehot(
        torch.zeros(1, 4, 4, dtype=torch.long)).shape == (1, 7, 4, 4)


def test_a2_parameter_counts_and_film_before_last_stage():
    a1 = nv.trainable_parameters(nv.A1Net("rgb"))
    matched = nv.build_model("A2", "primary")
    native = nv.build_model("A2", "native_size")
    p = nv.trainable_parameters(matched)
    assert abs(p / nv.U_PARAMETERS - 1) <= nv.MATCHED_TOLERANCE, p
    assert abs(nv.trainable_parameters(native) / a1 - 1) <= 0.05
    # every trainable module counted once (the FiLM module is held in a plain list and as an attribute)
    assert p == sum(t.numel() for t in {id(t): t for t in matched.parameters()}.values())
    assert matched.action_networks[3] is matched._action_network
    assert matched._action_network.net.output.out_features == 2 * matched.stages[3][0].conv1.in_channels
    assert matched.reason.in_features == matched.stages[3][-1].bn2.num_features
    carriers = torch.randn(5, 236)
    matched.eval()
    with torch.no_grad():
        assert matched.preprocess(carriers)["features"].shape == (5, 4 * nv.A2_MATCHED_BASE_WIDTH, 1, 1)
        assert matched(carriers, torch.rand(5, 4)).shape == (5,)
        assert nv.score_actions(matched, carriers[0], torch.rand(16, 4)).shape == (16,)


@pytest.mark.parametrize("batch", [64, 7])
def test_class_balanced_sampler(batch):
    labels = torch.tensor([True] * 30 + [False] * 970)
    sampler = nv.train_indices_sampler(labels, batch, np.random.RandomState(42))
    positives = [int(labels[next(sampler)].sum()) for _ in range(6)]
    if batch % 2 == 0:
        assert positives == [batch // 2] * 6
    else:  # the official sampler alternates the odd example between the classes
        assert positives == [batch // 2, batch - batch // 2] * 3


def test_trainset_one_example_per_shot_and_launch4():
    from scripts import external_baselines as eb

    action = {"drag_x": -120, "drag_y": 40, "release_time_ms": 1000, "tap_time_ms": 1500}
    assert torch.allclose(eb.launch4(action), torch.tensor([-0.25, 40 / 480, 1.0, 1.5]))
    members = ["b", "a", "b", "c"]
    anchors = {m: torch.full((3, 2, 2), i, dtype=torch.uint8) for i, m in enumerate("abc")}
    launch = torch.stack([eb.launch4(action)] * 4)
    data = nv.build_trainset(members, launch, [True, False, False, True], anchors)
    assert data.levels == ("a", "b", "c") and data.task_indices.tolist() == [1, 0, 1, 2]
    assert torch.equal(data.observations[data.task_indices][0], anchors["b"]) and len(data.is_solved) == 4


def test_train_step_runs_scheduler_first():
    torch.manual_seed(0)
    model = nv.A2Net(8)
    data = nv.build_trainset(["a", "b"] * 8, torch.rand(16, 4), [True, False] * 8,
                             {"a": torch.randn(236), "b": torch.randn(236)})
    optimizer, scheduler = nv.make_optimizer(model, updates=4)
    lrs = []
    sampler = nv.train_indices_sampler(data.is_solved, 4, np.random.RandomState(42))
    for _ in range(4):
        loss = nv.train_step(model, optimizer, scheduler, data, next(sampler))
        lrs.append(optimizer.param_groups[0]["lr"])
        assert torch.isfinite(loss)
    expected = [3e-4 * (1 + np.cos(np.pi * k / 4)) / 2 for k in range(1, 5)]
    assert np.allclose(lrs, expected, atol=1e-12)
