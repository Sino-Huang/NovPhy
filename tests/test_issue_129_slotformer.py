"""Issue #129 (B1 SlotFormer-E99): CPU checks of the adapter's uncertain plumbing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from world_model.baselines.slotformer import novphy as sf
from world_model.planning import scoring_harness as harness

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model/baselines/slotformer/vendor"
CLONE = Path("/mnt/array/sukaih/novphy-baselines-src/SlotFormer@10a6d9b")


def _carriers(b, t, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, sf.CARRIER_DIM, generator=g)


def _launch(b, seed=1):
    g = torch.Generator().manual_seed(seed)
    launch = torch.rand(b, sf.LAUNCH_DIM, generator=g)
    launch[:, -1] = 1.0
    return launch


def test_layout_matches_scoring_harness():
    assert sf.CARRIER_HEADER == harness.CARRIER_HEADER
    assert sf.SLOT_WIDTH == harness.SLOT_WIDTH
    assert sf.CARRIER_HEADER + sf.CARRIER_SLOTS * sf.SLOT_WIDTH == sf.CARRIER_DIM


def test_pack_unpack_round_trip_and_launch_token():
    carriers, launch = _carriers(3, 4), _launch(3)
    slots = sf.pack_slots(carriers, launch)
    assert slots.shape == (3, 4, sf.TOKENS, sf.SLOT_WIDTH)
    back = sf.unpack_carriers(slots)
    # every carrier slot column survives; the header becomes ROLLED_HEADER
    assert torch.equal(back[..., sf.CARRIER_HEADER:], carriers[..., sf.CARRIER_HEADER:])
    assert torch.equal(back[..., :sf.CARRIER_HEADER], torch.tensor(sf.ROLLED_HEADER).expand(3, 4, 2))
    # slot k feature f sits at carrier column 2 + 13 k + f (EndpointCosts' indexing)
    k, f = 7, harness.PRESENCE_FEATURE
    assert torch.equal(slots[..., k, f], carriers[..., harness.CARRIER_HEADER + harness.SLOT_WIDTH * k + f])
    # the launch token: [launch5, 0 x 8] in the last slot, identical at every frame
    token = torch.cat([launch, torch.zeros(3, sf.SLOT_WIDTH - sf.LAUNCH_DIM)], dim=-1)
    assert torch.equal(slots[:, :, sf.LAUNCH_TOKEN], token[:, None].expand(3, 4, -1))


def test_positions_and_rollout_lengths():
    assert sf.positions(15) == list(range(0, 226, 15)) and sf.rollout_steps(15) == 15
    assert sf.positions(5)[-1] == 225 and sf.rollout_steps(5) == 45
    with pytest.raises(ValueError):
        sf.positions(7)
    torch.manual_seed(0)
    for stride, steps in ((15, 15), (5, 45)):
        model = sf.SlotFormerE99({"d_model": 16, "num_layers": 1, "num_heads": 2, "ffn_dim": 32}, stride=stride).eval()
        assert model.rollout_len == steps
        with torch.no_grad():
            out = model.predict_carriers(_carriers(2, 1)[:, 0], _launch(2))
            end = model.predict_endpoint(_carriers(2, 1)[:, 0], _launch(2))
        assert out.shape == (2, steps, sf.CARRIER_DIM)
        assert torch.equal(end, out[:, -1])


def test_launch_token_enters_at_t0_only_as_burn_in():
    torch.manual_seed(0)
    model = sf.SlotFormerE99({"d_model": 16, "num_layers": 1, "num_heads": 2, "ffn_dim": 32}).eval()
    seen = []
    model.rollouter.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    anchor, launch = _carriers(2, 1)[:, 0], _launch(2)
    with torch.no_grad():
        model.predict_slots(anchor, launch)
    burn_in = seen[0]
    assert burn_in.shape == (2, 1, sf.TOKENS, sf.SLOT_WIDTH)  # burn-in 1: the anchor frame only
    assert torch.equal(burn_in[:, 0, :sf.CARRIER_SLOTS], sf.carrier_slots(anchor))
    assert torch.equal(burn_in[:, 0, sf.LAUNCH_TOKEN, :sf.LAUNCH_DIM], launch)
    # the launch changes the rollout (the token is not ignored)
    with torch.no_grad():
        a = model.predict_endpoint(anchor, launch)
        b = model.predict_endpoint(anchor, launch.flip(0))
    assert not torch.allclose(a, b)


def test_sequence_plan_matches_fix_video_len_and_vid_len():
    read, vid_len = sf.sequence_plan([285, 225, 224, 150, 138], 15)
    grid = torch.arange(0, 226, 15)
    assert torch.equal(read[0], grid) and torch.equal(read[1], grid)
    assert vid_len.tolist() == [16, 16, 15, 11, 10]
    # past the last real strided frame: repeat it (PHYRE fix_video_len)
    assert read[2].tolist() == list(range(0, 211, 15)) + [210]
    assert read[3].tolist() == list(range(0, 151, 15)) + [150] * 5
    assert read[4].tolist() == list(range(0, 136, 15)) + [135] * 6
    read5, vid5 = sf.sequence_plan([138], 5)
    assert vid5.tolist() == [28] and read5[0, 27:].unique().tolist() == [135]


def test_training_batch_reads_the_planned_rows_and_loss_ignores_padding():
    carriers = _carriers(3, 286, seed=3)
    calls = []

    def carrier_rows(shots, positions):
        calls.append((torch.as_tensor(shots), torch.as_tensor(positions)))
        return carriers[torch.as_tensor(shots), torch.as_tensor(positions)]

    launch = _launch(3)
    batch = sf.training_batch(carrier_rows, [2, 0, 1], [285, 150, 230], launch, 15)
    assert batch["vid_len"].tolist() == [16, 11, 16]
    assert torch.equal(batch["slots"][0, :, :sf.CARRIER_SLOTS], sf.carrier_slots(carriers[2, ::15][:16]))
    assert torch.equal(batch["slots"][1, 11:, :sf.CARRIER_SLOTS],
                       sf.carrier_slots(carriers[0, 150]).expand(5, -1, -1))
    torch.manual_seed(0)
    model = sf.SlotFormerE99({"d_model": 16, "num_layers": 1, "num_heads": 2, "ffn_dim": 32}).train()
    torch.manual_seed(1)
    loss, out = model.total_loss(dict(batch))
    assert set(k for k in out if k.endswith("_loss")) == {"slot_recon_loss"}
    changed = {k: v.clone() for k, v in batch.items()}
    changed["slots"][1, 11:] += 5.0  # padded frames of the short shot: masked by vid_len
    torch.manual_seed(1)
    loss_changed, _ = model.total_loss(changed)
    assert torch.allclose(loss, loss_changed)


def test_parameter_counts():
    native, matched = sf.SlotFormerE99("native"), sf.SlotFormerE99("matched")
    assert sf.trainable_parameters(native) == 6_325_005
    assert sf.trainable_parameters(matched) == 1_793_445
    assert abs(sf.trainable_parameters(matched) / sf.U_PARAMETERS - 1) <= sf.MATCHED_TOLERANCE
    # every trainable parameter belongs to the official rollouter (no decoder, no extra head); the only frozen
    # tensor is the official fixed sine temporal P.E. (an nn.Parameter with requires_grad False)
    for model in (native, matched):
        assert all(n.startswith("rollouter.") for n, p in model.named_parameters() if p.requires_grad)
        assert [n for n, p in model.named_parameters() if not p.requires_grad] == ["rollouter.enc_t_pe"]


def test_optimizer_schedule():
    model = sf.SlotFormerE99({"d_model": 16, "num_layers": 1, "num_heads": 2, "ffn_dim": 32})
    optimizer, scheduler = sf.build_optimizer(model, 1000)
    assert isinstance(optimizer, torch.optim.Adam) and optimizer.defaults["weight_decay"] == 0.
    assert optimizer.param_groups[0]["lr"] == sf.LR / 100.
    for _ in range(50):  # 5 % warm-up of 1000 steps
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(sf.LR)
    for _ in range(950):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(sf.LR / 100., rel=1e-3)


def _sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_vendor_record():
    record = json.loads((VENDOR / "VENDOR.json").read_text())
    assert record["schema"] == "issue_127_vendor_v1"
    assert record["commit"] == "10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa"
    on_disk = {p.name for p in VENDOR.iterdir() if p.is_file() and p.name != "VENDOR.json"}
    assert on_disk == set(record["files"])
    for rel, entry in record["files"].items():
        assert _sha(VENDOR / rel) == entry["vendored_sha256"], rel
        if entry["upstream_path"]:
            assert (entry["upstream_sha256"] == entry["vendored_sha256"]) == (not entry["edits"]), rel
    edited = sorted(rel for rel, e in record["files"].items() if e["edits"] and e["upstream_path"])
    assert edited == ["phyre_readout.py", "slotformer.py"]
    if CLONE.exists():
        for rel, entry in record["files"].items():
            if entry["upstream_path"] and "upstream" not in entry:
                assert _sha(CLONE / entry["upstream_path"]) == entry["upstream_sha256"], rel


# ---------------------------------------------------------------------------------------------------------------
# native success head (vendored PHYRE readout)
# ---------------------------------------------------------------------------------------------------------------

def test_head_input_and_forward():
    anchors, ends, launch = _carriers(5, 1)[:, 0], _carriers(5, 1, seed=4)[:, 0], _launch(5)
    x = sf.head_input(anchors, ends, launch)
    assert x.shape == (5, 2, sf.TOKENS, sf.SLOT_WIDTH)
    assert torch.equal(x[:, 0, :sf.CARRIER_SLOTS], sf.carrier_slots(anchors))
    assert torch.equal(x[:, 1, :sf.CARRIER_SLOTS], sf.carrier_slots(ends))
    assert torch.equal(x[:, 1, sf.LAUNCH_TOKEN, :sf.LAUNCH_DIM], launch)
    torch.manual_seed(0)
    head = sf.build_head().eval()
    assert head.sel_slots == [0, 1] and head.T == 2
    with torch.no_grad():
        logits = head({"slots": x})["logits"]
    assert logits.shape == (5,)
    head.train()
    loss, out = sf.head_total_loss(head, {"slots": x, "label": torch.tensor([1, 0, 0, 1, 0])})
    assert set(k for k in out if k.endswith("_loss")) == {"vqa_loss"} and torch.isfinite(loss)


# ---------------------------------------------------------------------------------------------------------------
# runner (scripts/run_issue_129_slotformer.py)
# ---------------------------------------------------------------------------------------------------------------

TINY = {"d_model": 16, "num_layers": 1, "num_heads": 2, "ffn_dim": 32}


def _runner():
    from scripts import run_issue_129_slotformer as runner
    return runner


def _tiny_spec(steps=12, stride=15):
    return {"name": "tiny", "size": TINY, "stride": stride, "lr": 1e-3, "batch": 8, "epochs": None, "steps": steps,
            "warmup_pct": 0.05}


def _tiny_data(n=40, stride=15, seed=5):
    g = torch.Generator().manual_seed(seed)
    t = len(sf.positions(stride))
    return {"slots": torch.randn(n, t, sf.TOKENS, sf.SLOT_WIDTH, generator=g),
            "vid_len": torch.tensor([t if i % 3 else 9 for i in range(n)])}


def test_schedule_constants():
    runner = _runner()
    assert runner.steps_per_epoch(13_919) == 217
    assert runner.config_spec("C1-s15", 13_919)["steps"] == 10_850
    assert runner.config_spec("C3-s15-long", 13_919)["steps"] == 54_250
    m = runner.arm_spec("B1-m", "C2-s5", 13_919)
    assert (m["size"], m["stride"], m["steps"], m["batch"], m["lr"]) == ("matched", 5, 9000, 64, 2e-4)
    assert runner.arm_spec("B1", "C2-s5", 13_919)["steps"] == 10_850
    assert runner.head_steps(13_919) == 54 * 50
    assert sorted(n for q in runner.SELECTION_QUEUES.values() for n in q) == sorted(runner.CONFIGS)
    out = Path("/tmp/x")
    assert runner.model_dir(out, "B1", 20260908, "C2-s5") == runner.selection_dir(out, "C2-s5")
    assert runner.model_dir(out, "B1", 20260909, "C2-s5") == out / "models/B1/seed-20260909"


def test_epoch_batches_are_permutations_and_deterministic():
    runner = _runner()
    a, b = runner.EpochBatches(7, 20, 6), runner.EpochBatches(7, 20, 6)
    assert a.spe == 3  # drop_last
    epoch0 = torch.cat([a(i) for i in range(3)])
    assert len(set(epoch0.tolist())) == 18
    assert torch.equal(a(4), b(4)) and torch.equal(a(1), b(1))  # random access == sequential
    assert not torch.equal(a(0), a(3))  # a new permutation per epoch


def test_train_run_resume_is_bitwise_identical(tmp_path, monkeypatch):
    runner = _runner()
    data, spec = _tiny_data(), _tiny_spec()
    full = runner.train_run(tmp_path / "full", spec, 3, data, "cpu")
    assert full["complete"]

    class Stopping(runner.EpochBatches):
        def __call__(self, b):
            if b == 4:
                eb.STOP_REQUESTED.append(15)
            return super().__call__(b)

    from scripts import external_baselines as eb
    monkeypatch.setattr(runner, "EpochBatches", Stopping)
    with pytest.raises(eb.TrainingPaused):
        runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    eb.STOP_REQUESTED.clear()
    assert (tmp_path / "split/resume.pt").is_file()
    monkeypatch.setattr(runner, "EpochBatches", runner.EpochBatches.__mro__[1])
    split = runner.train_run(tmp_path / "split", spec, 3, data, "cpu")
    assert split["complete"] and not (tmp_path / "split/resume.pt").exists()
    a = torch.load(tmp_path / "full/model.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "split/model.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    model, payload = runner.load_trained(tmp_path / "split/model.pt", "cpu")
    assert payload["complete"] and not model.training


def test_nonfinite_loss_is_a_typed_terminal_failure(tmp_path):
    runner = _runner()
    data = _tiny_data()
    data["slots"][:, 3] = float("nan")
    report = runner.train_run(tmp_path, _tiny_spec(), 3, data, "cpu")
    assert not report["complete"] and report["typed_failure"]["typed_failure"] == "nonfinite_loss"
    model, payload = runner.load_trained(tmp_path / "model.pt", "cpu")
    assert model is None and payload["state_dict"] is None
    again = runner.train_run(tmp_path, _tiny_spec(), 3, _tiny_data(), "cpu")  # never retried
    assert again == report


def test_train_head_and_lpt():
    runner = _runner()
    n = 300  # more than one head batch (256)
    anchors, ends, launch = _carriers(n, 1)[:, 0], _carriers(n, 1, seed=8)[:, 0], _launch(n)
    labels = torch.tensor([i % 4 == 0 for i in range(n)])
    head, report = runner.train_head(anchors, ends, launch, labels, 3, "cpu", steps=3)
    assert report["typed_failure"] is None and report["positives"] == 75 and not head.training
    assert runner.head_logits(head, anchors, ends, launch).shape == (n,)
    queues, loads = runner.lpt([("a", 5.0), ("b", 4.0), ("c", 3.0), ("d", 2.0), ("e", 1.0)], bins=2)
    assert queues == {0: ["a", "d", "e"], 1: ["b", "c"]} and loads == [8.0, 7.0]


def test_ledger_report_mapping_and_readout_items(tmp_path):
    runner = _runner()
    plan = {"selected": "C1-s15", "arms": {arm: {"spec": runner.arm_spec(arm, "C1-s15", 13_919)} for arm in runner.ARMS}}
    report_of = runner.training_report_of(tmp_path, plan)
    assert report_of("train-C1-s15-seed20260908") == runner.selection_dir(tmp_path, "C1-s15") / "report.json"
    assert report_of("train-C1-s15-seed20260910") == tmp_path / "models/B1/seed-20260910/report.json"
    assert report_of("train-B1-m@C1-s15-seed20260909") == tmp_path / "models/B1-m/seed-20260909/report.json"
    assert report_of("train-C3-s15-long-seed20260908") == runner.selection_dir(tmp_path, "C3-s15-long") / "report.json"
    assert report_of("readouts-B1-seed20260908") is None
    assert len(runner.readout_items()) == 9 and ("U+head", 20260910) in runner.readout_items()
    remaining = runner.remaining_items(tmp_path, "C1-s15", 13_919)
    assert [(a, s) for a, s, _, _ in remaining] == [("B1", 20260909), ("B1", 20260910), ("B1-m", 20260908),
                                                   ("B1-m", 20260909), ("B1-m", 20260910)]


def test_position_rows_read_the_rollout_at_each_time():
    runner = _runner()
    from scripts import external_baselines as eb
    times = eb.POSITION_TIMES
    pred = torch.zeros(1, sf.CARRIER_DIM)
    pred[0, 2 + 5], pred[0, 2 + 6] = 0.5, 0.25  # slot 0 centre
    at = {t: pred * (1 + i) for i, t in enumerate(times)}
    presence = torch.zeros(len(times), 18, dtype=torch.bool)
    presence[:, 0] = True
    cache = {"entries": {"k": {"presence": presence, "centers": torch.zeros(len(times), 18, 2)}}}
    rows = runner._position_rows(at, [("m", {"key": "k"})], cache, times)
    for i, t in enumerate(times):
        assert rows["k"][str(t)]["pred"] == pytest.approx(((0.5 * (1 + i)) ** 2 + (0.25 * (1 + i)) ** 2) / 2)


def test_bench_spec_windows_and_scoring_path(tmp_path):
    """Exploratory CLEVRER bench: B1 at 9 000 x 64, rollout to 120 frames, null launch; real #117 item-5 clips."""
    runner = _runner()
    from scripts import external_baselines as eb
    from scripts import run_request_interference_benchmarks as rib
    spec = runner.bench_spec("C1-s15", "clevrer")
    assert (spec["steps"], spec["batch"], spec["stride"], spec["rollout_len"], spec["size"]) == (9000, 64, 15, 8, "native")
    assert runner.bench_spec("C2-s5", "physion")["rollout_len"] == 24
    data = eb.rc.BenchData("clevrer")
    windows = runner.bench_windows(data, 15)
    assert windows["slots"].shape == (12 * 8, 9, sf.TOKENS, sf.SLOT_WIDTH)  # 128-frame clips: starts 0..7
    assert torch.equal(windows["slots"][:, :, sf.LAUNCH_TOKEN], torch.zeros(96, 9, sf.SLOT_WIDTH))
    scene, start = windows["units"][9]
    assert torch.equal(windows["slots"][9, 2, :sf.CARRIER_SLOTS], sf.carrier_slots(data.clips[scene][start + 30]))
    tiny = {**spec, "size": TINY, "steps": 2, "batch": 8}
    report = runner.train_run(tmp_path, tiny, 3, windows, "cpu")
    model, _ = runner.load_trained(tmp_path / "model.pt", "cpu")
    assert report["complete"] and model.rollout_len == 8
    pred = runner.rollout_carriers(model, data.clips[scene][0][None], torch.zeros(1, sf.LAUNCH_DIM))
    assert pred.shape == (1, 8, sf.CARRIER_DIM)
    errors = rib.unit_errors(pred[0, 120 // 15 - 1], data.clips[scene][120])
    assert errors is not None and errors[rib.METRIC] is not None
