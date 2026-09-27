from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import run_decision_chain_attribution as chain
from world_model.planning.task_objective import TaskObjective
from world_model.training.cnn_hybrid import PAIRS, CNNHybridPredictor

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="decision rollouts run on cuda")
VOCABULARY = ("bird:0000", "block:0000", "pig:0000")


class PositionEcho(CNNHybridPredictor):
    """Adds the pair's delta to the input carrier, so the output names the input position."""

    def carrier(self, z, action, pair):
        return z + pair.delta


class ShortensNearEnd(torch.nn.Module):
    """Chooses (15, continuous) while more than 20 positions remain, then (1, continuous)."""

    def __init__(self):
        super().__init__()
        self.visited = []

    def forward(self, z, action, remaining):
        self.visited.extend(z[:, 0].tolist())
        logits = torch.full((len(z), len(PAIRS)), -1.0, device=z.device)
        logits[remaining > 20, 6] = 1.0
        logits[remaining <= 20, 0] = 1.0
        return logits


def true_carriers(count):
    positions = torch.arange(chain.ENDPOINT + 1, dtype=torch.float32)
    return positions[None, :, None].expand(count, -1, 4).clone()


@cuda
@pytest.mark.parametrize("pair_index", range(len(PAIRS)))
def test_teacher_forced_fixed_pair_predicts_from_the_true_carrier_delta_before_the_endpoint(pair_index):
    actions = torch.zeros(2, 5, device="cuda")
    z_end, timing = chain.teacher_forced(PositionEcho().cuda(), None, true_carriers(2), actions,
                                         {"kind": "fixed", "pair_index": pair_index})
    assert z_end[:, 0].tolist() == [float(chain.ENDPOINT)] * 2
    assert timing["transition_calls"] == 2


@cuda
def test_teacher_forced_controller_walks_true_carriers_and_predicts_only_the_final_step():
    controller = ShortensNearEnd()
    actions = torch.zeros(1, 5, device="cuda")
    z_end, timing = chain.teacher_forced(PositionEcho().cuda(), controller, true_carriers(1), actions,
                                         {"kind": "joint"})
    assert controller.visited == [*range(0, 211, 15), *range(211, 225)]
    assert z_end[0, 0].item() == chain.ENDPOINT
    assert timing["transition_calls"] == 1


def carrier(pig_presence, pig_center, blocks=1.0):
    z = torch.zeros(2 + 13 * len(VOCABULARY))
    z[2 + 13 * 1] = blocks
    base = 2 + 13 * 2
    z[base] = pig_presence
    z[base + 5], z[base + 6] = pig_center
    return z


def test_tie_free_cost_orders_candidates_the_clamped_count_ties():
    costs = chain.Costs(TaskObjective.from_vocabulary(VOCABULARY))
    start = carrier(0.9, (0.5, 0.5))
    high = costs.row(0, carrier(1.6, (0.5, 0.5)), start)
    higher = costs.row(1, carrier(1.9, (0.5, 0.5)), start)
    assert high["count"] == higher["count"] == 1001.0
    assert high["tie_free"] < higher["tie_free"]
    moved = costs.row(2, carrier(1.6, (0.8, 0.9)), start)
    assert moved["pig_displacement"] == pytest.approx(0.5)
    assert moved["tie_free"] == pytest.approx(1.6 - chain.DISPLACEMENT_WEIGHT * 0.5)


def test_nonfinite_endpoint_is_a_typed_exclusion_under_both_costs():
    costs = chain.Costs(TaskObjective.from_vocabulary(VOCABULARY))
    row = costs.row(3, carrier(float("nan"), (0.5, 0.5)), carrier(1.0, (0.5, 0.5)))
    assert row["excluded"] and row["count"] is None and row["tie_free"] is None


def identity_metadata():
    eye = np.eye(4).reshape(-1).tolist()
    # ndc [-1, 1] -> pixels [0, 100] with y pointing down
    ndc = [50.0, 0.0, 50.0, 0.0, -50.0, 50.0, 0.0, 0.0, 1.0]
    return {"world_to_observation_transform": {"camera_to_clip_matrix": eye, "world_to_camera_matrix": eye,
                                               "ndc_to_observation_matrix": ndc},
            "viewport": {"width_pixels": 100, "height_pixels": 100}}


def entity(identity, lifecycle, position):
    return {"entity_id": f"runtime:{identity}", "scenario_object_id": identity, "lifecycle": lifecycle,
            "body_present": True, "body": {"position": position}}


def test_engine_projection_marks_removed_pigs_absent_and_skips_scenery():
    sample = {"entities": [entity("bird:0000", "active", (0.0, 0.0)),
                           entity("block:0000", "active", (0.5, 0.5)),
                           entity("pig:0000", "destroyed", (0.2, 0.2)),
                           entity("world:ground_extension:0000:0000", "active", (0.0, -1.0))]}
    parsed = chain.engine_parsed(sample, identity_metadata(), VOCABULARY)
    assert parsed["presence"].tolist() == [1.0, 1.0, 0.0]
    assert parsed["centers"][1].tolist() == pytest.approx([0.75, 0.25])
    assert parsed["kinds"].argmax(1).tolist() == [0, 2, 1]  # bird, block, pig


def test_spatial_slot_parser_feeds_the_frozen_temporal_carrier_adapter():
    from world_model.data.deployment_temporal import (AgentObservation, TemporalObservationContext,
                                                      TemporalVisualCarrierAdapter)
    from world_model.training.cohort_v2_visual_parser import CohortV2VisualParserConfig
    from world_model.training.spatial_slot_parser import SpatialSlotParser
    config = CohortV2VisualParserConfig(image_height=48, image_width=64, hidden_dim=32, device="cpu")
    model = SpatialSlotParser(config, VOCABULARY).eval()
    names = ("object_presence", "contact", "supports", "steady-state", "structure-unstable")
    adapter = TemporalVisualCarrierAdapter(model, parser_checkpoint_identity="test",
                                           temperatures={k: 1.0 for k in names},
                                           thresholds={k: 0.5 for k in names},
                                           latent_dim=2 + 13 * len(VOCABULARY), max_entities=len(VOCABULARY))
    stream = BytesIO()
    Image.fromarray(np.full((480, 640, 3), 128, dtype=np.uint8)).save(stream, format="PNG")
    observation = AgentObservation("frame", 30000, 12.0, stream.getvalue(), "agent")
    parsed = adapter.parse_batch((observation,))[0]
    assert parsed["centers"].min() >= 0 and parsed["centers"].max() <= 1
    tensor = adapter.build_from_parsed(TemporalObservationContext(None, observation), parsed, None).tensor
    assert tensor.shape == (2 + 13 * len(VOCABULARY),)
    assert torch.isfinite(tensor).all()
