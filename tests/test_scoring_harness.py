import math

import pytest
import torch

from world_model.model import Abstraction, PredictionPair
from world_model.planning import scoring_harness as sh
from world_model.planning.task_objective import TaskObjective
from world_model.training import jepa_slot_encoder as js
from world_model.training.cnn_hybrid import CNNHybridPredictor
from world_model.training.cohort_v2_visual_parser import CohortV2VisualParserConfig
from world_model.training.spatial_slot_parser import SpatialSlotParser

VOCABULARY = tuple([f"bird:{i:04d}" for i in range(3)] + [f"block:{i:04d}" for i in range(5)] + ["pig:0000"]
                   + [f"platform:{i:04d}" for i in range(6)] + ["slingshot:0000", "world:landscape:0000",
                                                                 "world:landscape:0001"])
PIG = VOCABULARY.index("pig:0000")


def _flags():
    return (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)


def _restore(flags):
    (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
     torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = flags


def _column(feature, slot=PIG):
    return sh.CARRIER_HEADER + sh.SLOT_WIDTH * slot + feature


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rollout_costs_are_identical_twice_and_after_an_autotuned_training_warm_up():
    """#100 G3: an autotuning training process must not change a later score of the same rollout."""
    saved = _flags()
    try:
        torch.manual_seed(0)
        parser = SpatialSlotParser(CohortV2VisualParserConfig(image_height=240, image_width=320,
                                                              hidden_dim=128), VOCABULARY).cuda().eval()
        predictor = CNNHybridPredictor().cuda().eval()
        generator = torch.Generator().manual_seed(1)
        images = torch.randint(0, 256, (16, 3, 240, 320), generator=generator).cuda()
        actions = torch.rand(8, 5, generator=generator).cuda()
        costs = sh.EndpointCosts(TaskObjective.from_vocabulary(VOCABULARY))
        pair = PredictionPair(1, Abstraction.CONTINUOUS)
        expected = js.expected_kind_indices(VOCABULARY)

        @torch.no_grad()
        def score():
            output = parser(images)
            none = torch.zeros(len(images), device="cuda")
            z0 = js.carrier_from_outputs(output, output, none, none, expected)[3]
            z = z0[None].expand(len(actions), -1).clone()
            for _ in range(225):  # recursive Delta = 1 to the decision endpoint
                z = predictor.carrier(z, actions, pair)
            return [costs.row(i, z[i].cpu(), z0.cpu()) for i in range(len(actions))]

        with sh.deterministic_scoring() as policy:
            first, second = score(), score()
            assert policy == sh.SCORING_POLICY
        # a training step under autotuning with TF32 on, in the same process
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        parser.train()
        for batch in (16, 4, 1):
            parser(torch.randint(0, 256, (batch, 3, 240, 320), device="cuda"))["centers"].sum().backward()
        parser.eval()
        warmed = _flags()
        with sh.deterministic_scoring():
            third = score()
        assert _flags() == warmed  # training after scoring keeps its own policy
        assert not any(row["excluded"] for row in first)
        assert first == second == third
    finally:
        _restore(saved)


def test_motion_values_are_held_to_twice_the_tolerance_over_their_frame_interval():
    """#100 G1: center noise divided by a few-ms terminal interval is not a parse difference."""
    reference = torch.zeros(3, 236)
    reference[:, sh.ELAPSED_COLUMN] = torch.tensor([0.0868, 0.004, 0.0])
    fresh = reference.clone()
    motion = _column(sh.MOTION_FEATURES[0])
    fresh[1, motion] = 0.4       # 2 x 8e-4 center noise / 0.004 s: within 2 x 1e-3 / 0.004 = 0.5
    fresh[2, motion] = 5.0       # no prior frame (elapsed 0): motion carries no parse information
    assert sh.carrier_replication(reference, fresh)["pass"]
    fresh[1, motion] = 0.6       # beyond 2 x tolerance / elapsed
    assert not sh.carrier_replication(reference, fresh)["pass"]
    fresh[1, motion] = 0.0
    fresh[0, _column(sh.CENTER_FEATURES[0])] = 2e-3  # a center is a non-motion value: plain tolerance
    result = sh.carrier_replication(reference, fresh)
    assert not result["pass"] and result["max_abs_nonmotion_delta"] == pytest.approx(2e-3)


def test_endpoint_replication_compares_parsed_quantities_not_the_weighted_count():
    """#99 G3: a 1.34e-4 pig-presence difference is 0.134 in the count cost."""
    costs = sh.EndpointCosts(TaskObjective.from_vocabulary(VOCABULARY))
    z0 = torch.zeros(236)
    z = torch.zeros(236)
    z[_column(sh.PRESENCE_FEATURE)] = 0.5
    z[_column(sh.CENTER_FEATURES[0])] = 0.3
    reference = [costs.row(0, z, z0), costs.row(1, torch.full((236,), math.nan), z0)]
    nudged = z.clone()
    nudged[_column(sh.PRESENCE_FEATURE)] += 1.34e-4
    fresh = [costs.row(0, nudged, z0), costs.row(1, torch.full((236,), math.nan), z0)]
    assert abs(fresh[0]["count"] - reference[0]["count"]) > 0.1
    assert fresh[0][sh.PRIMARY_COST] == pytest.approx(0.5 + 1.34e-4 - 0.1 * 0.3)
    assert sh.endpoint_replication(reference, fresh)["pass"]
    moved = z.clone()
    moved[_column(sh.CENTER_FEATURES[0])] += 2e-3  # displacement beyond tolerance
    assert not sh.endpoint_replication(reference, [costs.row(0, moved, z0), fresh[1]])["pass"]
    scored = [fresh[0], costs.row(1, z, z0)]  # excluded in one set, scored in the other
    assert not sh.endpoint_replication(reference, scored)["pass"]
