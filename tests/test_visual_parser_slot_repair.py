import unittest
from pathlib import Path
import tempfile
import torch
from world_model.training.cohort_v2_visual_parser import (
    CohortV2VisualParserConfig, CohortV2VisualPredicateParser,
    SlotConditionedVisualPredicateParser,
    save_visual_parser_checkpoint, CohortV2VisualParserError,
)


def fit_opposite_presence(parser_type):
    torch.manual_seed(71)
    model=parser_type(CohortV2VisualParserConfig(hidden_dim=8,device="cpu"),("pig:1","block:1"))
    features=torch.zeros(2,8);features[0,0]=1;features[1,0]=-1
    targets=torch.tensor([[1.,0.],[0.,1.]])
    optimizer=torch.optim.Adam(model.parameters(),lr=0.05)
    for _ in range(400):
        optimizer.zero_grad()
        logits=model.presence_head(model.slot_features(features)).squeeze(-1)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)
        loss.backward();optimizer.step()
    with torch.no_grad():
        predictions=model.presence_head(model.slot_features(features)).squeeze(-1).sigmoid()
    return float(loss.detach()),predictions,targets


class SlotRepairTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_legacy_presence_cannot_represent_independent_removal(self):
        loss,predicted,target=fit_opposite_presence(CohortV2VisualPredicateParser)
        self.assertGreater(loss,0.68)
        self.assertFalse(torch.equal(predicted>0.5,target.bool()))

    def test_repaired_presence_learns_independent_removal(self):
        loss,predicted,target=fit_opposite_presence(SlotConditionedVisualPredicateParser)
        self.assertLess(loss,0.05)
        self.assertTrue(torch.equal(predicted>0.5,target.bool()))

    def test_repair_cannot_silently_load_into_old_architecture(self):
        config=CohortV2VisualParserConfig(hidden_dim=8,device="cpu")
        original=CohortV2VisualPredicateParser(config,("pig:1","block:1"))
        repaired=SlotConditionedVisualPredicateParser(config,("pig:1","block:1"))
        with self.assertRaises(RuntimeError):original.load_state_dict(repaired.state_dict(),strict=True)

    def test_v2_cannot_be_mislabeled_as_v1_checkpoint(self):
        model=SlotConditionedVisualPredicateParser(CohortV2VisualParserConfig(hidden_dim=8,device="cpu"),("pig:1","block:1"))
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"checkpoint"
            with self.assertRaises(CohortV2VisualParserError):
                save_visual_parser_checkpoint(target,model,{}, {},1.0,role_data=(),readers=(),
                    model_selection=(),calibration_metrics={},implementation_revision="test")
            self.assertFalse(target.exists())


if __name__=="__main__":unittest.main()
