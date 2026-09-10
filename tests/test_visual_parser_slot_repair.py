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

    def test_negative_image_projection_keeps_image_gradients(self):
        from world_model.training.cohort_v2_visual_parser import NormalizedSlotVisualPredicateParser
        torch.manual_seed(71)
        config=CohortV2VisualParserConfig(hidden_dim=8,image_height=8,image_width=8,device="cpu")
        model=NormalizedSlotVisualPredicateParser(config,("pig:1","block:1"))
        with torch.no_grad():
            model.backbone[0].bias.copy_(torch.linspace(-20,-19,8))
        images=torch.randint(0,256,(4,3,8,8),dtype=torch.uint8)
        logits=model(images)["presence_logits"]
        logits.sum().backward()
        self.assertGreater(float(model.backbone[0].weight.grad.abs().max()),0)
        self.assertGreater(float(logits.detach().std(0).max()),1e-5)
        legacy=SlotConditionedVisualPredicateParser(config,("pig:1","block:1"))
        with self.assertRaises(RuntimeError):legacy.load_state_dict(model.state_dict(),strict=True)

    def test_full_resolution_image_path_learns_independent_presence(self):
        from world_model.training.cohort_v2_visual_parser import NormalizedSlotVisualPredicateParser
        torch.manual_seed(71)
        config=CohortV2VisualParserConfig(hidden_dim=16,image_height=64,image_width=96,device="cpu")
        model=NormalizedSlotVisualPredicateParser(config,("pig:1","block:1"))
        images=torch.full((4,3,64,96),160,dtype=torch.uint8)
        targets=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        images[[2,3],:,24:32,60:68]=220
        images[[1,3],:,40:48,72:80]=30
        model.fit_input_standardization(images)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        for _ in range(300):
            optimizer.zero_grad()
            logits=model(images)["presence_logits"]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        self.assertLess(float(loss.detach()),0.15)
        self.assertTrue(torch.equal(logits.detach()>0,targets.bool()))
        restored=NormalizedSlotVisualPredicateParser(config,model.object_vocabulary)
        restored.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():self.assertTrue(torch.equal(model(images)["presence_logits"],restored(images)["presence_logits"]))

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
