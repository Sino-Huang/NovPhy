import unittest

import torch

from world_model.training.cohort_v2_visual_parser import (
    CohortV2VisualParserConfig, NormalizedSlotVisualPredicateParser,
)


class CNNParserTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_cnn_learns_pixels_and_preserves_parser_interface(self):
        from world_model.training.cohort_v2_visual_parser import ConvSlotVisualPredicateParser
        config=CohortV2VisualParserConfig(image_height=64,image_width=96,hidden_dim=128)
        vocabulary=tuple(f"block:{i:04d}" for i in range(17))+("pig:0000",)
        model=ConvSlotVisualPredicateParser(config,vocabulary)
        images=torch.randint(0,256,(2,3,64,96),dtype=torch.uint8)
        output=model(images)
        self.assertEqual(output["presence_logits"].shape,(2,18))
        self.assertEqual(output["centers"].shape,(2,18,2))
        self.assertEqual(output["relation_logits"].shape,(2,18,18,2))
        self.assertEqual(output["macro_logits"].shape,(2,2))
        output["presence_logits"].sum().backward()
        conv=next(layer for layer in model.encoder if isinstance(layer,torch.nn.Conv2d))
        self.assertTrue(conv.weight.requires_grad)
        self.assertGreater(float(conv.weight.grad.abs().sum()),0)
        self.assertLess(sum(p.numel() for p in model.parameters()),500_000)
        restored=ConvSlotVisualPredicateParser(config,vocabulary)
        restored.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(output["presence_logits"],restored(images)["presence_logits"]))
        old=NormalizedSlotVisualPredicateParser(config,vocabulary)
        with self.assertRaises(RuntimeError):model.load_state_dict(old.state_dict(),strict=True)

    def test_cnn_learns_independent_object_presence(self):
        from world_model.training.cohort_v2_visual_parser import ConvSlotVisualPredicateParser
        config=CohortV2VisualParserConfig(image_height=16,image_width=24,hidden_dim=32)
        model=ConvSlotVisualPredicateParser(config,("pig:0000","block:0000"))
        images=torch.full((4,3,16,24),160,dtype=torch.uint8)
        images[[2,3],:,4:8,12:16]=220
        images[[1,3],:,10:14,18:22]=30
        target=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        for _ in range(300):
            optimizer.zero_grad()
            logits=model(images)["presence_logits"]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        self.assertLess(float(loss.detach()),0.15)
        self.assertTrue(torch.equal(logits.detach()>0,target.bool()))


if __name__=="__main__":unittest.main()
