from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import torch

from scripts import run_issue_70_parser_repair as repair
from scripts import run_issue_70_action_design as experiment
from scripts.issue_70_live_pilot import pilot_plan
from tests.test_issue_70_action_design import fixture


def plan_fixture():
    config=repair.CohortV2VisualParserConfig(seed=7002001,image_height=8,image_width=8,hidden_dim=8,epochs=2,device="cpu")
    predicates=("object_presence","contact","supports","steady-state","structure-unstable")
    return {"identity":repair.PARSER_ID,"carrier_identity":repair.CARRIER_ID,"config":asdict(config),
        "training_release_identity":"training-release","calibration_fits_parameters":False,
        "training_records":[{"scenario_lineage_identity":f"lineage-{i}","exposure_role":"training"} for i in range(3000)],
        "vocabulary":["pig:0000","block:0000"],"temperatures":{k:1.0 for k in predicates},
        "thresholds":{k:0.5 for k in predicates},"world_optimizer_examples":8_000_000}


class RepairTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_repaired_checkpoint_round_trip_has_distinct_carrier(self):
        plan=plan_fixture();config=repair.CohortV2VisualParserConfig(**plan["config"])
        model=repair.SlotConditionedVisualPredicateParser(config,tuple(plan["vocabulary"]))
        payload={"schema":"issue_70_repaired_parser_checkpoint_v2","identity":repair.PARSER_ID,
            "architecture":model.architecture_identity,"config":plan["config"],"vocabulary":plan["vocabulary"],
            "training_release_identity":plan["training_release_identity"],"epoch":2,"reports":[{},{}],"model_state":model.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repair.write(root/"plan.json",plan);repair.atomic_torch(root/"parser.pt",payload)
            adapter=repair.load_repaired_adapter(root,"cpu")
            self.assertEqual(adapter.identity,repair.CARRIER_ID)
            self.assertNotEqual(adapter.identity,repair.VisualPlanningObservationAdapter.identity)
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,adapter.model.state_dict()[k]))
            payload["architecture"]="legacy-v1"
            with self.assertRaises(ValueError):repair.validate_parser_payload(payload,plan)

    def test_sharded_batches_keep_every_frame_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[];start=0
            for i,n in enumerate((2,3,1)):
                path=Path(tmp)/f"{i}.pt";paths.append(path)
                repair.atomic_torch(path,{"tensors":{"images":torch.arange(start,start+n)[:,None]}});start+=n
            values=list(repair.batches(paths,4,torch.Generator().manual_seed(1)))
            self.assertEqual(sorted(torch.cat([v["images"] for v in values]).flatten().tolist()),list(range(6)))
            self.assertTrue(all(len(v["images"])<=4 for v in values))

    def test_carrier_bundle_recovers_interrupted_temporary(self):
        lineage=repair.unroll._fixture_lineages("training",count=1)[0]
        lineage=replace(lineage,carrier_identity=repair.CARRIER_ID,
            transitions=tuple(t for t in lineage.transitions if t.horizon==15))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"training.pt"
            path.with_suffix(".tmp.pt").write_bytes(b"interrupted")
            repair.atomic_bundle(path,(lineage,))
            restored=repair.load_carrier_lineage_bundle(path)[0]
            self.assertEqual(restored.carrier_identity,repair.CARRIER_ID)
            self.assertEqual(restored.segment_ends,lineage.segment_ends)
            self.assertTrue(torch.equal(restored.transitions[0].target,lineage.transitions[0].target))
            self.assertFalse(path.with_suffix(".tmp.pt").exists())

    def test_labels_preserve_absence_and_unavailable_masks(self):
        sample={"entities":[{"scenario_object_id":"pig:0000","entity_id":"p","lifecycle":"destroyed","body_present":False}]}
        metadata={}
        micro={"predicates":{k:{"availability":"unavailable"} for k in ("contact","supports")}}
        macro={"predicates":{k:{"availability":"unavailable"} for k in ("steady-state","structure-unstable")}}
        result=repair.targets(sample,metadata,micro,macro,["pig:0000"])
        self.assertEqual(result["presence"].tolist(),[0.0])
        self.assertFalse(result["macro_mask"].any());self.assertFalse(result["relation_mask"].any())

    def test_snapshot_sampling_includes_every_shot_and_its_endpoints(self):
        config=repair.CohortV2VisualParserConfig(image_height=8,image_width=8,hidden_dim=8)
        plan={"training_release":"/unused","training_release_identity":"source","config":asdict(config),
              "vocabulary":["pig:0000"],"frames_per_shot":2,"identity":"test"}
        raw={"shots":[{"shot_index":i} for i in range(2)]}
        samples=[{"fixed_step":i,"entities":[]} for i in range(3)]
        frames=[{"capture_metadata":{},"agent_observation":{"identity":f"obs-{i}","relative_path":"x"}} for i in range(3)]
        micro=[{"predicates":{k:{"availability":"unavailable"} for k in ("contact","supports")}}]*3
        macro=[{"predicates":{k:{"availability":"unavailable"} for k in ("steady-state","structure-unstable")}}]*3
        with patch.object(repair,"trajectory",return_value=raw),patch.object(repair,"load_shot",return_value=(Path("/unused"),samples,frames,{"micro":micro,"macro":macro})),patch.object(repair,"image_tensor",return_value=torch.zeros(3,8,8,dtype=torch.uint8)):
            result=repair.prepare_shard(plan,{})
        self.assertEqual([(r["shot"],r["fixed_step"]) for r in result["references"]],[(0,0),(0,2),(1,0),(1,2)])

    def test_calibration_cannot_enter_training_plan(self):
        plan=plan_fixture();plan["training_records"][0]["exposure_role"]="calibration"
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repair.write(root/"plan.json",plan)
            with self.assertRaises(ValueError):repair.load_plan(root)

    def test_repaired_ranking_uses_matched_teacher_forced_reference(self):
        plan,rows,scores=fixture()
        plan["parser_repair_root"]="/repaired"
        plan["models"]=[c for c in plan["models"] if not c["name"].startswith("original")]
        scores={k:v for k,v in scores.items() if not k.startswith("original")}
        result=experiment.freeze_payload(plan,rows,scores)
        self.assertTrue(result["original"]["name"].startswith("teacher-forced-h15"))
        self.assertEqual(len(result["comparison_table"]),24)
        self.assertEqual(pilot_plan(result)["systems"][0],"teacher_forced_cem")
        self.assertEqual(len(plan["models"]),9)

    def test_failed_parser_gate_stops_before_carrier_or_model_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(root=Path(tmp),device="cpu")
            mocks={k:Mock() for k in ("prepare","prepare_data","train_parser","prepare_rerun","rebuild_carriers","train_world_models")}
            with patch.multiple(repair,**mocks),patch.object(experiment,"prepare_endpoints"),patch.object(repair,"audit_parser",return_value=False),redirect_stdout(io.StringIO()):
                repair.run_repair(args)
            mocks["rebuild_carriers"].assert_not_called();mocks["train_world_models"].assert_not_called()

    def test_resume_after_last_epoch_publishes_without_retraining(self):
        plan=plan_fixture();config=repair.CohortV2VisualParserConfig(**plan["config"])
        model=repair.SlotConditionedVisualPredicateParser(config,tuple(plan["vocabulary"]))
        optimizer=torch.optim.AdamW(model.parameters())
        payload={"schema":"issue_70_repaired_parser_checkpoint_v2","identity":repair.PARSER_ID,
            "architecture":model.architecture_identity,"config":plan["config"],"vocabulary":plan["vocabulary"],
            "training_release_identity":plan["training_release_identity"],"epoch":2,"reports":[{},{}],
            "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repair.write(root/"plan.json",plan);repair.atomic_torch(root/"parser-progress.pt",payload)
            repair.write(root/"parser-data.json",{"role":"training","shards":[{"lineage":r["scenario_lineage_identity"]} for r in plan["training_records"]]})
            with patch.object(repair,"batches",side_effect=AssertionError("unexpected retraining")),redirect_stdout(io.StringIO()):
                repair.train_parser(SimpleNamespace(root=root,device="cpu"))
            self.assertTrue((root/"parser.pt").is_file())


if __name__=="__main__":unittest.main()
