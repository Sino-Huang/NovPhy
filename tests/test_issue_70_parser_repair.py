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

    def test_rare_absence_has_equal_total_task_gradient(self):
        labels=torch.ones(28,2);labels[-1,0]=0;labels[14:,1]=0
        weights=repair.class_balance_weights(len(labels),labels.sum(0))
        logits=torch.zeros_like(labels,requires_grad=True)
        repair.balanced_task_presence_loss(logits,labels,["pig:0000","block:0000"],weights).backward()
        self.assertAlmostEqual(float(logits.grad[:-1,0].sum()),-float(logits.grad[-1,0]),places=6)
        self.assertAlmostEqual(float(weights[1,0]),14.0)

    def test_inventory_includes_later_training_scenarios_and_runtime_slots(self):
        records=[{"path":str(i),"exposure_role":"training"} for i in range(2)]
        samples=[{"entities":[{"scenario_object_id":"world:landscape:0000"}]}]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,slot in enumerate(("pig:0000","platform:0005")):
                (root/str(i)).mkdir()
                (root/str(i)/"scenario.xml").write_text(
                    f'<Level><Object scenarioObjectId="{slot}"/></Level>')
            with patch.object(repair,"trajectory",return_value={"shots":[{}]}), \
                 patch.object(repair,"load_shot",return_value=(None,samples,None,None)):
                self.assertEqual(repair.training_vocabulary(root,records,"release"),
                    ["pig:0000","platform:0005","world:landscape:0000"])
                records[-1]["exposure_role"]="calibration"
                with self.assertRaisesRegex(ValueError,"training-only"):
                    repair.training_vocabulary(root,records,"release")

    def test_repaired_checkpoint_round_trip_has_distinct_carrier(self):
        plan=plan_fixture();config=repair.CohortV2VisualParserConfig(**plan["config"])
        # The full #62 training corpus has 18 slots, including platform:0003-0005.
        plan["vocabulary"] = ([f"bird:{i:04d}" for i in range(3)]
            + [f"block:{i:04d}" for i in range(5)] + ["pig:0000"]
            + [f"platform:{i:04d}" for i in range(6)]
            + ["slingshot:0000", "world:landscape:0000", "world:landscape:0001"])
        model=repair.ConvSlotVisualPredicateParser(config,tuple(plan["vocabulary"]))
        payload={"schema":"issue_70_repaired_parser_checkpoint_v2","identity":repair.PARSER_ID,
            "architecture":model.architecture_identity,"config":plan["config"],"vocabulary":plan["vocabulary"],
            "training_release_identity":plan["training_release_identity"],"epoch":2,"reports":[{},{}],"model_state":model.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repair.write(root/"plan.json",plan);repair.atomic_torch(root/"parser.pt",payload)
            adapter=repair.load_repaired_adapter(root,"cpu")
            self.assertEqual(adapter.identity,repair.CARRIER_ID)
            self.assertEqual(adapter.max_entities,18)
            self.assertEqual(adapter.latent_dim,236)
            for spec in repair.specs_for(root,plan):
                self.assertEqual(spec.predictor_config.latent_dim,adapter.latent_dim)
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

    def test_perception_only_stops_even_when_objective_passes(self):
        args=SimpleNamespace(root=Path("/unused"),device="cpu")
        mocks={k:Mock() for k in ("prepare","prepare_data","train_parser","prepare_rerun","rebuild_carriers","train_world_models")}
        with patch.multiple(repair,**mocks),patch.object(experiment,"prepare_endpoints"), \
             patch.object(repair,"audit_parser",return_value=True),redirect_stdout(io.StringIO()):
            repair.run_repair(args,perception_only=True)
        mocks["train_parser"].assert_called_once_with(args)
        mocks["rebuild_carriers"].assert_not_called();mocks["train_world_models"].assert_not_called()

    def test_perception_cli_routes_to_bounded_stage(self):
        with patch.object(repair,"run_repair") as run:
            self.assertEqual(repair.main(["--run-perception","--device","cpu"]),0)
        self.assertEqual(run.call_args.kwargs,{"perception_only":True})

    def test_audit_records_historical_comparison_without_changing_gate(self):
        _,rows,_=fixture()
        for row in rows:
            for candidate in row["candidates"]:candidate["parsed_counts"]=candidate["actual_counts"]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"cnn";old=Path(tmp)/"old"
            plan=plan_fixture();plan["historical_comparison"]={"root":str(old),"matched_backbone_ablation":False}
            repair.write(root/"plan.json",plan)
            repair.write(old/"experiment/objective-audit.json",experiment.audit_payload(rows))
            with patch.object(experiment,"endpoint_rows",return_value=rows),redirect_stdout(io.StringIO()):
                self.assertTrue(repair.audit_parser(SimpleNamespace(root=root,device="cpu")))
            result=repair.read(root/"experiment/backbone-comparison.json")
            self.assertTrue(result["objective_gate_unchanged"])
            self.assertFalse(result["matched_backbone_ablation"])
            self.assertEqual(result["reference_count_ranking"],result["cnn_count_ranking"])

    def test_resume_after_last_epoch_publishes_without_retraining(self):
        plan=plan_fixture();config=repair.CohortV2VisualParserConfig(**plan["config"])
        model=repair.ConvSlotVisualPredicateParser(config,tuple(plan["vocabulary"]))
        optimizer=torch.optim.AdamW(model.parameters())
        payload={"schema":"issue_70_repaired_parser_checkpoint_v2","identity":repair.PARSER_ID,
            "architecture":model.architecture_identity,"config":plan["config"],"vocabulary":plan["vocabulary"],
            "training_release_identity":plan["training_release_identity"],"epoch":2,"reports":[{},{}],
            "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repair.write(root/"plan.json",plan);repair.atomic_torch(root/"parser-progress.pt",payload)
            repair.write(root/"parser-data.json",{"role":"training","shards":[{"lineage":r["scenario_lineage_identity"],"frames":2} for r in plan["training_records"]]})
            with patch.object(repair,"batches",side_effect=AssertionError("unexpected retraining")), \
                 patch.object(repair,"training_probe",return_value=torch.randint(0,256,(4,3,8,8),dtype=torch.uint8)), \
                 patch.object(repair,"diagnostic_probe",return_value={}), \
                 patch.object(repair,"loss_contract",return_value={"frames":6000,"weights":[[1.,1.],[1.,1.]]}),redirect_stdout(io.StringIO()):
                repair.train_parser(SimpleNamespace(root=root,device="cpu"))
            self.assertTrue((root/"parser.pt").is_file())

    def test_training_class_weights_freeze_once_from_shards(self):
        plan=plan_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"shard.pt"
            labels=torch.ones(28,2);labels[-1,0]=0;labels[14:,1]=0
            repair.atomic_torch(path,{"tensors":{"presence":labels}})
            data={"shards":[{"path":str(path)}]}
            first=repair.loss_contract(root,plan,data)
            self.assertEqual(first["positive_counts"],[27.,14.])
            with patch.object(repair.torch,"load",side_effect=AssertionError("no label rescan")):
                self.assertEqual(repair.loss_contract(root,plan,data),first)

    def test_health_check_rejects_image_blind_parser(self):
        config=repair.CohortV2VisualParserConfig(hidden_dim=8,image_height=8,image_width=8)
        model=repair.ConvSlotVisualPredicateParser(config,("pig:0000","block:0000"))
        with torch.no_grad():model.backbone[0].weight.zero_()
        with self.assertRaisesRegex(ValueError,"image-sensitivity"):
            repair.parser_health(model,torch.randint(0,256,(4,3,8,8),dtype=torch.uint8),model.object_vocabulary)

    def test_reuse_training_shards_binds_sources_and_does_not_copy(self):
        plan=plan_fixture()
        plan.update(training_release="release",frames_per_shot=8,sampling="uniform")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"new";source=Path(tmp)/"old"
            previous={**plan,"identity":"previous"}
            repair.write(source/"plan.json",previous)
            shard=source/"data.pt";repair.atomic_torch(shard,{})
            inventory={"plan_identity":"previous","role":"training","shards":[
                {"lineage":r["scenario_lineage_identity"],"path":str(shard),"frames":2} for r in plan["training_records"]]}
            repair.write(source/"parser-data.json",inventory)
            plan["parser_data_source"]=str(source);repair.write(root/"plan.json",plan)
            with patch.object(repair,"prepare_shard",side_effect=AssertionError("no rescan")):
                repair.prepare_data(SimpleNamespace(root=root))
                repair.prepare_data(SimpleNamespace(root=root))
            reused=repair.read(root/"parser-data.json")
            self.assertEqual(reused["source_plan_identity"],"previous")
            self.assertEqual(reused["shards"],inventory["shards"])
            self.assertFalse((root/"parser-data").exists())
            with patch.object(repair,"load_plan",return_value={**plan,"vocabulary":["other"]}):
                with self.assertRaisesRegex(ValueError,"vocabulary"):
                    repair.prepare_data(SimpleNamespace(root=Path(tmp)/"mismatch"))


if __name__=="__main__":unittest.main()
