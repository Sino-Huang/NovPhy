import argparse
from contextlib import redirect_stdout
import copy
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from scripts import issue_76_expansion as e
from scripts import run_issue_76_compatibility as r
from scripts import issue_76_compatibility_media as media
from scripts import issue_76_asset_audit as assets
from scripts.issue_76_novelty_inventory import build_inventory
from tests.test_issue_76_dynamics_diagnostic import VOCABULARY


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = build_inventory(e.ROOT,VOCABULARY)

    def fake_read(self,path):
        if path.name == "plan.json": return {"identity":"old","contract":{"vocabulary":VOCABULARY},"source75_disposition":"not_supported_by_this_pilot"}
        if path.name == "result.json": return {"diagnostics_complete":True,"disposition":"readiness_or_precision_insufficient"}
        return self.inventory

    def make_plan(self,output):
        with patch.object(e,"read",side_effect=self.fake_read), patch.object(e,"exposure_sources",return_value=[]), \
             patch.object(e,"player_binding",return_value={}), redirect_stdout(StringIO()):
            return e.make_plan(output)

    def test_real_templates_materialize_without_writes_and_keep_paired_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"not-created"
            plan = self.make_plan(output)
            self.assertFalse(output.exists())
            self.assertEqual(len(plan["members"]),8)
            self.assertEqual(len({m["base_cluster"] for m in plan["members"]}),6)
            a,b = plan["members"][:2]
            self.assertEqual((a["generation_seed"],a["environment_seed"],a["base_cluster"]),
                             (b["generation_seed"],b["environment_seed"],b["base_cluster"]))
            self.assertNotEqual(a["scenario"],b["scenario"])
            self.assertTrue(all(m["exposure_role"] == "calibration" for m in plan["members"]))
            self.assertTrue(all(m["xml"].startswith("<?xml") for m in plan["members"]))
            self.assertFalse(plan["fresh_evaluation_opened"])

    def test_generator_name_uses_condition_specific_stem_without_instance_prefix(self):
        template = next(t for t in self.inventory["templates"] if t["novelty_level"] == 0 and t["family"] == "type010102")
        member = {"novelty_level":0,"generator_family":"type010102","generation_seed":760610001}
        with patch.object(e,"materialize_template_bound_level_instance",return_value=("generated","scenario")) as materializer:
            e.materialize(member,template,Path("unused"))
        request = materializer.call_args.args[0]
        self.assertEqual(request.template_name,"0_1_010102_0_2")
        self.assertFalse(materializer.call_args.kwargs["publish"])

    def test_runtime_keeps_actual_condition_not_normal_type2(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(Path(directory))
            for member in plan["members"][:2]:
                game = Path(directory)/str(member["ordinal"])
                relative = e.install_level(game,member)
                self.assertIn(f"novelty_level_{member['novelty_level']}",str(relative))
                self.assertIn(member["generator_family"],str(relative))
                config = ET.parse(game/"config.xml")
                self.assertEqual(config.find(".//game_levels").attrib["level_path"],str(relative))
                self.assertEqual(config.find(".//trial").attrib["notify_novelty"],"False")
                self.assertEqual((game/relative).read_text(),member["xml"])

    def test_prior_seed_collision_rejected_before_generation(self):
        with patch.object(e,"read",side_effect=self.fake_read), \
             patch.object(e,"exposure_sources",return_value=[{"generation_or_reserved_seeds":[760610001],"scenario_identities":[]}]), \
             patch.object(e,"materialize",side_effect=AssertionError("must not generate")):
            with self.assertRaisesRegex(ValueError,"overlaps prior"): e.make_plan()

    def test_exposure_projection_is_metadata_only(self):
        value = {"generation_seed":123,"exposure_role":"final_evaluation","source_text":{"seed":456},
                 "member":{"scenario_lineage_identity":"scenario-lineage-v1:example"}}
        projection = e.exposure_projection(value)
        self.assertEqual(projection["generation_or_reserved_seeds"],[123])
        self.assertEqual(projection["scenario_identities"],["scenario-lineage-v1:example"])

    def test_compatibility_does_not_override_candidate_or_archive_gate(self):
        result = e.readiness({}, {"complete":True,"compatibility_passed":True})
        self.assertTrue(result["compatibility_passed"])
        self.assertFalse(result["candidate_eligible"])
        self.assertFalse(result["fresh_collection_allowed"])
        self.assertFalse(result["issue_64_authorized"])


class CaptureTests(unittest.TestCase):
    def measured(self):
        frame = {"nonconstant_rgb":True,"carrier_finite":True,"carrier_values":[0.]*236,
                 "metrics":{"pig_count_absolute_error":.25,"block_count_absolute_error":.5,"task_center_absolute_error":.1}}
        return {"runtime_missing_slots":[],"last_offset":225,"frames":[frame]}

    def test_frozen_parser_thresholds_and_no_silent_slot_truncation(self):
        measured = self.measured()
        self.assertTrue(all(r.compatibility_checks(measured,e.limits()).values()))
        measured["frames"][0]["metrics"]["pig_count_absolute_error"] = .251
        self.assertFalse(r.compatibility_checks(measured,e.limits())["pig_count_absolute_error"])
        measured["runtime_missing_slots"] = ["platform:0006"]
        self.assertFalse(r.compatibility_checks(measured,e.limits())["runtime_slots_supported"])
        measured["last_offset"] = 601
        self.assertFalse(r.compatibility_checks(measured,e.limits())["within_fixed_step_limit"])

    def test_missing_center_or_nonfinite_carrier_is_not_a_pass(self):
        measured = self.measured()
        measured["frames"][0]["metrics"]["task_center_absolute_error"] = None
        measured["frames"][0]["carrier_finite"] = False
        checks = r.compatibility_checks(measured,e.limits())
        self.assertFalse(checks["task_center_absolute_error"])
        self.assertFalse(checks["finite_carriers"])

    def test_partial_publication_retains_failures_and_unrun_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            members = [{"identity":f"member-{i}","ordinal":i+1,"novelty_level":0,"generator_family":"type010102"} for i in range(8)]
            plan = {"identity":"plan","members":members,"limits":e.limits()}
            e.write(r.result_path(output,members[0]),{"member_identity":"member-0","failure":"timeout","measurements":None,"checks":{},"video":None,"compatibility_passed":False})
            args = argparse.Namespace(output=output,review=output/"review",device="cpu")
            report = r.publication(args,plan)
            self.assertFalse(report["complete"])
            self.assertFalse(report["compatibility_passed"])
            self.assertEqual(report["attempts_recorded"],1)
            self.assertEqual(report["entries"][0]["result"]["failure"],"timeout")
            self.assertIsNone(report["entries"][1]["result"])
            self.assertIn("Not run",r.gallery(args,report))

    def test_completed_or_interrupted_attempts_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            members = [{"identity":f"member-{i}","ordinal":i+1} for i in range(2)]
            e.write(r.result_path(output,members[0]),{"member_identity":"member-0","compatibility_passed":True})
            (output/"attempts/member-1").mkdir(parents=True)
            with patch.object(r.multiprocessing,"get_context",side_effect=AssertionError("retry")), patch.object(r,"log"):
                r.run_smoke(argparse.Namespace(output=output,device="cpu"),{"identity":"plan","members":members,"limits":e.limits()},True)
            result = e.read(r.result_path(output,members[1]))
            self.assertEqual(result["failure"],"interrupted_single_attempt_retained_no_retry")

    def test_proc_memory_measurement_needs_no_optional_package(self):
        self.assertGreater(r.process_rss(os.getpid()),0)

    def test_failed_partial_capture_gets_media_without_changing_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            member = {"identity":"member","novelty_level":1,"xml":"<Level><GameObjects><Pig type='PinkBigPig'/></GameObjects></Level>"}
            plan = {"identity":"plan","members":[member],"limits":e.limits()}
            result = {"member_identity":"member","failure":"fixed_step_capture_limit","video":None}
            e.write(r.result_path(output,member),result)
            partial = output/"attempts/member/aligned-current"; partial.mkdir(parents=True)
            for i in range(3):
                e.write(partial/f"frame_{i:06d}.json",{"fixed_step":100+i,"fixed_time_seconds":2+i*.02})
                (partial/f"frame_{i:06d}.png").write_bytes(b"fixture")
            with patch.object(e,"load_plan",return_value=plan):
                root,manifest = media.media_inventory(output,output/"review")
            entry = manifest["entries"][0]
            self.assertTrue(entry["incomplete_capture"])
            self.assertEqual(entry["status"],"failed")
            self.assertEqual(len(entry["frames"]),3)
            self.assertAlmostEqual(entry["video"]["fps"],50)
            self.assertEqual(entry["resource_diagnosis"]["unresolved_builtin_pigs"],["PinkBigPig"])
            self.assertIn("incomplete=True",media.page(root,manifest))
            self.assertEqual(e.read(r.result_path(output,member)),result)

    def test_video_validation_rejects_wrong_frame_count_or_rate(self):
        expected = {"path":"unused","frames":3,"fps":50.}
        for frames,rate in (("2","50/1"),("3","25/1")):
            response = json.dumps({"streams":[{"nb_read_frames":frames,"r_frame_rate":rate,"time_base":"1/1000"}]})
            with patch.object(media.subprocess,"check_output",return_value=response):
                with self.assertRaisesRegex(ValueError,"frame count/timing"): media.check_video(expected)

    def test_webm_rational_rate_rounding_within_time_tick_is_accepted(self):
        response = json.dumps({"streams":[{"nb_read_frames":"148","r_frame_rate":"18199/364","time_base":"1/1000"}]})
        with patch.object(media.subprocess,"check_output",return_value=response):
            media.check_video({"path":"unused","frames":148,"fps":49.99725015123725})

    def test_float32_absolute_times_do_not_invent_variable_frame_rate(self):
        frames = [{"fixed_step":994+i,"fixed_time_seconds":t} for i,t in enumerate((20.76,20.7799988,20.8,20.82,20.84))]
        self.assertEqual(media.fixed_cadence(frames),.02)
        frames[2]["fixed_step"] += 1
        with self.assertRaisesRegex(ValueError,"cadence"): media.fixed_cadence(frames)

    def test_asset_audit_keeps_missing_prefab_and_physics_mismatch_separate(self):
        body = dict(zip(assets.BODY_FIELDS,(1.5,.5,0.,.05,1),strict=True))
        canonical = {"prefabs":{"BasicBig":{"rigidbody":body,"scripts":[{"class":"DieOnBirdHitCountPig"}]},
                                 "PinkBigPig":{"rigidbody":{**body,"m_CollisionDetection":0},"scripts":[]}}}
        aligned = {"prefabs":{"BasicBig":{"rigidbody":{**body,"m_Mass":.7},"scripts":[{"class":"ABPig"}]}}}
        result = assets.compare(canonical,aligned)
        self.assertEqual(result["missing_aligned_prefabs"],["PinkBigPig"])
        self.assertEqual(result["normal_basic_big_body_differences"]["m_Mass"],{"canonical":1.5,"aligned":.7})
        self.assertEqual(result["canonical_normal_novel_body_differences"],{"m_CollisionDetection":[1,0]})
        self.assertFalse(result["appearance_only_equivalence_established"])


if __name__ == "__main__": unittest.main()
