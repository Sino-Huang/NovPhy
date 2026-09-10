from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from scripts import issue_70_live_pilot as live
from scripts import run_issue_70_parser_repair as repair


class LiveCaptureTests(unittest.TestCase):
    def test_aligned_collector_rejects_null_graphics_before_launch(self):
        with patch.object(live.capture,"_materialize_slot") as materialize:
            with self.assertRaisesRegex(live.capture.SuccessorCohortError,"requires graphics"):
                live.capture._collect_lineage_attempt({},Path("/unused"),Path("/unused"),
                                                     release_identity="test",speed=10,headless=True)
        materialize.assert_not_called()

    def test_rgb_pilot_keeps_graphics_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(output=root,pilot_output=root/"corrective",audit=root/"audit",device="cpu")
            live.experiment.write(root/"pilot-source-inventory.json",{"authored_birds":[1]})
            adapter=SimpleNamespace(model=SimpleNamespace(object_vocabulary=("pig:0000","block:0000")),
                                    parser_checkpoint_identity="parser")
            frozen={"objective":{"pig_slots":[0],"block_slots":[1]},"parser_identity":"parser","corrected":{}}
            level={"ordinal":0,"generation_seed":700500000,"generator_family":"type010101"}
            old_path=root/"pilot-results/issue-70-pilot-l01-fixed_prior.json"
            live.experiment.write(old_path,{"failure":"original cached failure"})
            record={"terminal_reason":"shot_limit","executed_action_count":1,
                    "trajectory_identity":"trajectory","scenario_lineage_identity":"lineage","level_instance_identity":"level"}
            with patch.object(live.experiment,"load_adapter",return_value=adapter), \
                 patch.object(live.capture,"_collect_lineage_attempt",return_value=record) as collect, \
                 patch.object(live,"audit_video",return_value=None):
                result=live.run_trial(args,frozen,level,"fixed_prior")
            self.assertIsNone(result["failure"])
            self.assertFalse(collect.call_args.kwargs["headless"],"RGB capture must not use Unity -nographics")
            self.assertEqual(live.experiment.read(old_path),{"failure":"original cached failure"})
            self.assertTrue((args.pilot_output/"pilot-results"/old_path.name).is_file())

    def failed_fixture(self,root):
        args=SimpleNamespace(output=root/"experiment",audit=root/"audit",device="cpu",start_display=False)
        frozen={"pilot_allowed":True};plan=live.pilot_plan(frozen)
        live.experiment.write(args.output/"pilot-plan.json",plan)
        live.experiment.write(args.output/"pilot-freeze.json",frozen)
        live.experiment.write(args.output/"pilot-source-inventory.json",{
            "authored_birds":[1]*12,"seeds":[v["generation_seed"] for v in plan["levels"]],"source_overlap_count":0})
        for level in plan["levels"]:
            for system in plan["systems"]:
                slot=live.slot_for({**level,"authored_birds":1},system)
                live.experiment.write(args.output/"pilot-results"/f"{slot['slot_identity']}.json",{
                    "slot":slot,"shots":0,"video":None,"success":False,"final_evaluation_opened":False,
                    "failure":"LegacyGroundTruthProtocolError: request-38 record_count: is truncated"})
        return args

    def test_repair_preserves_failures_and_routes_to_new_capture_and_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.failed_fixture(root)
            result=live.prepare_render_repair(args)
            self.assertEqual(len(result["original_results"]),48)
            self.assertEqual(live.prepare_render_repair(args),result)
            for prior in result["original_results"]:
                self.assertEqual(live.experiment.read(args.output/prior["path"]),prior["result"])
            routed=repair.rerun_args(SimpleNamespace(root=root,device="cpu"))
            self.assertEqual(routed.pilot_output,args.output/live.RENDER_REPAIR_DIRECTORY)
            self.assertTrue(str(routed.audit).endswith("v6-rendered"))
            self.assertTrue(str(routed.summary).endswith("v6-rendered.json"))
            self.assertFalse(routed.pilot_output.exists())
            read=live.experiment.read
            def changed(path):
                value=read(path)
                if Path(path).name==Path(result["original_results"][0]["path"]).name:
                    return {**value,"shots":1}
                return value
            with patch.object(live.experiment,"read",side_effect=changed):
                with self.assertRaisesRegex(ValueError,"wholly uncaptured"):
                    live.prepare_render_repair(args)

    def test_all_capture_failures_cannot_be_published_as_gameplay(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.failed_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,"no completed captures"):
                live.pilot_report(args)

    def test_live_pilot_pauses_after_first_new_uncaptured_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(output=root,audit=root/"audit",device="cpu",start_display=False)
            frozen={"pilot_allowed":True};plan=live.pilot_plan(frozen)
            live.experiment.write(root/"pilot-plan.json",plan)
            live.experiment.write(root/"pilot-freeze.json",frozen)
            live.experiment.write(root/"pilot-game/provenance.json",{})
            receive=Mock();receive.poll.return_value=True
            receive.recv.return_value=("ok",{"success":False,"shots":0,"failure":"capture failed"})
            process=Mock(exitcode=0);ctx=Mock()
            ctx.Pipe.return_value=(receive,Mock());ctx.Process.return_value=process
            with patch.object(live.experiment,"endpoint_rows",return_value=[]), \
                 patch.object(live.experiment,"plan_for",return_value={}), \
                 patch.object(live.experiment,"load_scores",return_value={}), \
                 patch.object(live.experiment,"freeze_payload",return_value=frozen), \
                 patch.object(live,"check_disjointness"),patch.object(live.capture,"_verify_webm_encoder"), \
                 patch.object(live.multiprocessing,"get_context",return_value=ctx), \
                 patch.object(live,"write_gallery") as gallery:
                with self.assertRaisesRegex(RuntimeError,"uncaptured trial"):
                    live.run_pilot(args)
            process.start.assert_called_once()
            self.assertTrue(gallery.call_args.args[1]["incomplete"])


if __name__=="__main__":unittest.main()
