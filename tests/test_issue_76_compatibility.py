import argparse
from contextlib import redirect_stdout
import copy
from io import StringIO
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from scripts import issue_76_expansion as e
from scripts import run_issue_76_compatibility as r
from scripts import issue_76_compatibility_media as media
from scripts import issue_76_asset_audit as assets
from scripts import process_lifecycle as lifecycle
from scripts.issue_76_novelty_inventory import build_inventory
from scripts.process_lifecycle import (cleanup_actions, persist_after_cleanup,
    record_cleanup_failures, registered_session_popen, registry_snapshot)
from tests.test_issue_76_dynamics_diagnostic import VOCABULARY


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
        return stat[stat.rindex(")") + 2] != "Z"
    except (OSError, ProcessLookupError):
        return False


def _wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.02)
    return predicate()


def _worker_with_late_grandchild(directory):
    directory = Path(directory)
    child_ready = directory / "child.pid"
    grandchild_pid = directory / "grandchild.pid"
    grandchild = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(300)"
    child = f"""import os,signal,subprocess,sys,time
spawned=[]
ready={str(child_ready)!r}
pidfile={str(grandchild_pid)!r}
code={grandchild!r}
open(ready,'w').write(str(os.getpid()))
def stop(*_):
    if not spawned:
        spawned.append(subprocess.Popen([sys.executable,'-c',code]))
        open(pidfile,'w').write(str(spawned[0].pid))
signal.signal(signal.SIGTERM,stop)
while True:
    time.sleep(1)
"""
    subprocess.Popen([sys.executable, "-c", child])
    while True:
        time.sleep(1)


def _worker_exits_before_child(directory):
    pidfile = str(Path(directory) / "orphan.pid")
    child = (
        "import os,signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"open({pidfile!r},'w').write(str(os.getpid()));"
        "time.sleep(300)"
    )
    subprocess.Popen([sys.executable, "-c", child])


def _worker_exits_before_detached_child(directory):
    pidfile = str(Path(directory)/"detached.pid")
    child = (
        "import os,signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"open({pidfile!r},'w').write(str(os.getpid()));"
        "time.sleep(300)"
    )
    registered_session_popen([sys.executable,"-c",child])


def _worker_launches_registered_engine(directory):
    from scripts import manual_agent
    root = Path(directory)
    game = root/"game"; game.mkdir()
    (game/"game_playing_interface.jar").write_text("fixture",encoding="ascii")
    binary = root/"bin"; binary.mkdir()
    java = binary/"java"
    java.write_text("""#!/usr/bin/env python3
import os,signal,subprocess,sys,time
open(sys.argv[-1] if sys.argv[-1].endswith('.pid') else os.environ['ENGINE_PID'],'w').write(str(os.getpid()))
child=subprocess.Popen([sys.executable,'-c',"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(300)"])
open(os.environ['ENGINE_CHILD_PID'],'w').write(str(child.pid))
time.sleep(300)
""",encoding="ascii")
    java.chmod(0o755)
    os.environ["PATH"] = str(binary)+os.pathsep+os.environ["PATH"]
    os.environ["ENGINE_PID"] = str(root/"engine.pid")
    os.environ["ENGINE_CHILD_PID"] = str(root/"engine-child.pid")
    manual_agent.start_engine(game,False)
    (root/"launch-returned").write_text("yes",encoding="ascii")
    time.sleep(300)


def _term_ignoring_worker(ready):
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    Path(ready).write_text(str(os.getpid()),encoding="ascii")
    while True: time.sleep(1)


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

    def test_supervisor_interrupt_persists_result_and_budget_before_reraising(self):
        class InterruptProcess:
            pid = 999991
            exitcode = None
            def is_alive(self): return True
            def join(self,timeout=None): raise KeyboardInterrupt("compatibility interrupted")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            member = {"identity":"member","ordinal":1,"base_cluster":"base","novelty_level":0,
                      "generator_family":"fixture","generation_seed":1}
            limits = {"active_seconds":10,"attempt_seconds":10,"fixed_step_offset_max":10,
                      "cpu_rss_mib":10,"artifact_bytes":100}
            args = argparse.Namespace(output=output,device="cpu")
            process = InterruptProcess()
            with patch.object(r,"start_isolated_worker",return_value=process), \
                    patch.object(r,"terminate_worker",side_effect=RuntimeError("registry retained at fixture")), \
                    patch.object(r,"log"):
                with self.assertRaisesRegex(KeyboardInterrupt,"compatibility interrupted"):
                    r.run_smoke(args,{"identity":"plan","members":[member],"limits":limits,"player":{}},True)
            self.assertTrue(r.result_path(output,member).is_file())
            budget = e.read(output/"budget.json")
            self.assertIn("registry retained at fixture",budget["cleanup_failures"][0])

    def test_proc_memory_measurement_needs_no_optional_package(self):
        self.assertGreater(r.process_rss(os.getpid()),0)

    def test_registry_append_retries_short_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"registry"; path.write_bytes(b"")
            real_write = os.write
            calls = []
            def short_write(descriptor,data):
                count = max(1,len(data)//2)
                calls.append(count)
                return real_write(descriptor,data[:count])
            with patch.object(lifecycle.os,"write",side_effect=short_write):
                lifecycle._append_registry(path,"P short-write-token")
            self.assertGreater(len(calls),1)
            self.assertEqual(path.read_text(encoding="ascii"),"P short-write-token\n")

    def test_registry_write_failure_prevents_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"registry"; path.write_bytes(b"")
            popen = Mock()
            with patch.dict(os.environ,{lifecycle.REGISTRY_ENV:str(path)}), \
                    patch.object(lifecycle.os,"write",side_effect=OSError("quota exhausted")):
                with self.assertRaisesRegex(OSError,"quota exhausted"):
                    lifecycle.registered_session_popen(["never-exec"],popen=popen)
            popen.assert_not_called()
            self.assertEqual(path.read_bytes(),b"")

    def test_pending_registration_does_not_skip_group_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory)/"ready"
            worker = r.start_isolated_worker(multiprocessing.get_context("spawn"),_term_ignoring_worker,(ready,))
            registry = Path(worker.novphy_process_group_registry)
            try:
                self.assertTrue(_wait_for(ready.exists))
                with registry.open("a",encoding="ascii") as stream: stream.write("P never-ready\n")
                with patch.object(r,"WORKER_STOP_GRACE_SECONDS",.2):
                    with self.assertRaisesRegex(RuntimeError,f"never-ready.*{registry}"):
                        r.terminate_worker(worker)
                self.assertFalse(worker.is_alive())
                self.assertTrue(registry.exists())
            finally:
                if worker.is_alive():
                    try: os.killpg(worker.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    worker.join(2)

    def test_terminate_worker_drains_ready_group_registered_after_late_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker_ready = root/"worker-ready"
            trigger = root/"register-now"
            child_ready = root/"child-ready"
            token = "late-ready-token"
            worker = r.start_isolated_worker(multiprocessing.get_context("spawn"),
                _term_ignoring_worker,(worker_ready,))
            registry = Path(worker.novphy_process_group_registry)
            child = None
            child_pid = None
            try:
                self.assertTrue(_wait_for(worker_ready.exists))
                with registry.open("a",encoding="ascii") as stream: stream.write(f"P {token}\n")
                code = """import os,signal,sys,time
from pathlib import Path
from scripts.process_lifecycle import _append_registry,process_identity
registry,token,trigger,ready=sys.argv[1:]
os.setsid()
pid=os.getpid()
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path(ready).write_text(str(pid),encoding='ascii')
while not Path(trigger).exists(): time.sleep(.005)
_append_registry(registry,f'R {token} {os.getpgrp()} {pid} {process_identity(pid)}')
Path(ready+'.registered').write_text('yes',encoding='ascii')
time.sleep(300)
"""
                child = subprocess.Popen([sys.executable,"-c",code,str(registry),token,
                    str(trigger),str(child_ready)])
                self.assertTrue(_wait_for(child_ready.exists))
                child_pid = int(child_ready.read_text(encoding="ascii"))
                self.assertEqual(os.getpgid(child_pid),child_pid)
                registered = Path(str(child_ready)+".registered")
                real_worker_groups = r._worker_groups
                def trigger_after_dead_snapshot(process):
                    snapshot = real_worker_groups(process)
                    if not process.is_alive() and not trigger.exists():
                        trigger.write_text("go",encoding="ascii")
                        self.assertTrue(_wait_for(registered.exists))
                    return snapshot
                error = None
                with patch.object(r,"_worker_groups",side_effect=trigger_after_dead_snapshot), \
                        patch.object(r,"WORKER_STOP_GRACE_SECONDS",.2):
                    try: r.terminate_worker(worker)
                    except RuntimeError as caught: error = caught
                self.assertTrue(registered.exists())
                self.assertTrue(_wait_for(lambda: not _pid_is_running(child_pid)),
                    f"late-registered process group survived cleanup: {error}")
                self.assertIsNone(error)
            finally:
                if child_pid is not None and _pid_is_running(child_pid):
                    try: os.killpg(child_pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                if child is not None:
                    try: child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait(timeout=2)
                if worker.is_alive():
                    try: os.killpg(worker.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                worker.join(2)

    def test_terminate_worker_reaps_term_ignoring_late_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = multiprocessing.get_context("spawn")
            worker = r.start_isolated_worker(context,_worker_with_late_grandchild,(directory,))
            tracked = []
            try:
                self.assertTrue(_wait_for((root/"child.pid").exists))
                tracked.append(int((root/"child.pid").read_text(encoding="ascii")))
                with patch.object(r,"WORKER_STOP_GRACE_SECONDS",.2):
                    r.terminate_worker(worker)
                self.assertTrue(_wait_for((root/"grandchild.pid").exists))
                tracked.append(int((root/"grandchild.pid").read_text(encoding="ascii")))
                self.assertTrue(_wait_for(lambda: all(not _pid_is_running(pid) for pid in tracked)))
            finally:
                for pid in tracked:
                    try: os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                if worker.is_alive(): worker.kill()
                worker.join(2)

    def test_terminate_worker_reaps_group_after_worker_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            worker = r.start_isolated_worker(context,_worker_exits_before_child,(directory,))
            pidfile = Path(directory)/"orphan.pid"
            child_pid = None
            try:
                self.assertTrue(_wait_for(pidfile.exists))
                child_pid = int(pidfile.read_text(encoding="ascii"))
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertTrue(_pid_is_running(child_pid))
                with patch.object(r,"WORKER_STOP_GRACE_SECONDS",.2):
                    r.terminate_worker(worker)
                self.assertTrue(_wait_for(lambda: not _pid_is_running(child_pid)))
            finally:
                if child_pid is not None and _pid_is_running(child_pid):
                    try: os.killpg(worker.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                if worker.is_alive(): worker.kill()
                worker.join(2)

    def test_terminate_worker_reaps_registered_detached_group_after_worker_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = r.start_isolated_worker(multiprocessing.get_context("spawn"),
                _worker_exits_before_detached_child,(directory,))
            pidfile = Path(directory)/"detached.pid"
            child_pid = None
            try:
                self.assertTrue(_wait_for(pidfile.exists))
                child_pid = int(pidfile.read_text(encoding="ascii"))
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertNotEqual(os.getpgid(child_pid),worker.pid)
                with patch.object(r,"WORKER_STOP_GRACE_SECONDS",.2): r.terminate_worker(worker)
                self.assertTrue(_wait_for(lambda: not _pid_is_running(child_pid)))
            finally:
                if child_pid is not None and _pid_is_running(child_pid):
                    try: os.killpg(child_pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                if worker.is_alive(): worker.kill()
                worker.join(2)

    def test_worker_death_during_detached_launch_does_not_orphan_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = r.start_isolated_worker(multiprocessing.get_context("spawn"),
                _worker_launches_registered_engine,(directory,))
            engine_pid = child_pid = None
            try:
                self.assertTrue(_wait_for((root/"engine-child.pid").exists))
                engine_pid = int((root/"engine.pid").read_text(encoding="ascii"))
                child_pid = int((root/"engine-child.pid").read_text(encoding="ascii"))
                worker.kill(); worker.join(2)
                with patch.object(r,"WORKER_STOP_GRACE_SECONDS",.2): r.terminate_worker(worker)
                self.assertTrue(_wait_for(lambda: not _pid_is_running(child_pid)))
            finally:
                if engine_pid is not None and (_pid_is_running(engine_pid) or (child_pid and _pid_is_running(child_pid))):
                    try: os.killpg(engine_pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                if worker.is_alive(): worker.kill()
                worker.join(2)

    def test_start_worker_interruption_reaps_started_descendants(self):
        before = set(Path("/tmp").glob("novphy-worker-groups-*.txt"))
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/"orphan.pid"
            def interrupt_after_child(_pid):
                self.assertTrue(_wait_for(pidfile.exists))
                raise KeyboardInterrupt("injected readiness interrupt")
            with patch("scripts.process_lifecycle.process_identity",side_effect=interrupt_after_child):
                with self.assertRaisesRegex(KeyboardInterrupt,"readiness interrupt"):
                    r.start_isolated_worker(multiprocessing.get_context("spawn"),
                        _worker_exits_before_child,(directory,))
            child_pid = int(pidfile.read_text(encoding="ascii"))
            self.assertTrue(_wait_for(lambda: not _pid_is_running(child_pid)))
        self.assertEqual(set(Path("/tmp").glob("novphy-worker-groups-*.txt")),before)

    def test_terminate_worker_raises_if_worker_survives_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory)/"groups.txt"; registry.write_text("",encoding="ascii")
            class StuckWorker:
                pid = 99999999
                novphy_process_group_registry = str(registry)
                def is_alive(self): return True
                def terminate(self): pass
                def kill(self): pass
                def join(self,timeout=None): pass
            with patch.object(r,"WORKER_STOP_GRACE_SECONDS",0):
                with self.assertRaisesRegex(RuntimeError,f"survived SIGKILL.*{registry}"):
                    r.terminate_worker(StuckWorker())
            self.assertTrue(registry.exists())

    def test_cleanup_runs_all_actions_and_persists_before_reraising(self):
        events = []
        result = {"failure":None}
        first = RuntimeError("engine cleanup failed")
        def fail(name,error):
            def action(): events.append(name); raise error
            return action
        failures = cleanup_actions((("engine",fail("engine",first)),
                                    ("display",fail("display",ValueError("display cleanup failed"))),
                                    ("environment",lambda: events.append("environment"))))
        record_cleanup_failures(result,failures)
        with self.assertRaisesRegex(RuntimeError,"engine cleanup failed"):
            persist_after_cleanup(failures,lambda: events.append(("persist",dict(result))))
        self.assertEqual(events[:3],["engine","display","environment"])
        self.assertEqual(events[3][0],"persist")
        self.assertEqual(len(events[3][1]["cleanup_failures"]),2)

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
