"""Source-bound broader-condition development membership; no gameplay or fitting."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

from scripts import run_issue_76_dynamics_diagnostic as diagnostic
from scripts.cohort_v2_scenarios import create_scenario_template_record, materialize_template_bound_level_instance
from scripts.scenario_manifest import BenchmarkCondition
from tasks.task_generator.canonical_materialization import CanonicalMaterializationRequest


ROOT = diagnostic.ROOT
OUTPUT = ROOT/".local-artifacts/issue-76-compatibility-v1"
REVIEW = ROOT/"data/issue-76-compatibility"
PROTOCOL = "docs/issue-76-expansion-protocol.md"
SOURCES = ("scripts/issue_76_expansion.py","scripts/run_issue_76_compatibility.py",PROTOCOL,
           "scripts/run_issue_62_successor_cohort.py","scripts/cohort_v2_scenarios.py","scripts/scenario_manifest.py",
           "scripts/collect_rollouts.py","scripts/manual_agent.py","scripts/slingshot_readiness.py",
           "tasks/task_generator/canonical_materialization.py","tasks/task_generator/utils/generate_variations.py",
           "scripts/run_issue_70_parser_repair.py","world_model/training/cohort_v2_visual_parser.py",
           "world_model/data/deployment_temporal.py")
MEMBERS = (
    ("appearance-a",760610001,1,0,"type010102"),
    ("appearance-a",760610001,1,1,"type010102"),
    ("single",760610002,2,0,"type010101"),
    ("rolling",760610003,3,0,"type010103"),
    ("falling",760610004,4,0,"type010204"),
    ("sliding",760610005,5,0,"type010105"),
    ("appearance-b",760610006,6,0,"type010102"),
    ("appearance-b",760610006,6,1,"type010102"),
)


def read(path):
    return diagnostic.read(path)


def write(path,value):
    diagnostic.write(path,value)


def limits():
    return {"attempts":8,"independent_base_clusters":6,"workers":1,"active_seconds":1800,
            "attempt_seconds":240,"cpu_rss_mib":3072,"cuda_allocated_mib":2048,
            "artifact_bytes":2*2**30,"fixed_step_offset_max":600,"technical_retries":0,
            "engine_speed":10,"capture_stride":1,"shots":1,
            "action":{"drag_x":-10,"drag_y":-80,"tap_time_ms":0,"release_time_ms":600},
            "parser_criteria":{"pig_count_absolute_error_max":.25,"block_count_absolute_error_max":.5,
                               "task_center_absolute_error_max":.10},
            "parser_batch_size":1,"model_seeds":list(diagnostic.SEEDS),
            "new_training_updates_per_arm":0,"new_parser_training_updates":0}


def player_binding():
    archive = ROOT/"sciencebirdsgames/aligned-observation-v1/novphy-physics-player-2019.4.41f2.tar.gz"
    with tarfile.open(archive) as bundle:
        provenance = json.load(bundle.extractfile("provenance.json"))
    return {"archive":str(archive),"archive_bytes":archive.stat().st_size,"provenance":provenance}


def exposure_projection(value):
    seeds,identities,roles = set(),set(),set()
    def visit(item):
        if isinstance(item,dict):
            for key,v in item.items():
                if key in ("source_text","source_xml","config","state_dict"): continue
                if key in ("generation_seed","seed") and type(v) is int: seeds.add(v)
                if key == "exposure_role" and isinstance(v,str): roles.add(v)
                visit(v)
        elif isinstance(item,list):
            for v in item: visit(v)
        elif isinstance(item,str) and item.startswith(("level-instance-v1:","scenario-lineage-v1:")):
            identities.add(item)
    visit(value)
    return {"generation_or_reserved_seeds":sorted(seeds),"scenario_identities":sorted(identities),"roles":sorted(roles)}


def exposure_sources():
    """Only published plans/lineage metadata, never sealed captures or outcome files."""
    paths = set()
    for issue in (62,63,66,67,68,69,70,71,72,73,74,75):
        for directory in (ROOT/".local-artifacts").glob(f"issue-{issue}-*"):
            if not directory.is_dir(): continue
            for pattern in ("plan.json","production-plan.json","pilot-plan.json","manifest.json",
                            "frozen-collection-plan.json","pilot-source-inventory.json",
                            "experiment/pilot-source-inventory.json","release/production-plan.json",
                            "release/manifest.json"):
                path = directory/pattern
                if path.is_file(): paths.add(path)
    public = ROOT/"data/runtime_evidence"
    for pattern in ("*/partition-exposure-manifest.json","*/production-parameter-plan.json",
                    "issue-57/cohort-v2-gameplay-success-protocol-v*.json"):
        paths.update(public.glob(pattern))
    required = (ROOT/".local-artifacts/issue-62-successor-cohort-v4/production-plan.json",
                ROOT/".local-artifacts/issue-68-corrective-cohort-v2/production-plan.json",
                public/"issue-57/cohort-v2-gameplay-success-protocol-v2.json")
    if any(p not in paths for p in required): raise ValueError("required prior exposure inventory missing")
    return [{"path":str(p.relative_to(ROOT)),**exposure_projection(read(p))} for p in sorted(paths)]


def materialize(member,template,output):
    """Use the condition's workbook coordinates, including the novel columns."""
    source = ROOT/"tasks/task_templates"/template["template"]
    row = template["constraints"]
    coordinates = row["active_coordinates"]
    if row["active_restrictions"] is not None:
        raise ValueError("this frozen smoke requires the selected unrestricted workbook rows")
    condition = BenchmarkCondition(f"novelty_level_{member['novelty_level']}",member["generator_family"])
    record = create_scenario_template_record(source.read_bytes(),source_reference=str(source.relative_to(ROOT)),
                                             benchmark_conditions=[condition])
    # The old generic constraint binder only understands normal columns B-H.
    # This wrapper binds the exact inventory row and active normal/novel columns instead.
    request = CanonicalMaterializationRequest(template_path=source,output_xml_path=output/"scenario.xml",
        output_manifest_path=output/"generated-scenario.json",template_name=source.stem.split("_",1)[1],
        benchmark_condition=condition,template_identity=record.identity,generation_seed=member["generation_seed"],
        reference_point=tuple(coordinates[:2]),min_coordinate=tuple(coordinates[2:4]),
        max_coordinate=tuple(coordinates[4:6]),restricted_objects=(),template_source_reference=str(source.relative_to(ROOT)))
    with redirect_stdout(StringIO()):
        generated,scenario = materialize_template_bound_level_instance(request,record,publish=False)
    return generated,scenario


def make_plan(output=OUTPUT):
    original = read(diagnostic.OUTPUT/"plan.json")
    report = read(diagnostic.OUTPUT/"result.json")
    if not report["diagnostics_complete"] or report["disposition"] != "readiness_or_precision_insufficient":
        raise ValueError("requires the preserved complete diagnostic pre-access stop")
    inventory = read(diagnostic.OUTPUT/"novelty-inventory.json")
    lookup = {(t["novelty_level"],t["family"]):t for t in inventory["templates"]}
    previous = exposure_sources()
    old_seeds = {s for p in previous for s in p["generation_or_reserved_seeds"]}
    old_ids = {s for p in previous for s in p["scenario_identities"]}
    if old_seeds & {m[1] for m in MEMBERS}: raise ValueError("development generation seed overlaps prior exposure")
    members = []
    for i,(cluster,seed,cluster_ordinal,novelty,family) in enumerate(MEMBERS,1):
        template = lookup[novelty,family]
        if not template["static_slot_action_fit"]: raise ValueError("selected metadata template is not statically compatible")
        member = {"ordinal":i,"identity":f"issue-76-compatibility-{i:02d}","base_cluster":cluster,
                  "generation_seed":seed,"environment_seed":760620000+cluster_ordinal,
                  "novelty_level":novelty,"generator_family":family,"exposure_role":"calibration",
                  "role_scope":"compatibility_design_only_excluded_from_fresh_evaluation","template":template}
        generated,scenario = materialize(member,template,output/"authorities"/member["identity"])
        ids = exposure_projection(scenario.to_dict())["scenario_identities"]
        if old_ids & set(ids): raise ValueError("generated scenario identity overlaps prior exposure")
        tree = ET.fromstring(generated.xml_content)
        slots = sorted(n.attrib["scenarioObjectId"] for n in tree.iter() if "scenarioObjectId" in n.attrib)
        member.update({"scenario":scenario.to_dict(),"xml":generated.xml_content.decode("utf-8"),
                       "generated_slots":slots,"missing_slots":sorted(set(slots)-set(original["contract"]["vocabulary"]))})
        members.append(member)
    return {"schema":"issue_76_compatibility_plan_v1","identity":"issue-76-compatibility-v1",
            "authorization":"user explicitly requested implementation of the ten conditional todos; development compatibility stage only",
            "original_plan_identity":original["identity"],"original_disposition":report["disposition"],
            "contract":original["contract"],"limits":limits(),"members":members,"player":player_binding(),
            "parser_source":str(diagnostic.base.old.repair.ROOT),
            "prior_exposure":previous,"source_text":{p:(ROOT/p).read_text() for p in SOURCES},
            "source75_disposition":original["source75_disposition"],
            "exposure_audit_scope":"known repository public/cohort/pilot metadata; no sealed outcomes; not a fresh-access authorization or claim about external unregistered datasets",
            "fresh_evaluation_opened":False,"final_evaluation_opened":False,"issue_64_authorized":False}


def load_plan(output=OUTPUT):
    saved = read(output/"plan.json")
    if saved != make_plan(output): raise ValueError("compatibility source/membership/exposure freeze changed; preserve this run")
    for name in ("plan.json","parser.pt"):
        if (output/"parser"/name).read_bytes() != (Path(saved["parser_source"])/name).read_bytes():
            raise ValueError("frozen shared parser bytes differ from the original checkpoint/plan")
    return saved


def runtime_config(member,relative_xml):
    evaluation = ET.Element("evaluation")
    ET.SubElement(evaluation,"novelty_detection_measurement",{"step":"1","measure_in_training":"False","measure_in_testing":"False"})
    trial = ET.SubElement(ET.SubElement(evaluation,"trials"),"trial",{
        "id":"0","number_of_executions":"1","checkpoint_time_limit":"9999999",
        "checkpoint_interaction_limit":"9999999","notify_novelty":"False"})
    level_set = ET.SubElement(trial,"game_level_set",{"mode":"training","time_limit":"9999999",
        "total_interaction_limit":"9999999","attempt_limit_per_level":"1","allow_level_selection":"False"})
    ET.SubElement(level_set,"game_levels",{"level_path":relative_xml})
    ET.indent(evaluation,space="  ")
    return ET.tostring(evaluation,encoding="utf-8",xml_declaration=True)+b"\n"


def install_level(game,member):
    relative = Path("9001_Data/StreamingAssets/Levels")/f"novelty_level_{member['novelty_level']}"/member["generator_family"]/"Levels"/f"{member['identity']}.xml"
    from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_bytes
    write_immutable_cohort_v2_bytes(member["xml"].encode(),game/relative)
    (game/"config.xml").write_bytes(runtime_config(member,relative.as_posix()))
    return relative


def readiness(plan,report=None):
    return {"compatibility_complete":bool(report and report["complete"]),
            "compatibility_passed":bool(report and report["compatibility_passed"]),
            "candidate_eligible":False,"reason":"#75 candidate failed; compatibility cannot reverse its disposition",
            "fresh_numeric_protocol_frozen":False,"archive_verified":False,
            "fresh_collection_allowed":False,"issue_64_authorized":False,"issue_65_authorized":False,
            "next_required":"report compatibility evidence, then cost/freeze any necessary symmetric candidate repair; obtain archive authority/destination before fresh access"}
