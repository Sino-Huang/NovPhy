"""Freeze the issue-77 N2-normal-side appearance campaign; never capture or fit.

Stage N2 of docs/issue-77-novelty-experiments-plan.md: the normal (level 0)
side of the appearance pair type010102, 8 lineages, 13 fixed actions per
lineage (reference drag (-80,10) plus 12 angles 5..82 degrees at radius
80 px), one shot per branch, captured by the unchanged R3 shared-history
collector. The type010101 normal side reuses the frozen R3 bounded-transfer
held_out_model_selection lineages (declared in the N2 evaluation module);
no contract-identical 13-action normal-side corpus exists for type010102
(the issue-62 cohort used behavior-policy actions, not the fixed grid), so
it is captured fresh here. Deviation from plan section 5.2 ("normal side
already exists") is declared: that statement holds for type010101 via the
R3 corpus, not for type010102.

Small-project mode: no authorization gates or audit cycles. Membership, roles,
seeds, templates and limits are pinned here before any rendered run.

Role contract: every lineage is held_out_evaluation (exposure calibration,
no fit partition); this corpus is evaluation-only and is never fitted.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts import issue_76_expansion as files
from world_model.planning.gameplay import SlingshotAction
from world_model.training.native_history_data import VOCABULARY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'issue-77-n2n-v1'
ROOT = PROJECT_ROOT / '.local-artifacts' / IDENTITY
PREREQUISITE = PROJECT_ROOT / '.local-artifacts/issue-76-shared-history-smoke-v4/validation.json'
PLAYER_PLAN = PROJECT_ROOT / '.local-artifacts/issue-76-shared-history-smoke-v4/plan.json'
NOVELTY_INVENTORY = PROJECT_ROOT / '.local-artifacts/issue-76-dynamics-diagnostic-v1/novelty-inventory.json'
PROTOCOL = 'docs/issue-77-novelty-experiments-plan.md'
FAMILIES = ('type010102',)
LINEAGES_PER_FAMILY = 8
NOVELTY_LEVEL = 0
EXPECTED_TEMPLATES = {
    'type010102': 'novelty_level_0/type010102/Levels/00001_0_1_010102_0_2.xml',
}
GENERATION_SEED_BASE = 764500000
ENGINE_SEED_BASE = 764600000
RESERVED_RANGES = (
    ('issue-62 pilot cohort generation offsets', 6300000, 6300000),
    ('issue-62 production cohort generation offsets', 63000000, 63000000),
    ('#74 matched training seeds', 20260908, 20260910),
    ('parked section 8 generation inventory', 761700001, 761700700),
    ('parked section 8 engine inventory', 761800001, 761800700),
    ('R3 generation seeds', 761900001, 761900100),
    ('R3 engine seeds', 762000001, 762000100),
    ('R3 readout initialization seeds', 763100000, 763100999),
    ('R3 selector initialization seeds', 763200000, 763200999),
    ('issue-77 N1 generation seeds', 764000001, 764000016),
    ('issue-77 N1 engine seeds', 764100001, 764100016),
    ('issue-77 N2 generation seeds', 764200001, 764200016),
    ('issue-77 N2 engine seeds', 764300001, 764300016),
    ('issue-77 reserved adaptation initialization seeds', 764400000, 764400099),
)
SOURCES = (
    'scripts/prepare_issue_77_n2n.py',
    'scripts/run_issue_77_n2n.py',
    'tests/test_issue_77_n2n_campaign.py',
    'tests/test_issue_77_n2n_inventory.py',
    PROTOCOL,
    'scripts/issue_76_bounded_transfer_capture.py',
    'scripts/issue_76_shared_history_capture.py',
    'scripts/issue_76_display_start.py',
    'scripts/issue_76_live_episode.py',
    'scripts/issue_76_censored_episode.py',
    'scripts/issue_76_episode_capture.py',
    'scripts/issue_76_fixed_replay_policy.py',
    'scripts/issue_76_shared_player_storage.py',
    'scripts/issue_76_native_outcomes.py',
    'scripts/issue_76_expansion.py',
    'scripts/native_segment_trace.py',
    'scripts/canonical_native_trace.py',
    'scripts/observation_trace.py',
    'scripts/process_lifecycle.py',
    'scripts/collect_rollouts.py',
    'scripts/slingshot_readiness.py',
    'scripts/cohort_v2_scenarios.py',
    'scripts/scenario_manifest.py',
    'scripts/run_issue_62_successor_cohort.py',
    'scripts/run_issue_76_compatibility.py',
    'src/webui/bridge.py',
    'world_model/training/native_history_data.py',
    'world_model/data/deployment_temporal.py',
    'world_model/planning/gameplay.py',
    'tasks/task_generator/canonical_materialization.py',
    'tasks/task_generator/utils/data_classes.py',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/ObservationCaptureProtocol.cs',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureV2EngineProtocol.cs',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicalSnapshotRuntime.cs',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicalCaptureModels.cs',
)
PARTIAL_SOURCE_NOTES = {
    'scripts/run_issue_62_successor_cohort.py':
        'Frozen in full; the capture chain adopts only its engine/bridge primitives and _replace_json.',
    'scripts/run_issue_76_compatibility.py':
        'Frozen in full; the supervisor adopts only process_rss/terminate_worker.',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs':
        'Frozen in full only to bind the verified repo/cached-player request-72 divergence.',
}


def lineage_assignments():
    """Return the outcome-independent per-family membership; every lineage is
    evaluation-only (held_out_evaluation, calibration exposure, no fit partition)."""
    rows = []
    for family in FAMILIES:
        for family_rank in range(1, LINEAGES_PER_FAMILY + 1):
            ordinal = len(rows) + 1
            identity = f'issue-77-n2n-{ordinal:03}'
            exposure_role, fit_partition, study_role = 'calibration', None, 'held_out_evaluation'
            rows.append({
                'identity': identity,
                'base_cluster': identity,
                'ordinal': ordinal,
                'generation_seed': GENERATION_SEED_BASE + ordinal,
                'engine_seed': ENGINE_SEED_BASE + ordinal,
                'generator_family': family,
                'novelty_level': NOVELTY_LEVEL,
                'exposure_role': exposure_role,
                'fit_partition': fit_partition,
                'study_role': study_role,
                'family_rank': family_rank,
            })
    generation = {row['generation_seed'] for row in rows}
    engine = {row['engine_seed'] for row in rows}
    if generation & engine:
        raise ValueError('issue-77 N2-normal generation and engine seeds overlap each other')
    for name, low, high in RESERVED_RANGES:
        if any(low <= seed <= high for seed in generation | engine):
            raise ValueError(f'issue-77 N2-normal seeds overlap the reserved range: {name}')
    return rows


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False).encode('utf-8')
    return sha256(encoded).hexdigest()


def _prior_details(value):
    member_identities = set()
    scenario_content = set()
    xml_content = set()

    def visit(item):
        if isinstance(item, dict):
            if ('generation_seed' in item or 'generator_family' in item or 'scenario' in item):
                for key in ('identity', 'base_cluster', 'source_member_identity'):
                    if isinstance(item.get(key), str):
                        member_identities.add(item[key])
            if isinstance(item.get('scenario'), dict):
                scenario_content.add(_digest(item['scenario']))
            if isinstance(item.get('xml'), str):
                xml_content.add(sha256(item['xml'].encode('utf-8')).hexdigest())
            for key, child in item.items():
                if key not in ('source_text', 'runtime_source_text', 'source_xml', 'state_dict'):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return {
        'member_identities': sorted(member_identities),
        'scenario_content_sha256': sorted(scenario_content),
        'xml_content_sha256': sorted(xml_content),
    }


def prior_exposure():
    """Project every declared prior/reserved inventory without reading outcomes."""
    by_path = {}
    for projection in files.exposure_sources():
        path = PROJECT_ROOT / projection['path']
        value = files.read(path)
        by_path[projection['path']] = {**projection, **_prior_details(value)}
    for pattern in ('issue-76-*/plan.json', 'issue-77-*/plan.json'):
        for path in sorted((PROJECT_ROOT / '.local-artifacts').glob(pattern)):
            if path == ROOT / 'plan.json':
                continue
            relative = str(path.relative_to(PROJECT_ROOT))
            if relative in by_path:
                continue
            value = files.read(path)
            if (relative.startswith('.local-artifacts/issue-77-')
                    and not (isinstance(value.get('members'), list) and isinstance(value.get('branches'), list))):
                continue  # issue-77 train/diagnostic/eval plans quote membership; not prior exposure
            # Sibling plans embed their own prior_exposure projections; project only
            # the plan's direct membership/scenarios, never transitively embedded priors.
            if isinstance(value, dict):
                value = {k: v for k, v in value.items() if k != 'prior_exposure'}
            by_path[relative] = {
                'path': relative,
                **files.exposure_projection(value),
                **_prior_details(value),
            }
    return [by_path[path] for path in sorted(by_path)]


def audit_disjointness(members, prior):
    prior_seeds = {seed for entry in prior for seed in entry['generation_or_reserved_seeds']}
    prior_members = {identity for entry in prior for identity in entry.get('member_identities', ())}
    prior_scenarios = {identity for entry in prior for identity in entry['scenario_identities']}
    prior_scenario_content = {digest for entry in prior for digest in entry.get('scenario_content_sha256', ())}
    prior_xml_content = {digest for entry in prior for digest in entry.get('xml_content_sha256', ())}
    generation = {member['generation_seed'] for member in members}
    engine = {member['engine_seed'] for member in members}
    identities = {member['identity'] for member in members} | {member['base_cluster'] for member in members}
    scenario_identities = set()
    scenario_content = set()
    xml_content = set()
    for member in members:
        projected = set(files.exposure_projection(member['scenario'])['scenario_identities'])
        scenario_digest = _digest(member['scenario'])
        xml_digest = sha256(member['xml'].encode('utf-8')).hexdigest()
        if projected & scenario_identities or scenario_digest in scenario_content or xml_digest in xml_content:
            raise ValueError('issue-77 N2-normal generated scenarios overlap within the prospective inventory')
        scenario_identities.update(projected)
        scenario_content.add(scenario_digest)
        xml_content.add(xml_digest)
    collisions = {
        'reserved_seed': (generation | engine) & prior_seeds,
        'lineage_identity': identities & prior_members,
        'scenario_identity': scenario_identities & prior_scenarios,
        'scenario_content': scenario_content & prior_scenario_content,
        'xml_content': xml_content & prior_xml_content,
    }
    populated = {key: sorted(value) for key, value in collisions.items() if value}
    if populated:
        raise ValueError(f'issue-77 N2-normal lineage/scenario overlaps prior inventory: {populated}')
    return {
        'prior_inventory_count': len(prior),
        'audited_generation_seeds': len(generation),
        'audited_engine_seeds': len(engine),
        'audited_lineage_identities': len(identities),
        'audited_scenario_identities': len(scenario_identities),
        'audited_scenario_content_digests': len(scenario_content),
        'audited_xml_content_digests': len(xml_content),
        'collisions': {},
        'passed': True,
    }


def template_sources(path=NOVELTY_INVENTORY):
    """Bind the two metadata-selected normal templates from the frozen novelty inventory."""
    path = Path(path)
    try:
        inventory = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError('the frozen issue-76 novelty inventory must exist before the N2 freeze') from error
    if inventory.get('schema') != 'issue_76_novelty_inventory_v1':
        raise ValueError('unexpected novelty inventory schema')
    bound = {}
    for family in FAMILIES:
        matches = [entry for entry in inventory['templates']
                   if entry['novelty_level'] == NOVELTY_LEVEL and entry['family'] == family]
        if len(matches) != 1:
            raise ValueError(f'novelty inventory does not hold exactly one N2 template for {family}')
        entry = matches[0]
        if entry['template'] != EXPECTED_TEMPLATES[family] or entry['static_slot_action_fit'] is not True:
            raise ValueError(f'N2 template for {family} differs from the plan-selected static-fit path')
        bound[family] = entry
    return bound


def bind_scenarios(members, prior, output_root=ROOT, inventory_path=NOVELTY_INVENTORY):
    templates = template_sources(inventory_path)
    allowed_birds = set()
    for entry in templates.values():
        template_path = Path(entry['template'])
        if not template_path.is_absolute():
            template_path = PROJECT_ROOT / 'tasks' / 'task_templates' / template_path
        template_xml = template_path.read_bytes()
        template_tree = ET.fromstring(template_xml.replace(b'encoding=\"utf-16\"', b'encoding=\"utf-8\"', 1))
        allowed_birds.update(bird.attrib['type'] for bird in template_tree.findall('./Birds/Bird'))
    bound = []
    for member in members:
        member = deepcopy(member)
        member['template'] = deepcopy(templates[member['generator_family']])
        generated, scenario = files.materialize(
            member, member['template'], Path(output_root) / 'authorities' / member['identity'])
        tree = ET.fromstring(generated.xml_content)
        slots = sorted(node.attrib['scenarioObjectId'] for node in tree.iter()
                       if 'scenarioObjectId' in node.attrib)
        if (not set(slots).issubset(VOCABULARY)
                or not {bird.attrib['type'] for bird in tree.findall('./Birds/Bird')}.issubset(allowed_birds)):
            raise ValueError('issue-77 N2-normal materialization expanded slots or birds outside the template-declared types')
        xml = generated.xml_content.decode('utf-8')
        member.update({
            'scenario': scenario.to_dict(),
            'xml': xml,
            'generated_xml_identity': 'xml-sha256:' + sha256(xml.encode('utf-8')).hexdigest(),
            'generated_slots': slots,
        })
        bound.append(member)
    audit = audit_disjointness(bound, prior)
    return bound, {
        'template_source_inventory': str(Path(inventory_path)),
        'template_source_inventory_schema': 'issue_76_novelty_inventory_v1',
        'template_declared_bird_types': sorted(allowed_birds),
        'audit': audit,
    }


def candidate_actions():
    angular = [asdict(SlingshotAction(round(-80 * math.cos(math.radians(angle))),
                                    round(80 * math.sin(math.radians(angle))), 0))
               for angle in range(5, 83, 7)]
    return [{'drag_x': -80, 'drag_y': 10, 'tap_time_ms': 0, 'release_time_ms': 1000},
            *[{**action, 'release_time_ms': 1000} for action in angular]]


def input_variants():
    return [
        {'identity': 'legacy-single', 'input_correction': 'legacy', 'history_mode': 'single',
         'native_steps': [0]},
        {'identity': 'corrected-single', 'input_correction': 'corrected', 'history_mode': 'single',
         'native_steps': [0]},
        {'identity': 'corrected-history', 'input_correction': 'corrected', 'history_mode': 'history',
         'native_steps': [-100, -50, 0]},
    ]


def branch_assignments(members):
    variants = input_variants()
    rows = []
    for member in members:
        for candidate_ordinal, action in enumerate(candidate_actions()):
            rows.append({
                'identity': member['identity'] + f'-a{candidate_ordinal:02}',
                'ordinal': len(rows) + 1,
                'source_member_identity': member['identity'],
                'candidate_ordinal': candidate_ordinal,
                'original_action_reference': candidate_ordinal == 0,
                'action': dict(action),
                'input_variants': [dict(variant) for variant in variants],
                'exposure_role': member['exposure_role'],
                'fit_partition': member['fit_partition'],
            })
    return rows


def prerequisite_validation(path=PREREQUISITE):
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError('the v4 shared-history smoke validation.json must exist before freezing') from error
    if (value.get('plan_identity') != 'issue-76-shared-history-smoke-v4'
            or value.get('validated') is not True
            or value.get('individual_timing_and_rgb_passed') is not True):
        raise ValueError('the validated v4 individual timing and RGB smoke fixture must pass first')
    return value


def freeze_source_text(project_root=PROJECT_ROOT, sources=SOURCES):
    frozen = {}
    for name in sources:
        path = Path(project_root) / name
        if not path.is_file():
            raise ValueError(f'mandatory issue-77 N2-normal frozen source is missing: {name}')
        frozen[name] = path.read_text()
    return frozen


def player_binding(path=PLAYER_PLAN):
    path = Path(path)
    try:
        source = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError('the frozen v4 player plan must exist before the issue-77 N2-normal freeze') from error
    player_source = source.get('player_source')
    runtime_source = source.get('runtime_source_text')
    runtime_protocol = 'Assets/Scripts/CanonicalCapture/PhysicsCaptureProtocol.cs'
    if (source.get('identity') != 'issue-76-shared-history-smoke-v4'
            or not source.get('player_build')
            or not isinstance(player_source, dict)
            or player_source.get('schema') != 'issue_76_shared_history_player_source_v1'
            or not isinstance(runtime_source, dict)
            or runtime_protocol not in runtime_source
            or 'RenderCanonicalRgb' not in runtime_source[runtime_protocol]):
        raise ValueError('the v4 plan lacks its recorded cached-player identity/runtime source binding')
    repo_protocol = (PROJECT_ROOT /
                     'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs').read_text()
    if 'ScreenCapture.CaptureScreenshotAsTexture()' not in repo_protocol:
        raise ValueError('the verified repo/cached-player request-72 divergence is no longer present')
    return {
        'source_plan_path': str(path),
        'source_plan_identity': source['identity'],
        'parent_plan_identity': source.get('parent_plan_identity'),
        'player_build_origin': source['player_build'],
        'player_source': player_source,
        'runtime_source_text': runtime_source,
        'runtime_source_sha256': _digest(runtime_source),
        'bound_runtime_protocol': runtime_protocol,
        'verified_request_72_divergence': {
            'repo_source': 'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs:479-504',
            'repo_behavior': 'request 72 uses ScreenCapture.CaptureScreenshotAsTexture',
            'cached_source': runtime_protocol + ':request-72 handler',
            'cached_behavior': 'request 72 uses PhysicsCaptureV2AlignedObservationRecorder.RenderCanonicalRgb',
            'binding_decision': 'The study binds the cached frozen v4 player, not the divergent repo file.',
        },
    }


def role_contract():
    return {
        'controller_only_rule': 'not used in this evaluation-only campaign; no ordinal supervises anything',
        'controller_only_ordinals': [],
        'held_out_evaluation_ordinals': [row['ordinal'] for row in lineage_assignments()],
        'predictor_train_ordinals': [],
        'calibration_role_states_are_development_scoring_only': True,
        'lineage_roles_are_disjoint_and_outcome_independent': True,
        'evaluation_only_corpus_never_fitted': True,
    }


def make_plan(validation_path=PREREQUISITE, player_plan_path=PLAYER_PLAN):
    fixture = prerequisite_validation(validation_path)
    prior = prior_exposure()
    members, scenario_binding = bind_scenarios(lineage_assignments(), prior)
    branches = branch_assignments(members)
    role_counts = Counter(member['study_role'] for member in members)
    frozen_sources = freeze_source_text()
    player = player_binding(player_plan_path)
    return {
        'schema': 'issue_77_n2n_plan_v1',
        'identity': IDENTITY,
        'stage': 'N2 appearance-novelty normal-side capture (type010102; type010101 reuses the R3 corpus)',
        'protocol': PROTOCOL,
        'members': members,
        'branches': branches,
        'families': list(FAMILIES),
        'novelty_level': NOVELTY_LEVEL,
        'counts': {
            'lineages': len(members),
            'physical_branch_executions': len(branches),
            'paired_input_views_per_execution': 3,
            'dispatch_contract': '104 physical branch executions; each execution produces three paired input views; never 312 dispatches',
            'study_roles': dict(role_counts),
        },
        'scenario_binding': scenario_binding,
        'prior_exposure': prior,
        'action_contract': {
            'actions': candidate_actions(),
            'one_shot_per_branch': True,
            'tap_actions_authorized': False,
            'requested_release_time_ms': 1000,
            'recovered_release_semantics': 'fixed one-second delay before launch',
            'recovered_fixed_release_delay_seconds': 1.0,
        },
        'input_variant_contract': {
            'variants': input_variants(),
            'same_three_variants_required_per_branch': True,
            'corrected_history_native_steps': [-100, -50, 0],
            'paired_view_failure_rule': 'a missing required view invalidates the physical branch for every downstream use',
        },
        'endpoint_and_label_contract': {
            'observed_window_seconds': 12,
            'decision_fixed_step': 30000,
            'stop_kinds': ['native_clear', 'native_fail', 'stable_without_clear', 'right_censored'],
            'stable_and_right_censored_are_not_converted_to_failures': True,
        },
        'role_contract': role_contract(),
        'limits': {
            'workers': 8,
            'worker_cpu_rss_mib': 4096,
            'aggregate_cpu_rss_mib': 32768,
            'combined_collection_seconds': 302400,
            'collection_wall_seconds': 86400,
            'attempt_seconds': 1200,
            'shot_manifest_seconds': 180,
            'shot_manifest_deadline_seconds': 180,
            'shot_seconds': 180,
            'history_ready_seconds': 90,
            'technical_retries': 0,
            'minimum_free_bytes': 256 * 2**30,
            'artifact_bytes': 150 * 2**30,
            'decision_fixed_step': 30000,
            'native_step_seconds': 0.0004,
            'rgb_stride_native_steps': 50,
            'native_steps_per_shot_max': 30000,
            'predecision_frames': 3,
            'rgb_frames_per_shot_max': 601,
        },
        'seed_contract': {
            'generation_seed_formula': '764500000 + lineage ordinal (1..8)',
            'engine_seed_formula': '764600000 + lineage ordinal (1..8)',
            'reserved_ranges': [{'name': name, 'low': low, 'high': high}
                                for name, low, high in RESERVED_RANGES],
            'all_ranges_disjoint': True,
        },
        'prerequisite_validation_path': str(Path(validation_path)),
        'prerequisite_validation': fixture,
        'player_binding': player,
        'unsupported_cells_recorded_not_dropped': True,
        'small_project_mode_no_authorization_gates': True,
        'source_freeze': {
            'sources': list(SOURCES),
            'protocol_path': PROTOCOL,
            'mandatory_and_fail_closed': True,
            'partial_module_notes': PARTIAL_SOURCE_NOTES,
        },
        'source_text': frozen_sources,
    }


def make_inventory(plan):
    return {
        'schema': 'issue_77_n2n_inventory_v1',
        'identity': IDENTITY,
        'plan_identity': plan['identity'],
        'members': plan['members'],
        'branches': plan['branches'],
        'counts': plan['counts'],
    }


def write_plan(root=ROOT, validation_path=PREREQUISITE):
    root = Path(root)
    if root.exists():
        raise ValueError('issue-77 N2-normal plan already exists; preserve the frozen prospective inventory')
    plan = make_plan(validation_path)
    inventory = make_inventory(plan)
    root.mkdir(parents=True)
    (root / 'plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    (root / 'inventory.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    return plan


def load_plan(root=ROOT):
    plan = json.loads((Path(root) / 'plan.json').read_text())
    if plan.get('identity') != IDENTITY:
        raise ValueError('issue-77 N2-normal plan identity differs from the authorized frozen study')
    frozen_sources = plan.get('source_freeze', {}).get('sources')
    if (set(frozen_sources or ()) != set(SOURCES)
            or set(frozen_sources or ()) != set(plan.get('source_text', {}))
            or plan.get('source_freeze', {}).get('mandatory_and_fail_closed') is not True):
        raise ValueError('issue-77 N2-normal source manifest differs from its frozen source_text')
    for name, source in plan['source_text'].items():
        if (PROJECT_ROOT / name).read_text() != source:
            raise ValueError(f'issue-77 N2-normal frozen source changed: {name}')
    return plan


def summary(plan):
    return {
        'identity': plan['identity'],
        'lineages': plan['counts']['lineages'],
        'scenario_bound_lineages': sum('scenario' in member for member in plan['members']),
        'physical_branch_executions': plan['counts']['physical_branch_executions'],
        'paired_input_views_per_execution': plan['counts']['paired_input_views_per_execution'],
        'study_roles': plan['counts']['study_roles'],
        'frozen_sources': len(plan['source_text']),
        'families': plan['families'],
        'novelty_level': plan['novelty_level'],
        'write_requested': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-plan', action='store_true',
                        help='write only plan.json and inventory.json beneath the frozen output root')
    args = parser.parse_args()
    plan = write_plan() if args.write_plan else make_plan()
    value = summary(plan)
    value['write_requested'] = args.write_plan
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
