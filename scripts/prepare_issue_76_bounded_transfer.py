"""Freeze the authorized bounded input-corrected transfer study; never capture or fit."""
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
from scripts import issue_76_bounded_transfer_frontier as frontier
from scripts import run_issue_76_development as previous
from world_model.planning.gameplay import SlingshotAction
from world_model.training.native_history_data import VOCABULARY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'issue-76-bounded-transfer-v1'
ROOT = PROJECT_ROOT / '.local-artifacts' / IDENTITY
PREREQUISITE = PROJECT_ROOT / '.local-artifacts/issue-76-shared-history-smoke-v4/validation.json'
PLAYER_PLAN = PROJECT_ROOT / '.local-artifacts/issue-76-shared-history-smoke-v4/plan.json'
TEMPLATE_PLAN = previous.DEVELOPMENT / 'plan.json'
CONDITIONED_REPORT = PROJECT_ROOT / '.local-artifacts/issue-76-conditioned-event-engineering-v1/report.json'
CHECKPOINT_ROOT = PROJECT_ROOT / '.local-artifacts/issue-76-balanced-native-v2/checkpoints'
PROTOCOL = 'docs/issue-76-bounded-transfer-protocol.md'
FAMILIES = ('type010101', 'type010105')
PREDICTOR_SEEDS = (760930001, 760930002, 760930003)
SOURCES = (
    'scripts/prepare_issue_76_bounded_transfer.py',
    'tests/test_issue_76_bounded_transfer_inventory.py',
    PROTOCOL,
    'scripts/issue_76_bounded_transfer_capture.py',
    'scripts/issue_76_bounded_transfer_metrics.py',
    'scripts/issue_76_bounded_transfer_frontier.py',
    'tests/test_issue_76_bounded_transfer_capture.py',
    'tests/test_issue_76_bounded_transfer_frontier.py',
    'docs/issue-76-bounded-transfer-capture-mechanism.md',
    'scripts/run_issue_76_shared_history_smoke.py',
    'scripts/issue_76_shared_history_capture.py',
    'scripts/issue_76_display_start.py',
    'scripts/prepare_issue_76_shared_development.py',
    'scripts/issue_76_expansion.py',
    'scripts/run_issue_76_development.py',
    'scripts/run_issue_76_conditioned_event.py',
    'scripts/run_issue_76_balanced_event.py',
    'scripts/run_issue_76_event_model.py',
    'scripts/run_issue_76_native_refit.py',
    'scripts/run_issue_76_balanced_native.py',
    'scripts/issue_76_matched_batches.py',
    'scripts/issue_76_event_targets.py',
    'scripts/issue_76_native_outcomes.py',
    'scripts/native_segment_trace.py',
    'scripts/canonical_native_trace.py',
    'scripts/issue_76_scaled_fit.py',
    'scripts/issue_76_fit_budget.py',
    'world_model/training/action_event_readout.py',
    'world_model/training/event_ranking.py',
    'world_model/training/event_trajectory_ranking.py',
    'world_model/training/native_history_data.py',
    'world_model/training/native_history_fit.py',
    'world_model/training/native_history_model.py',
    'world_model/training/cnn_hybrid.py',
    'scripts/cohort_v2_scenarios.py',
    'tasks/task_generator/canonical_materialization.py',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/ObservationCaptureProtocol.cs',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureV2EngineProtocol.cs',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicalSnapshotRuntime.cs',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicalCaptureModels.cs',
)
PARTIAL_SOURCE_NOTES = {
    'scripts/run_issue_76_event_model.py': 'Frozen in full; this study directly adopts _prior_scores and metric grouping only.',
    'scripts/run_issue_76_native_refit.py': 'Frozen in full; this study directly adopts checkpoint naming/loading and generator(seed, update).',
    'scripts/issue_76_matched_batches.py': 'Frozen in full for the matched batch/RNG scheduling contract.',
    'tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs':
        'Frozen in full only to bind the verified repo/cached-player request-72 divergence.',
}


def lineage_assignments():
    """Return the outcome-independent, deliberately non-contiguous family ranks."""
    rows = []
    ranks = (
        *((rank, 'training', 'propagator_and_readout', 'readout_train') for rank in range(20)),
        *((rank, 'training', 'controller', 'selector_train') for rank in range(75, 85)),
        *((rank, 'model_selection', None, 'held_out_model_selection') for rank in range(120, 140)),
    )
    for family in FAMILIES:
        for family_rank, exposure_role, fit_partition, study_role in ranks:
            ordinal = len(rows) + 1
            identity = f'issue-76-bounded-transfer-{ordinal:03}'
            rows.append({
                'identity': identity,
                'base_cluster': identity,
                'ordinal': ordinal,
                'generation_seed': 761900000 + ordinal,
                'engine_seed': 762000000 + ordinal,
                'generator_family': family,
                'novelty_level': 0,
                'exposure_role': exposure_role,
                'fit_partition': fit_partition,
                'study_role': study_role,
                'family_rank': family_rank,
            })
    generation = {row['generation_seed'] for row in rows}
    engine = {row['engine_seed'] for row in rows}
    shared_development = set(range(761700001, 761700701)) | set(range(761800001, 761800701))
    if generation & engine or (generation | engine) & shared_development:
        raise ValueError('bounded-transfer seeds overlap each other or the parked section 8 inventory')
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
    for path in sorted((PROJECT_ROOT / '.local-artifacts').glob('issue-76-*/plan.json')):
        if path == ROOT / 'plan.json':
            continue
        relative = str(path.relative_to(PROJECT_ROOT))
        value = files.read(path)
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
            raise ValueError('bounded-transfer generated scenarios overlap within the prospective inventory')
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
        raise ValueError(f'bounded-transfer lineage/scenario overlaps prior inventory: {populated}')
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


def bind_scenarios(members, prior, output_root=ROOT, template_plan=TEMPLATE_PLAN):
    source = files.read(Path(template_plan))
    templates = {member['generator_family']: member['template'] for member in source['members']}
    if set(templates) < set(FAMILIES):
        raise ValueError('bounded-transfer template source lacks an authorized family')
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
                or any(bird.attrib['type'] != 'BirdRed' for bird in tree.findall('./Birds/Bird'))):
            raise ValueError('bounded-transfer materialization expanded slots or unsupported bird actions')
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
        'template_source_plan': str(Path(template_plan)),
        'template_source_plan_identity': source['identity'],
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
            raise ValueError(f'mandatory bounded-transfer frozen source is missing: {name}')
        frozen[name] = path.read_text()
    return frozen


def player_binding(path=PLAYER_PLAN):
    path = Path(path)
    try:
        source = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError('the frozen v4 player plan must exist before bounded-transfer freeze') from error
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


def _checkpoint_paths(kind):
    paths = [CHECKPOINT_ROOT / f'predictor-{kind}-{seed}.pt' for seed in PREDICTOR_SEEDS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f'bounded-transfer predictor checkpoint is missing: {missing[0]}')
    return [str(path.relative_to(PROJECT_ROOT)) for path in paths]


def arm_contract(report_path=CONDITIONED_REPORT):
    report_path = Path(report_path)
    try:
        report = json.loads(report_path.read_text())
    except FileNotFoundError as error:
        raise ValueError('conditioned-event report is required to pin fixed policies') from error
    hybrid_fixed = report.get('calibration', {}).get('hybrid', {}).get('selected_fixed_pair')
    pure_fixed = report.get('calibration', {}).get('pure', {}).get('selected_fixed_pair')
    if hybrid_fixed != 'fixed-50-macro' or pure_fixed != 'fixed-250-continuous':
        raise ValueError('calibration-selected fixed policy identities differ from the authorized contract')
    variants = [variant['identity'] for variant in input_variants()]
    model_arms = (
        ('hybrid-adaptive', 'hybrid', 'adaptive', None, True),
        ('hybrid-fixed', 'hybrid', 'fixed', hybrid_fixed, False),
        ('pure-fixed-250-continuous', 'pure', 'fixed', pure_fixed, False),
    )
    cells = []
    for variant_index, variant in enumerate(variants):
        for arm_index, (arm, predictor, policy, fixed_identity, trains_selector) in enumerate(model_arms):
            by_seed = []
            for seed_index, seed in enumerate(PREDICTOR_SEEDS):
                offset = variant_index * 100 + arm_index * 10 + seed_index
                by_seed.append({
                    'predictor_seed': seed,
                    'readout_initialization_seed': 763100000 + offset,
                    'selector_initialization_seed': 763200000 + offset if trains_selector else None,
                    'selector_update_rng_seed_formula':
                        f'{763200000 + offset} + 1009 * update' if trains_selector else None,
                })
            cells.append({
                'arm': arm,
                'input_variant': variant,
                'predictor_kind': predictor,
                'policy': policy,
                'fixed_policy_identity': fixed_identity,
                'selection': 'trained adaptive selector' if trains_selector else 'pinned fixed policy',
                'input_invariant': False,
                'conditioned_readout_retrained_from_scratch': True,
                'selector_retrained_from_scratch': trains_selector,
                'readout_updates': 9000,
                'selector_updates': 2000 if trains_selector else 0,
                'readout_updates_per_seed': 9000,
                'selector_updates_per_seed': 2000 if trains_selector else 0,
                'readout_batch_schedule':
                    'all valid readout-TRAIN records in frozen lineage/branch order; pair cycles by update modulo pair count',
                'selector_batch_rng': ('torch.Generator.manual_seed(selector initialization seed + 1009 * update)'
                                       if trains_selector else None),
                'training_by_seed': by_seed,
            })
    for variant in variants:
        cells.extend((
            {
                'arm': 'training_family_prior',
                'input_variant': variant,
                'predictor_kind': None,
                'policy': 'no-model',
                'selection': 'training-label family/candidate prior',
                'input_invariant': True,
                'input_invariant_note':
                    'Computed once from TRAIN labels and duplicated identically across all three input-variant columns.',
                'conditioned_readout_retrained_from_scratch': False,
                'selector_retrained_from_scratch': False,
                'readout_updates': 0,
                'selector_updates': 0,
                'readout_updates_per_seed': 0,
                'selector_updates_per_seed': 0,
            },
            {
                'arm': 'candidate-0-original',
                'input_variant': variant,
                'predictor_kind': None,
                'policy': 'candidate_ordinal_0',
                'selection': 'pinned original candidate',
                'input_invariant': True,
                'input_invariant_note':
                    'Computed once from the frozen candidate list and duplicated identically across all three input-variant columns.',
                'conditioned_readout_retrained_from_scratch': False,
                'selector_retrained_from_scratch': False,
                'readout_updates': 0,
                'selector_updates': 0,
                'readout_updates_per_seed': 0,
                'selector_updates_per_seed': 0,
            },
        ))
    return {
        'arms': ['hybrid-adaptive', 'hybrid-fixed', 'pure-fixed-250-continuous',
                 'training_family_prior', 'candidate-0-original'],
        'input_variants': variants,
        'matrix': cells,
        'matrix_cells': len(cells),
        'predictor_seeds': list(PREDICTOR_SEEDS),
        'predictor_checkpoints': {
            'hybrid': _checkpoint_paths('hybrid'),
            'pure': _checkpoint_paths('pure'),
        },
        'hybrid_fixed_policy_identity': hybrid_fixed,
        'hybrid_fixed_policy_evidence':
            '.local-artifacts/issue-76-conditioned-event-engineering-v1/report.json:593',
        'pure_fixed_policy_identity': pure_fixed,
        'pure_fixed_policy_evidence':
            '.local-artifacts/issue-76-conditioned-event-engineering-v1/report.json:1439',
        'no_model_identity': 'training_family_prior',
        'no_model_smoothing_rule': {
            'grouping': 'generator family x candidate ordinal; valid readout-TRAIN labels only',
            'initial_clear_hits': 1,
            'initial_total': 2,
            'score': '(1 + native_clear_count) / (2 + valid_training_count)',
            'source': 'scripts/run_issue_76_event_model.py:474-483 (_prior_scores)',
        },
        'candidate_0_identity': 'candidate-0-original',
        'matched_exposure': ('all predictor cells retrain one conditioned readout per input variant and seed; '
                             'only hybrid-adaptive trains a selector; fixed controls use pinned policies'),
        'dynamics_optimizer_updates': 0,
    }


def make_plan(validation_path=PREREQUISITE, player_plan_path=PLAYER_PLAN):
    fixture = prerequisite_validation(validation_path)
    prior = prior_exposure()
    members, scenario_binding = bind_scenarios(lineage_assignments(), prior)
    branches = branch_assignments(members)
    role_counts = Counter(member['study_role'] for member in members)
    frozen_sources = freeze_source_text()
    arms = arm_contract()
    player = player_binding(player_plan_path)
    return {
        'schema': 'issue_76_bounded_transfer_plan_v1',
        'identity': IDENTITY,
        'members': members,
        'branches': branches,
        'families': list(FAMILIES),
        'counts': {
            'lineages': len(members),
            'physical_branch_executions': len(branches),
            'paired_input_views_per_execution': 3,
            'dispatch_contract': '1,300 physical branch executions; each execution produces three paired input views; never 3,900 dispatches',
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
            'paired_view_failure_rule': 'a missing required view invalidates the physical branch for every arm',
        },
        'arm_contract': arms,
        'metric_binding': {
            'adapter_source': 'scripts/issue_76_bounded_transfer_metrics.py',
            'adapter_entrypoint': 'adapt_retained_results',
            'output_schema': 'issue_76_bounded_transfer_metric_inputs_v1',
            'consumer': 'scripts/run_issue_76_event_model.py:443-470 (_metrics)',
            'consumer_rows_field': 'rows',
            'consumer_labels_field': 'label_indices',
            'group_index': 'held-out base-lineage ordinal',
            'valid_rule': ('bounded capture complete with no failure, successful zero-retry receipt, '
                           'exactly one segment, and observed_window_valid=true'),
            'paired_ranking_admissible_rule':
                'every branch in the lineage is operationally valid with all three paired views present',
            'branch_paired_ranking_admissible_rule':
                'the branch is operationally valid and all three paired views are present',
            'missing_view_rule':
                'the branch and its entire base lineage are inadmissible for every arm',
            'informative_group_rule':
                'all scheduled branches admissible and at least one observed native_clear among the 13 branches',
            'labels': ['native_clear', 'native_fail', 'stable_without_clear', 'right_censored'],
            'stable_and_censored_preserved_as_typed_labels': True,
            'later_clear_label_allowed': False,
            'frontier_source': 'scripts/issue_76_bounded_transfer_frontier.py',
            'frontier_entrypoint': 'physical_timeline',
            'frontier_contract_entrypoint': 'frozen_contract',
            'frontier_entrypoints': {
                'physical_timeline': 'physical_timeline',
                'checkpoint_availability': 'checkpoint_availability',
                'settled_state': 'settled_state',
                'regime_segments': 'regime_segments',
                'auec': 'area_under_error_curve',
                'equal_cumulative_macs': 'error_at_equal_cumulative_macs',
                'compute_ledger': 'compute_ledger',
                'controller_trace': 'controller_trace',
                'physical_plausibility': 'physical_plausibility',
            },
            'frontier_contract': frontier.frozen_contract(),
        },
        'endpoint_and_label_contract': {
            'predictor_endpoint_native_steps': 11250,
            'predictor_endpoint_seconds': 4.5,
            'predictor_endpoint_role': 'latent feature supplied to the conditioned event readout only',
            'observed_outcome_window_seconds': 12,
            'stop_kinds': ['native_clear', 'native_fail', 'stable_without_clear', 'right_censored'],
            'later_clear_by_deadline_label': None,
            'stable_and_right_censored_are_not_converted_to_failures': True,
        },
        'decision_rule': {
            'minimum_informative_held_out_groups': 10,
            'corrected_single_hybrid_minimum_clear_hit_gain_over_legacy_single_hybrid': 1,
            'corrected_history_hybrid_minimum_clear_hit_gain_over_pure_fixed': 1,
            'corrected_history_hybrid_minimum_clear_hit_gain_over_hybrid_fixed': 1,
            'corrected_history_hybrid_minimum_clear_hit_gain_over_training_family_prior': 1,
            'corrected_history_not_worse_than_corrected_single': True,
            'minimum_second_mode_fraction': 0.05,
            'minimum_second_horizon_fraction': 0.05,
            'maximum_hybrid_to_pure_linear_mac_ratio': 1.1,
            'coverage_insufficient_below_floor': True,
        },
        'role_contract': {
            'readout_train_family_ranks': list(range(20)),
            'selector_train_family_ranks': list(range(75, 85)),
            'held_out_model_selection_family_ranks': list(range(120, 140)),
            'controller_stop_labels_excluded_from_readout_fit': True,
            'lineage_roles_are_disjoint_and_outcome_independent': True,
        },
        'limits': {
            'workers': 8,
            'worker_cpu_rss_mib': 4096,
            'aggregate_cpu_rss_mib': 32768,
            'combined_collection_seconds': 345600,
            'collection_wall_seconds': 345600,
            'attempt_seconds': 420,
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
            'generation_seed_formula': '761900000 + lineage ordinal (1..100)',
            'engine_seed_formula': '762000000 + lineage ordinal (1..100)',
            'parked_section_8_generation_range': [761700001, 761700700],
            'parked_section_8_engine_range': [761800001, 761800700],
            'all_ranges_disjoint': True,
        },
        'prerequisite_validation_path': str(Path(validation_path)),
        'prerequisite_validation': fixture,
        'player_binding': player,
        'capture_execution_authorized': False,
        'model_training_authorized': False,
        'fresh_access': False,
        'advancement_authorized': False,
        'parked_section_8_note': 'The 700-lineage/9,100-capture section 8 shared-development draft remains parked.',
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
        'schema': 'issue_76_bounded_transfer_inventory_v1',
        'identity': IDENTITY,
        'plan_identity': plan['identity'],
        'members': plan['members'],
        'branches': plan['branches'],
        'counts': plan['counts'],
        'capture_execution_authorized': False,
        'model_training_authorized': False,
        'fresh_access': False,
        'advancement_authorized': False,
    }


def write_plan(root=ROOT, validation_path=PREREQUISITE):
    root = Path(root)
    if root.exists():
        raise ValueError('bounded-transfer plan already exists; preserve the frozen prospective inventory')
    plan = make_plan(validation_path)
    inventory = make_inventory(plan)
    root.mkdir(parents=True)
    (root / 'plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    (root / 'inventory.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    return plan


def load_plan(root=ROOT):
    plan = json.loads((Path(root) / 'plan.json').read_text())
    if plan.get('identity') != IDENTITY:
        raise ValueError('bounded-transfer plan identity differs from the authorized frozen study')
    frozen_sources = plan.get('source_freeze', {}).get('sources')
    if (set(frozen_sources or ()) != set(SOURCES)
            or set(frozen_sources or ()) != set(plan.get('source_text', {}))
            or plan.get('source_freeze', {}).get('mandatory_and_fail_closed') is not True):
        raise ValueError('bounded-transfer source manifest differs from its frozen source_text')
    for name, source in plan['source_text'].items():
        if (PROJECT_ROOT / name).read_text() != source:
            raise ValueError(f'bounded-transfer frozen source changed: {name}')
    return plan


def summary(plan):
    return {
        'identity': plan['identity'],
        'lineages': plan['counts']['lineages'],
        'scenario_bound_lineages': sum('scenario' in member for member in plan['members']),
        'physical_branch_executions': plan['counts']['physical_branch_executions'],
        'paired_input_views_per_execution': plan['counts']['paired_input_views_per_execution'],
        'arm_variant_cells': plan['arm_contract']['matrix_cells'],
        'frozen_sources': len(plan['source_text']),
        'families': plan['families'],
        'capture_execution_authorized': plan['capture_execution_authorized'],
        'model_training_authorized': plan['model_training_authorized'],
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
