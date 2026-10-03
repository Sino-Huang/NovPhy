"""Issue #110 design: per-cell readiness for all 45 NovPhy cells and the frozen capture membership.

Zero engine seconds. Nothing is captured, fitted or scored. The runner

1. verifies the force laws of representation contract v2 (``world_model/data/novelty_contract.py``)
   against the recovered original NovPhy scripts/prefabs and confirms that the capture player
   (the one every #77/#109 campaign used) contains each novelty class and prefab;
2. sweeps the level generator (outcome-free seeds, nothing kept) to measure the share of generated
   levels whose authored slots fit the frozen 18-slot contract and contract v2, per template;
3. derives a readiness row for each of the 45 scenario x novelty cells (5 scenarios x 9 levels),
   under the frozen v1 contract and under contract v2, with typed reasons;
4. freezes the capture membership: the 40 upstream normal/novel template pairs, each with paired
   smoke / fit / evaluation rosters (same generation and engine seed on both sides), screened
   outcome-free, seed- and content-disjoint from every prior exposure including #109's cohort;
5. budgets the campaign per roster prefix from the measured #109 capture throughput and declares
   the post-capture identifiability and rollout-error protocols.

Modes: ``--dry-run`` (derive, print, write nothing), ``--prepare`` (freeze), ``--validate``
(re-derive and byte-compare). Output: ``.local-artifacts/issue-110-novelty-coverage-v1/``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET

from scripts import prepare_sealed_cohort as sc
from scripts.cohort_v2_scenarios import create_scenario_template_record, materialize_template_bound_level_instance
from scripts.scenario_manifest import BenchmarkCondition
from tasks.task_generator.canonical_materialization import CanonicalMaterializationRequest
from world_model.data import novelty_contract as contract

ROOT = sc.ROOT
IDENTITY = 'issue-110-novelty-coverage-v1'
SCHEMA = 'issue_110_novelty_coverage_plan_v1'
OUTPUT = ROOT / '.local-artifacts' / IDENTITY
VALIDATION_COMMAND = 'python -u -m scripts.prepare_novelty_coverage --validate'
INVENTORY = sc.n1.NOVELTY_INVENTORY
TEMPLATE_ROOT = ROOT / 'tasks/task_templates'
RECOVERED = ROOT / '.local-artifacts/issue-76-canonical-recovery-v1/ExportedProject/Assets'
RECOVERED_SCRIPTS = RECOVERED / 'Scripts/Assembly-CSharp'
RECOVERED_PREFABS = RECOVERED / 'Resources/prefabs/gameworld'
PLAYER_DATA = ROOT / '.local-artifacts/issue-76-shared-history-smoke-v4/player/9001_Data'
PLAYER_USERS = ('issue-77-n1-v1', 'issue-77-n2-appearance-v1', 'issue-109-capture-smoke-v1')
SEALED_PLAN = sc.OUTPUT / 'plan.json'

SCENARIOS = {1: 'single_force', 2: 'multiple_forces', 3: 'rolling', 4: 'falling', 5: 'sliding'}
CATEGORIES = {0: 'normal', 1: 'objects (appearance)', 2: 'agents (fan)', 3: 'actions (stronger turbulence)',
              4: 'interactions (magnet)', 5: 'relations (right slingshot)', 6: 'environments (inverse gravity)',
              7: 'goals (inverted turbulence)', 8: 'events (storm)'}
SWEEP_SEED_BASE = 767_000_000
SWEEP_SEEDS = 200
SPLITS = (('smoke', 1), ('fit', 24), ('evaluation', 24))
SPLIT_OFFSETS = {'smoke': 0, 'fit': 100_000, 'evaluation': 200_000}
GENERATION_SEED_BASE = 768_000_000
ENGINE_SEED_BASE = 768_500_000
PAIR_SEED_BLOCK = 2_000
PREFIXES = (4, 8, 12, 16, 24)
RECOMMENDED_PREFIX = 8
SMOKE_CANDIDATES = ('c05', 'c10')
FIT_SHARE_FLOOR = 0.85
ALLOWED_BIRDS = ('BirdRed',)
SIDES = ('normal', 'novel')
REJECTION_REASONS = ('reserved_seed_collision', 'materialization_error', 'slot_outside_v2_vocabulary',
                     'bird_not_birdred', 'duplicate_xml_content', 'duplicate_scenario_content',
                     'duplicate_scenario_identity')


def log(message):
    print(f'[{IDENTITY}] {message}', flush=True)


# ------------------------------------------------------- engine sources

def _text(path):
    return Path(path).read_text(errors='replace')


def _field(text, name):
    match = re.search(rf'^\s*{re.escape(name)}:\s*(.+?)\s*$', text, re.M)
    if not match:
        raise ValueError(f'serialized field {name} missing')
    return match.group(1)


def _number(text, name):
    return float(_field(text, name))


def _sprite(text):
    return re.search(r'm_Sprite: \{fileID: \d+, guid: ([0-9a-f]+)', text).group(1)


PREFAB_CHECKS = (
    # (prefab, field, contract value)
    ('benchmarknovelties/NonNovelAirTurbulence.prefab', 'turbulenceForce',
     contract.TURBULENCE_FORCE['NonNovelAirTurbulence']),
    ('benchmarknovelties/NovelAirTurbulence.prefab', 'turbulenceForce', contract.TURBULENCE_FORCE['NovelAirTurbulence']),
    ('benchmarknovelties/InverseAirTurbulence.prefab', 'turbulenceForce',
     contract.TURBULENCE_FORCE['InverseAirTurbulence']),
    ('benchmarknovelties/InverseGravity.prefab', 'gravity', contract.INVERSE_GRAVITY[1]),
    ('benchmarknovelties/Magnet.prefab', 'magneticForce', contract.MAGNET_FORCE),
    ('benchmarknovelties/Magnet.prefab', 'magneticFieldRadius', contract.MAGNET_RADIUS),
)
SCRIPT_CHECKS = (
    # (script, literal that must appear verbatim, what it pins)
    ('ABFan.cs', 'private float thrust = 25f;', 'FAN_THRUST'),
    ('ABFan.cs', 'private float windWidth = 2f;', 'FAN_WIND_WIDTH'),
    ('ABFan.cs', 'private float windHeight = 3f;', 'FAN_WIND_HEIGHT'),
    ('ABFan.cs', 'if (num < 0f && num2 < windWidth && num3 < windHeight / 2f)', 'fan zone'),
    ('ABFan.cs', 'rigidBody.AddForce(-base.transform.right * thrust);', 'fan direction'),
    ('ABFan.cs', 'if (ABGameWorld.wasBirdLaunched && currentBird != null)', 'fan acts on the launched bird only'),
    ('AirTurbulence.cs', 'if ((bool)item && item.velocity.magnitude > 0.1f)', 'SPEED_GATE'),
    ('AirTurbulence.cs', 'item.AddForce(base.transform.up * turbulenceForce);', 'turbulence direction'),
    ('AirTurbulence.cs', '(text == "PigSmall") | (text == "PigMedium") | (text == "PigBig") | (text == "Platform")',
     'TURBULENCE_EXCLUDED_KINDS'),
    ('Magnet.cs', 'item.AddForce(vector * magneticForce);', 'magnet attraction'),
    ('Magnet.cs', 'Vector3 vector = -(base.transform.position - item2.transform.position);', 'magnet repulsion'),
    ('Magnet.cs', 'if (collision.GetComponent<ABBlock>()._material == repellingMaterial)', 'magnet polarity'),
    ('Magnet.cs', 'attractingBodiesInsideMagneticField.Remove(collision.rigidbody);', 'removal on collision'),
    ('InverseGravity.cs', 'Physics2D.gravity = new Vector3(0f, gravity, 0f);', 'global gravity'),
    ('InverseGravity.cs', 'aBBird.GetComponent<Rigidbody2D>().gravityScale = -1f;', 'INVERSE_GRAVITY_BIRD_SCALE'),
    ('Storm.cs', 'if (Object.FindObjectsOfType<ABBird>().Length == initialBirdCount - 1)', 'storm onset'),
    ('Storm.cs', 'if ((bool)aBBird && (aBBird.IsFlying || aBBird.IsDying))', 'storm birds'),
    ('Storm.cs', 'if ((bool)aBBlock && aBBlock.getRigidBody().velocity.magnitude > 0.1f)', 'storm blocks'),
    ('ABBird.cs', 'InvokeRepeating("CheckVelocityToDie", 3f, 1f);', 'bird death >= 3 s after its first collision'),
    ('DieOnBirdHitCountPig.cs', 'if (collidedBirdIDs.Count >= numberOfBirdShotsToDie)', 'multi-hit pig'),
    ('ABGameWorld.cs', 'AddNovelty(ABWorldAssets.NOVELTIES[novelty.type], vector7, rotation6, novelty.scaleX, '
     'novelty.scaleY);', 'novelty instantiation with template scale'),
    ('ABGameWorld.cs', 'if (_pigs.Count == 0 && !_isSimulation)', 'clear condition (every level, including 7)'),
    ('ABGameWorld.cs', 'Invoke("ShowLevelFailedBanner", _timeToResetLevel);', 'fail condition: birds exhausted'),
)
PIG_PREFABS = {'BasicSmall': 'characters/pigs/BasicSmall.prefab', 'BasicMedium': 'characters/pigs/BasicMedium.prefab',
               'BasicBig': 'characters/pigs/BasicBig.prefab', 'PinkPig': 'benchmarknovelties/PinkPig.prefab',
               'PinkMediumPig': 'benchmarknovelties/PinkMediumPig.prefab',
               'PinkBigPig': 'benchmarknovelties/PinkBigPig.prefab',
               'BasicBigTwoShots': 'benchmarknovelties/BasicBigTwoShots.prefab',
               'BasicBigFourShots': 'benchmarknovelties/BasicBigFourShots.prefab'}
PLAYER_CLASSES = ('ABFan', 'AirTurbulence', 'Magnet', 'InverseGravity', 'Storm', 'DieOnBirdHitCountPig',
                  'NoveltyData', 'ScenarioObjectIdentity')
PLAYER_PREFABS = ('Fan', 'NonNovelAirTurbulence', 'NovelAirTurbulence', 'InverseAirTurbulence', 'Magnet',
                  'InverseGravity', 'Storm', 'StormEffect', 'PinkPig', 'PinkBigPig', 'PinkCircle', 'PinkRectFat',
                  'BasicBigTwoShots', 'BasicBigFourShots')


def engine_sources():
    """Contract constants vs the recovered original sources, and the capture player's contents."""
    prefab_rows = []
    for name, field, expected in PREFAB_CHECKS:
        observed = _number(_text(RECOVERED_PREFABS / name), field)
        if observed != expected:
            raise ValueError(f'{name} {field} = {observed}, contract says {expected}')
        prefab_rows.append({'prefab': name, 'field': field, 'value': observed})
    storm = _field(_text(RECOVERED_PREFABS / 'benchmarknovelties/Storm.prefab'), 'windForce')
    if storm != '{x: 3, y: 0}' or contract.STORM_WIND != (3.0, 0.0):
        raise ValueError(f'Storm windForce {storm} differs from the contract')
    prefab_rows.append({'prefab': 'benchmarknovelties/Storm.prefab', 'field': 'windForce', 'value': storm})
    for variant in contract.TURBULENCE_FORCE:
        sizes = re.findall(r'm_Size: \{x: ([\d.]+), y: ([\d.]+)\}', _text(RECOVERED_PREFABS / f'benchmarknovelties/{variant}.prefab'))
        if [tuple(map(float, size)) for size in sizes] != [(contract.TURBULENCE_BOX_LOCAL,) * 2] * 2:
            raise ValueError(f'{variant} sprite/collider size differs from TURBULENCE_BOX_LOCAL')
    script_rows = []
    for script, literal, pins in SCRIPT_CHECKS:
        if literal not in _text(RECOVERED_SCRIPTS / script):
            raise ValueError(f'{script} no longer contains {literal!r}')
        script_rows.append({'script': script, 'literal': literal, 'pins': pins})
    sprites = {name: {'sprite_guid': _sprite(_text(RECOVERED_PREFABS / path)),
                      'color': _field(_text(RECOVERED_PREFABS / path), 'm_Color')}
               for name, path in (('NonNovelAirTurbulence', 'benchmarknovelties/NonNovelAirTurbulence.prefab'),
                                  ('NovelAirTurbulence', 'benchmarknovelties/NovelAirTurbulence.prefab'),
                                  ('InverseAirTurbulence', 'benchmarknovelties/InverseAirTurbulence.prefab'),
                                  ('Magnet', 'benchmarknovelties/Magnet.prefab'),
                                  ('Circle (normal block)', 'blocks/Circle.prefab'),
                                  ('PinkCircle', 'benchmarknovelties/PinkCircle.prefab'),
                                  ('Fan', 'benchmarknovelties/Fan.prefab'))}
    turbulence = {sprites[v]['sprite_guid'] + sprites[v]['color'] for v in contract.TURBULENCE_FORCE}
    visibility = {
        'air_turbulence_variants_share_sprite_and_colour': len(turbulence) == 1,
        'magnet_uses_normal_circle_sprite': sprites['Magnet']['sprite_guid'] == sprites['Circle (normal block)']['sprite_guid'],
        'pink_circle_sprite_differs_from_circle': sprites['PinkCircle']['sprite_guid'] != sprites['Circle (normal block)']['sprite_guid'],
        'inverse_gravity_and_storm_have_no_renderer': all(
            'm_Sprite' not in _text(RECOVERED_PREFABS / f'benchmarknovelties/{n}.prefab') for n in ('InverseGravity', 'Storm')),
        'storm_effect_overlay': {'scale': _field(_text(RECOVERED_PREFABS / 'benchmarknovelties/StormEffect.prefab'), 'm_LocalScale'),
                                 'color': _field(_text(RECOVERED_PREFABS / 'benchmarknovelties/StormEffect.prefab'), 'm_Color')},
    }
    if not (visibility['air_turbulence_variants_share_sprite_and_colour'] and visibility['magnet_uses_normal_circle_sprite']
            and visibility['pink_circle_sprite_differs_from_circle'] and visibility['inverse_gravity_and_storm_have_no_renderer']):
        raise ValueError(f'visibility facts differ from the contract: {visibility}')
    pigs = {}
    for pig, path in PIG_PREFABS.items():
        text = _text(RECOVERED_PREFABS / path)
        shots = re.search(r'^\s*numberOfBirdShotsToDie:\s*(\d+)', text, re.M)
        pigs[pig] = {'life': _number(text, '_life'), 'distinct_bird_hits_to_die': int(shots.group(1)) if shots else None}
    dll = (PLAYER_DATA / 'Managed/Assembly-CSharp.dll').read_bytes()
    resources = (PLAYER_DATA / 'resources.assets').read_bytes()
    classes = {name: name.encode() in dll for name in PLAYER_CLASSES}
    # Unity serializes object names as a little-endian length prefix followed by the bytes.
    prefabs = {name: struct.pack('<I', len(name)) + name.encode() in resources for name in PLAYER_PREFABS}
    if not all(classes.values()) or not all(prefabs.values()):
        raise ValueError(f'capture player lacks novelty code or prefabs: {classes} {prefabs}')
    dll_sha = sc.file_sha256(PLAYER_DATA / 'Managed/Assembly-CSharp.dll')
    users = {name: sc.file_sha256(ROOT / '.local-artifacts' / name / 'player/9001_Data/Managed/Assembly-CSharp.dll') == dll_sha
             for name in PLAYER_USERS}
    files = sorted({RECOVERED_PREFABS / row[0] for row in PREFAB_CHECKS}
                   | {RECOVERED_PREFABS / p for p in PIG_PREFABS.values()}
                   | {RECOVERED_SCRIPTS / row[0] for row in SCRIPT_CHECKS}
                   | {RECOVERED_PREFABS / f'benchmarknovelties/{n}.prefab' for n in
                      ('Storm', 'StormEffect', 'Fan', 'PinkCircle', *contract.TURBULENCE_FORCE)}
                   | {RECOVERED_PREFABS / 'blocks/Circle.prefab'})
    return {
        'recovered_source': ('original NovPhy player (sciencebirdsgames/Linux) recovered with AssetRipper 2.0.0; '
                             'decompiled behaviour checked against IL in docs/issue-76-canonical-migration-progress.md'),
        'prefab_constants': prefab_rows,
        'script_literals': script_rows,
        'visibility': visibility,
        'sprites': sprites,
        'pigs': pigs,
        'capture_player': {'assembly': sc.relative(PLAYER_DATA / 'Managed/Assembly-CSharp.dll'), 'sha256': dll_sha,
                           'resources_sha256': sc.file_sha256(PLAYER_DATA / 'resources.assets'),
                           'classes_present': classes, 'prefabs_present': prefabs,
                           'same_assembly_as': users},
        'designer_player_note': ('sciencebirdsgames/physics-v2 (built from tasks/task_template_designer) drops <Novelty> '
                                 'nodes in LevelLoader; it is not the capture player and is not used'),
        'files': [{'path': sc.relative(path), 'sha256': sc.file_sha256(path)} for path in files],
    }


# ---------------------------------------------------------- templates

def load_inventory():
    inventory = json.loads(INVENTORY.read_text())
    if inventory.get('schema') != 'issue_76_novelty_inventory_v1' or len(inventory['templates']) != 80:
        raise ValueError('unexpected novelty inventory')
    return inventory


def restrictions(template):
    text = template['constraints']['active_restrictions'] or ''
    return tuple(item.strip().lower() for item in text.split(',') if item.strip())


def materialize(template, generation_seed):
    """In-memory materialization with the template's own workbook row, restrictions included."""
    source = TEMPLATE_ROOT / template['template']
    coordinates = template['constraints']['active_coordinates']
    condition = BenchmarkCondition(f"novelty_level_{template['novelty_level']}", template['family'])
    record = create_scenario_template_record(source.read_bytes(), source_reference=str(source.relative_to(ROOT)),
                                             benchmark_conditions=[condition])
    request = CanonicalMaterializationRequest(
        template_path=source, output_xml_path=OUTPUT / 'unpublished/scenario.xml',
        output_manifest_path=OUTPUT / 'unpublished/generated-scenario.json',
        template_name=source.stem.split('_', 1)[1], benchmark_condition=condition, template_identity=record.identity,
        generation_seed=generation_seed, reference_point=tuple(coordinates[:2]),
        min_coordinate=tuple(coordinates[2:4]), max_coordinate=tuple(coordinates[4:6]),
        restricted_objects=restrictions(template), template_source_reference=str(source.relative_to(ROOT)))
    with redirect_stdout(StringIO()):
        generated, scenario = materialize_template_bound_level_instance(request, record, publish=False)
    return generated.xml_content.decode('utf-8'), scenario.to_dict()


def slots_of(xml):
    return sorted(node.attrib['scenarioObjectId'] for node in ET.fromstring(xml).iter() if 'scenarioObjectId' in node.attrib)


def template_facts(template):
    root = ET.fromstring(template['source_xml'].replace('encoding="utf-16"', 'encoding="utf-8"').encode('utf-8'))
    pigs = [node.attrib['type'] for node in root.iter('Pig')]
    novelty = [node.attrib['type'] for node in root.iter('Novelty')]
    if len(pigs) != 1 or len(novelty) > 1:
        raise ValueError(f"{template['template']}: contract v2 expects one pig and at most one novelty source")
    return {'pig_type': pigs[0], 'novelty_type': novelty[0] if novelty else None,
            'birds': len(root.findall('./Birds/Bird')),
            'bird_types': sorted({node.attrib['type'] for node in root.findall('./Birds/Bird')}),
            'slingshot_side': contract.slingshot_side(float(root.find('Slingshot').attrib['x']))}


def slot_sweep(templates):
    """Outcome-free generator sweep: share of generated levels inside the v1 and v2 vocabularies."""
    rows = {}
    for index, template in enumerate(templates):
        fit_v1 = fit_v2 = failures = 0
        maxima = Counter()
        for attempt in range(1, SWEEP_SEEDS + 1):
            try:
                xml, _ = materialize(template, SWEEP_SEED_BASE + index * 1_000 + attempt)
            except Exception:  # recorded as a count; a template that never materializes fails readiness
                failures += 1
                continue
            slots = slots_of(xml)
            fit_v1 += set(slots) <= set(contract.VOCABULARY_V1)
            fit_v2 += set(slots) <= set(contract.VOCABULARY_V2)
            for kind, count in Counter(contract.slot_kind(slot) for slot in slots).items():
                maxima[kind] = max(maxima[kind], count)
        rows[template['template']] = {
            'seeds': SWEEP_SEEDS, 'materialization_failures': failures,
            'fit_share_v1': round(fit_v1 / SWEEP_SEEDS, 4), 'fit_share_v2': round(fit_v2 / SWEEP_SEEDS, 4),
            'max_slots_per_kind': dict(sorted(maxima.items()))}
    return rows


# ------------------------------------------------------------ readiness

def visibility_class(facts):
    novelty_type = facts['novelty_type']
    if novelty_type is None:
        return 'visible_appearance' if facts['pig_type'].startswith('Pink') else 'none'
    if novelty_type.startswith('Pink'):
        return 'visible_appearance'
    return contract.FORCE_LAWS[contract.LAW_OF_TYPE[novelty_type]]['visibility']


def representation(facts):
    kind, law = facts['novelty_type'], contract.LAW_OF_TYPE.get(facts['novelty_type'])
    parts = ['pig:0000 appearance shift (same kind)'] if facts['pig_type'].startswith('Pink') else []
    if kind is not None:
        visual = contract.NOVELTY_VISUAL_KIND[kind]
        parts.append(f'novelty:0000 slot, kind {visual}' if visual else 'novelty:0000 slot declared, never visible')
    if law in ('region_push', 'bird_zone_push', 'radial_linear', 'event_wind'):
        parts.append('micro force-on edges + macro external-force-active')
    if law == 'global_gravity':
        parts.append('scene state gravity_y')
    if law == 'event_wind':
        parts.append('scene state storm_active')
    if facts['slingshot_side'] == 'right':
        parts.append('mirrored action inventory')
    return '; '.join(parts) or 'v1 slot kinds only'


def decision_expectation(pig, pigs):
    spec = pigs[pig]
    if spec['distinct_bird_hits_to_die'] and spec['distinct_bird_hits_to_die'] > 1:
        return (f"degenerate in single-shot capture: {pig} dies after {spec['distinct_bird_hits_to_die']} distinct "
                f"bird hits (life {spec['life']:g})")
    return f"single-shot pig removal possible ({pig} life {spec['life']:g})"


def identifiability(visibility):
    return {'none': 'not required', 'visible_appearance': 'not required (no force)',
            'visible_entity': 'source visible; force-on edges probed post-capture',
            'visible_region_hidden_law': 'region visible, magnitude/sign hidden: probe from motion history',
            'hidden_law_normal_appearance': 'source looks like a normal wood circle: probe from motion history',
            'invisible_global': 'gravity sign from configuration and motion history: probe',
            'invisible_trigger_visible_after_onset': 'onset hidden until the overlay appears: probe'}[visibility]


def template_readiness(template, facts, sweep, sources):
    level = template['novelty_level']
    v1_reasons, v2_reasons = [], []
    # v1 as the #76/#77 records defined it: authored slots outside the 18-slot vocabulary (generator
    # distraction overflow was handled by outcome-free rejection screening, as in #109).
    if sweep['fit_share_v1'] == 0:
        v1_reasons.append('slot_outside_v1_vocabulary (authored slots)')
    if facts['slingshot_side'] == 'right':
        v1_reasons.append('left_pull_only_action_contract')
    if facts['novelty_type'] in contract.LAW_OF_TYPE:
        v1_reasons.append('force_unrepresented')
    if level == 7:
        v1_reasons.append('objective_undeclared')
    if sweep['materialization_failures']:
        v2_reasons.append(f"materialization_failures ({sweep['materialization_failures']}/{sweep['seeds']})")
    if sweep['fit_share_v2'] < FIT_SHARE_FLOOR:
        v2_reasons.append(f"slot_outside_v2_vocabulary (generated fit share {sweep['fit_share_v2']})")
    if set(facts['bird_types']) - set(ALLOWED_BIRDS):
        v2_reasons.append('bird_type_outside_capture_support')
    if facts['novelty_type'] is not None and facts['novelty_type'] not in contract.NOVELTY_VISUAL_KIND:
        v2_reasons.append('novelty_type_outside_contract')
    present = sources['capture_player']['prefabs_present']
    for novel in (facts['novelty_type'], facts['pig_type']):
        if novel in PLAYER_PREFABS and not present[novel]:
            v2_reasons.append(f'{novel}_prefab_absent_from_capture_player')
    if facts['novelty_type'] is not None and facts['novelty_type'] not in PLAYER_PREFABS:
        v2_reasons.append('novelty_prefab_not_checked_in_capture_player')
    vis = visibility_class(facts)
    return {
        'template': template['template'], 'novelty_level': level, 'family': template['family'],
        'scenario': SCENARIOS[int(template['family'][-2:])], 'pig_type': facts['pig_type'], 'birds': facts['birds'],
        'novelty_type': facts['novelty_type'], 'slingshot_side': facts['slingshot_side'],
        'restricted_distractions': list(restrictions(template)),
        'fit_share_v1': sweep['fit_share_v1'], 'fit_share_v2': sweep['fit_share_v2'],
        'max_slots_per_kind': sweep['max_slots_per_kind'],
        'force_law': contract.LAW_OF_TYPE.get(facts['novelty_type']), 'visibility': vis,
        'representation_v2': representation(facts), 'identifiability': identifiability(vis),
        'decision_verdict_expectation': decision_expectation(facts['pig_type'], sources['pigs']),
        'v1_status': 'unsupported' if v1_reasons else 'supported', 'v1_reasons': v1_reasons,
        'v2_status': 'unsupported' if v2_reasons else 'supported', 'v2_reasons': v2_reasons,
    }


def cells(rows):
    """The 45 scenario x novelty cells; a normal cell aggregates the 8 level-0 counterparts."""
    table = []
    for level in range(9):
        for scenario_index, scenario in SCENARIOS.items():
            members = [row for row in rows if row['novelty_level'] == level and row['scenario'] == scenario]
            if len(members) != (8 if level == 0 else 1):
                raise ValueError(f'cell {level}/{scenario} has {len(members)} templates')
            v1 = sorted({reason.split(' (')[0] for row in members for reason in row['v1_reasons']})
            v2 = sorted({f"{row['family']}: {reason}" for row in members for reason in row['v2_reasons']})
            table.append({
                'cell': f'{level}/{scenario}', 'novelty_level': level, 'category': CATEGORIES[level],
                'scenario': scenario, 'templates': [row['template'] for row in members],
                'v1_status': 'unsupported' if v1 else 'supported', 'v1_reasons': v1,
                'v1_templates_supported': f"{sum(row['v1_status'] == 'supported' for row in members)}/{len(members)}",
                'v2_status': 'unsupported' if v2 else 'supported', 'v2_reasons': v2,
                'v2_templates_supported': f"{sum(row['v2_status'] == 'supported' for row in members)}/{len(members)}",
                'min_fit_share_v2': min(row['fit_share_v2'] for row in members),
                'force_laws': sorted({row['force_law'] for row in members if row['force_law']}),
                'visibility': sorted({row['visibility'] for row in members}),
                'representation_v2': sorted({row['representation_v2'] for row in members}),
                'decision_verdict_expectation': sorted({row['decision_verdict_expectation'] for row in members}),
                'captured_by_pairs': [f"pair-{row['family']}" for row in members],
            })
    return table


# -------------------------------------------------------------- rosters

def pairs(templates):
    by_key = {(t['novelty_level'], t['family']): t for t in templates}
    result = []
    for level in range(1, 9):
        for scenario in SCENARIOS:
            family = f'type010{level}0{scenario}'
            result.append({'pair': f'pair-{family}', 'family': family, 'novelty_level': level,
                           'scenario': SCENARIOS[scenario],
                           'normal': by_key[(0, family)], 'novel': by_key[(level, family)]})
    return result


def seed_blocks(pair_list):
    blocks = []
    for split, _ in SPLITS:
        low = SPLIT_OFFSETS[split] + 1
        high = SPLIT_OFFSETS[split] + len(pair_list) * PAIR_SEED_BLOCK - 1
        for kind, base in (('generation', GENERATION_SEED_BASE), ('engine', ENGINE_SEED_BASE)):
            blocks.append({'name': f'issue-110 {split} {kind} seeds', 'split': split, 'kind': kind,
                           'low': base + low, 'high': base + high})
    blocks.append({'name': 'issue-110 outcome-free generator slot sweep (never captured)', 'split': 'sweep',
                   'kind': 'generation', 'low': SWEEP_SEED_BASE + 1, 'high': SWEEP_SEED_BASE + 80 * 1_000})
    return blocks


def seeds(split, pair_index, attempt):
    if not 1 <= attempt < PAIR_SEED_BLOCK:
        raise ValueError(f'{split}/{pair_index} exhausted its seed block')
    offset = SPLIT_OFFSETS[split] + pair_index * PAIR_SEED_BLOCK + attempt
    return GENERATION_SEED_BASE + offset, ENGINE_SEED_BASE + offset


def sealed_cohort_exposure():
    """#109's frozen sealed cohort is prior exposure for this plan (seeds and level content)."""
    value = sc.files.read(SEALED_PLAN)
    entry = {'path': sc.relative(SEALED_PLAN), **sc.files.exposure_projection(value), **sc.n1._prior_details(value)}
    levels = [level for rosters in value['splits'].values() for roster in rosters.values() for level in roster]
    entry['generation_or_reserved_seeds'] = sorted(set(entry['generation_or_reserved_seeds'])
                                                   | {level['generation_seed'] for level in levels}
                                                   | {level['engine_seed'] for level in levels})
    entry['xml_content_sha256'] = sorted(set(entry.get('xml_content_sha256', ())) | {l['xml_sha256'] for l in levels})
    entry['scenario_content_sha256'] = sorted(set(entry.get('scenario_content_sha256', ()))
                                              | {l['scenario_sha256'] for l in levels})
    return entry


def prior_exposure(frozen_paths=None):
    entries, _ = sc.prior_exposure()
    entries = {entry['path']: entry for entry in entries}
    sealed = sealed_cohort_exposure()
    entries[sealed['path']] = sealed
    if frozen_paths is None:
        return [entries[key] for key in sorted(entries)], []
    missing = sorted(set(frozen_paths) - set(entries))
    if missing:
        raise ValueError(f'frozen prior-exposure sources disappeared: {missing}')
    return ([entries[key] for key in sorted(frozen_paths)],
            [entries[key] for key in sorted(set(entries) - set(frozen_paths))])


def screen(xml):
    tree = ET.fromstring(xml)
    slots = slots_of(xml)
    outside = sorted(set(slots) - set(contract.VOCABULARY_V2))
    if outside:
        return 'slot_outside_v2_vocabulary', {'slots': outside}, slots
    birds = sorted({bird.attrib.get('type') for bird in tree.findall('./Birds/Bird')} - set(ALLOWED_BIRDS))
    if birds:
        return 'bird_not_birdred', {'bird_types': birds}, slots
    return None, None, slots


def build_rosters(pair_list, prior, sizes=dict(SPLITS), materializer=materialize):
    """Paired screening: an attempt is accepted only when both sides pass every rule."""
    index = sc.prior_index(prior)
    owners = {'xml': {}, 'scenario': {}, 'scenario_identity': {}}
    rosters, rejected, bound = {}, [], []
    for split, _ in SPLITS:
        rosters[split] = {}
        for pair_index, pair in enumerate(pair_list):
            roster, attempt = [], 0
            while len(roster) < sizes[split]:
                attempt += 1
                generation_seed, engine_seed = seeds(split, pair_index, attempt)
                base = {'split': split, 'pair': pair['pair'], 'attempt': attempt,
                        'generation_seed': generation_seed, 'engine_seed': engine_seed}
                reason, detail, sides = None, None, {}
                if {generation_seed, engine_seed} & index['seeds']:
                    reason, detail = 'reserved_seed_collision', {}
                for side in SIDES:
                    if reason is not None:
                        break
                    try:
                        xml, scenario = materializer(pair[side], generation_seed)
                    except Exception as error:  # generator failure is outcome-free; record it
                        reason, detail = 'materialization_error', {'side': side, 'error': f'{type(error).__name__}: {error}'}
                        break
                    reason, detail, slots = screen(xml)
                    if reason is not None:
                        detail = {'side': side, **detail}
                        break
                    xml_sha, scenario_sha = sc.xml_digest(xml), sc.n1._digest(scenario)
                    identities = sc.files.exposure_projection(scenario)['scenario_identities']
                    for key, digest, prior_set, name in (
                            ('xml', xml_sha, index['xml_content'], 'duplicate_xml_content'),
                            ('scenario', scenario_sha, index['scenario_content'], 'duplicate_scenario_content')):
                        if digest in owners[key] or digest in prior_set:
                            reason, detail = name, {'side': side, 'duplicates': owners[key].get(digest, 'prior exposure')}
                            break
                    if reason is None:
                        clash = [item for item in identities
                                 if item in owners['scenario_identity'] or item in index['scenario_identities']]
                        if clash:
                            reason = 'duplicate_scenario_identity'
                            detail = {'side': side, 'duplicates': owners['scenario_identity'].get(clash[0], 'prior exposure')}
                    sides[side] = (xml, scenario, slots, xml_sha, scenario_sha, identities)
                if reason is not None:
                    rejected.append({**base, 'reason': reason, 'detail': detail})
                    continue
                ordinal = len(roster) + 1
                level = {'ordinal': ordinal, 'generation_seed': generation_seed, 'engine_seed': engine_seed}
                for side in SIDES:
                    xml, scenario, slots, xml_sha, scenario_sha, identities = sides[side]
                    template = pair[side]
                    identity = f"issue-110-{split}-{pair['family']}-{side}-{ordinal:03}"
                    owners['xml'][xml_sha] = identity
                    owners['scenario'][scenario_sha] = identity
                    owners['scenario_identity'].update({item: identity for item in identities})
                    side_name = contract.slingshot_side(float(ET.fromstring(xml).find('Slingshot').attrib['x']))
                    level[side] = {'identity': identity, 'novelty_level': template['novelty_level'],
                                   'generator_family': pair['family'], 'template': template['template'],
                                   'slingshot_side': side_name, 'xml_sha256': xml_sha, 'scenario_sha256': scenario_sha,
                                   'generated_slots': slots, 'xml_path': f'levels/{identity}.xml',
                                   'scenario_path': f'levels/{identity}.scenario.json'}
                    bound.append({'identity': identity, 'base_cluster': f"issue-110-{split}-{pair['family']}-{ordinal:03}",
                                  'generation_seed': generation_seed, 'engine_seed': engine_seed,
                                  'scenario': scenario, 'xml': xml})
                roster.append(level)
            rosters[split][pair['pair']] = roster
    return rosters, rejected, bound


def disjointness(rosters):
    seen = {'identity': set(), 'xml': set(), 'scenario': set(), 'seed': {}}
    for split, by_pair in rosters.items():
        for pair, roster in by_pair.items():
            for level in roster:
                key = (split, pair)
                if seen['seed'].setdefault(level['generation_seed'], key) != key:
                    raise ValueError(f"generation seed {level['generation_seed']} is shared across rosters")
                for side in SIDES:
                    item = level[side]
                    for field, value in (('identity', item['identity']), ('xml', item['xml_sha256']),
                                         ('scenario', item['scenario_sha256'])):
                        if value in seen[field]:
                            raise ValueError(f'{field} {value} appears twice')
                        seen[field].add(value)
    return {'levels': len(seen['identity']), 'splits_level_disjoint': True, 'pairs_share_seeds_within_level_only': True}


# ----------------------------------------------------------- candidates

def candidates():
    """#93 Arm B 4x4 grid (16), the primary inventory of #96/#104/Gate B; mirrored for the right slingshot."""
    grid = sc.candidate_set(('grid',), sc.inventories())
    sides = {side: [{'suffix': item['suffix'], 'ordinal': item['ordinal'],
                     'action': contract.action_for_side(item['action'], side),
                     'inventory_entries': item['inventory_entries']} for item in grid['candidates']]
             for side in ('left', 'right')}
    for side, items in sides.items():
        bad = [item['suffix'] for item in items if not contract.action_within_v2(item['action'], side)]
        if bad:
            raise ValueError(f'{side} candidates outside the v2 action contract: {bad}')
    return {'inventory': 'grid', 'source': 'scripts/run_second_parameterization_probe.grid_inventory (#93 Arm B)',
            'unique_candidates': grid['unique_candidates'], 'by_slingshot_side': sides,
            'identity_rule': '<level-identity>-<suffix>; normal and novel sides of a level share suffixes'}


def branch_digests(rosters, cand):
    """Expanded branch identities per split; the smoke split carries only SMOKE_CANDIDATES."""
    digests = {}
    for split, by_pair in rosters.items():
        rows = [{'identity': f"{level[side]['identity']}-{item['suffix']}", 'engine_seed': level['engine_seed'],
                 'action': item['action']}
                for roster in by_pair.values() for level in roster for side in SIDES
                for item in cand['by_slingshot_side'][level[side]['slingshot_side']]
                if split != 'smoke' or item['suffix'] in SMOKE_CANDIDATES]
        if len({row['identity'] for row in rows}) != len(rows):
            raise ValueError(f'{split} branch identities are not unique')
        digests[split] = {'branches': len(rows), 'sha256': sc.n1._digest(rows)}
    return digests


# --------------------------------------------------------------- budget

def budget(cost, pair_count, cand_count):
    throughput = cost['throughput']
    per_level = 2 * cand_count  # both sides of a paired level
    rows = {str(prefix): {
        'levels_per_split_per_side': prefix,
        'fit': sc.budget_row(pair_count * prefix, per_level, throughput),
        'evaluation': sc.budget_row(pair_count * prefix, per_level, throughput),
        'total': sc.budget_row(2 * pair_count * prefix, per_level, throughput),
    } for prefix in PREFIXES}
    smoke = sc.budget_row(pair_count * 2, len(SMOKE_CANDIDATES), throughput)
    return {
        'units': ('levels count paired levels (one normal + one novel instance); wall_hours at the measured #109 '
                  'smoke rate (6 workers on the RTX 3090 workstation); worker count on the 48-core server is '
                  'unmeasured and must be re-measured by the smoke'),
        'prefixes': rows, 'smoke': smoke,
        'recommended_prefix': RECOMMENDED_PREFIX,
        'recommendation_rationale': (
            f'{RECOMMENDED_PREFIX} paired evaluation levels per cell gives every per-cell bootstrap at least '
            f'{RECOMMENDED_PREFIX} clusters (#109 flagged 3-cluster intervals as understating uncertainty) and '
            f'{5 * RECOMMENDED_PREFIX} per novelty level for the #107 per-novelty-type table; the campaign freeze '
            'may go deeper along the frozen roster order'),
    }


# ------------------------------------------------------------- protocols

def protocols():
    return {
        'encoder': {
            'architecture': ('the #99 SpatialSlotParser (E99) with one query per v2 slot (28) and the v2 kind '
                             'vocabulary; carrier = v1 header + 4 scene fields + 13 features x 28 slots'),
            'carrier_dim': contract.CARRIER_DIM_V2, 'recipe': 'E99: 12 epochs, batch 64, lr 1e-3, 320x240 input',
            'presence_target': ('lifecycle active and body present, except the novelty slot: present when its source '
                                'has a renderer (Fan, air turbulence, Magnet, Pink*); InverseGravity/Storm never present'),
            'scene_targets': {'gravity_y_normalized': 'world.gravity_vector[1] / 9.8 from the native trace',
                              'storm_active': 'storm_onset_step <= step (first bird entity destroyed) on level 8'},
            'scene_head_input': 'carriers of the last k = 15 agent frames, the #113 temporal-head interface',
            'arms': 'one encoder per condition shared by the hybrid and continuous predictor families (identical budgets)',
        },
        'rollout_error': {
            'requests': [f'F-{delta}-{alpha}' for delta in (1, 5, 15) for alpha in ('continuous', 'micro', 'macro')],
            'anchor': 'decision state (native step 30000) of every evaluation level x candidate',
            'endpoints_frames': [15, 60, 225],
            'error': 'presence-masked carrier MSE against the engine-projected v2 carrier',
            'conditions': {
                'zero_shot': ('encoder and predictors fitted on normal data only: the normal side of every pair\'s fit '
                              'split plus the #104 normal fit split'),
                'few_shot': ('per novelty level, encoder and predictors adapted on the novel side of that level\'s five '
                             'fit rosters: 2000 updates, batch 64, lr 1e-4, identical for every arm'),
                'pooling': 'never pooled; reported per cell',
            },
            'contrast': ('novel-side error minus paired normal-side error, paired by roster ordinal; DESCRIPTIVE '
                         'level-clustered percentile bootstrap, 10 000 draws'),
        },
        'identifiability': {
            'question': 'is the force state at t recoverable from the permitted observations up to t?',
            'targets': {'external-force-active': 'levels 2, 3, 4, 7, 8 and the level-0 turbulence families',
                        'force-on edges': 'visible sources (Fan, air turbulence)',
                        'gravity-inverted': 'level 6 against its paired normal levels',
                        'storm-active': 'level 8'},
            'inputs': 'v2 carriers of the last k agent frames plus 1-frame differences; k in (1, 5, 15), primary 15',
            'probe': 'MLP (hidden 64), class-balanced BCE, 2000 updates, lr 1e-3, seed 1100001',
            'ceiling': 'the same probe on engine features (positions and velocities of every slot)',
            'split': 'leave-one-level-out over the cell\'s fit roster; scored once on its evaluation roster (DESCRIPTIVE)',
            'metric': 'AUROC, level-clustered bootstrap 10 000 draws',
            'rule': {'identifiable': 'point >= 0.75 and lower bound > 0.65',
                     'not_identifiable': 'upper bound < 0.75',
                     'otherwise': 'readiness_or_precision_insufficient',
                     'guard': 'fewer than 20 positive or 20 negative frames -> readiness_or_precision_insufficient',
                     'label_invalid': 'engine-feature ceiling below 0.75 -> the label, not the encoder, is at fault'},
            'rationale': ('#113 exploratory probes: a recoverable predicate (steady-state) reached 0.81-0.92, an '
                          'unrecoverable one (structure-unstable) at most 0.74 even on engine features; 0.75 separates them'),
            'consequence': ('a non-identifiable force keeps its cell captured and its rollout error reported; claims about '
                            'reading that force are withheld'),
        },
        'force_predicates': {
            'micro': list(contract.MICRO_PREDICATES_V2), 'macro_added': list(contract.MACRO_FORCE_PREDICATES),
            'scene_labels': list(contract.SCENE_LABELS),
            'per_delta_rule': ('a predicate requested at Delta is true when it holds at any frame in (t, t + Delta]; the '
                               '#113 settle / structure-change redefinitions apply unchanged'),
            'labeler': 'world_model/data/novelty_contract.py force_on / force_edges / storm_onset_step / gravity_inverted',
            'approximations': contract.LABEL_APPROXIMATIONS,
        },
        'smoke_gate': {
            'membership': f'the smoke roster (1 paired level per pair, both sides) x candidates {list(SMOKE_CANDIDATES)}',
            'pass_criteria_per_template': [
                'zero typed capture failures (#109 taxonomy) and render invariance at the decision seal',
                'every authored slot, including novelty:0000, present in the trace with its authored identity',
                'world gravity (0, 6) on level 6 and (0, -9.8) elsewhere',
                'right-slingshot candidates launch leftward (bird_launched velocity x < 0)',
                'residual acceleration of contact-free bodies agrees in sign with every active force-on label',
            ],
            'reported_not_gated': ['storm onset inside the capture window', 'force-on prevalence per template'],
            'failure': ('a template failing a pass criterion makes its cell unsupported (typed reason) before the '
                        'campaign freeze; a labeler disagreement is repaired in the derivation before any training'),
        },
        'campaign_dependencies': [
            '#113 capture-window spec merged before launch (storm onset needs the first bird destroyed, >= 3 s after '
            'its first collision; level 6 and 8 are window-sensitive)',
            'one joint #104 + #110 freeze on the issue-109 pipeline (scripts/capture_pipeline_v2.py)',
            'dispatch order ordinal-major: every pair at ordinal k before any pair at k + 1, so a stopped campaign '
            'keeps balanced cells',
        ],
    }


# ------------------------------------------------------------------ plan

def derive(frozen_prior_paths=None):
    """Deterministic derivation: (plan without frozen_at, level files, readiness rows, post-freeze prior)."""
    if len(contract.VOCABULARY_V2) != 28 or not set(contract.VOCABULARY_V1) <= set(contract.VOCABULARY_V2):
        raise ValueError('contract v2 vocabulary must be a 28-slot superset of v1')
    if list(contract.VOCABULARY_V1) != sc.slot_vocabulary():
        raise ValueError('contract VOCABULARY_V1 differs from the frozen issue-71 vocabulary')
    sources = engine_sources()
    inventory = load_inventory()
    templates = inventory['templates']
    log('generator slot sweep (outcome-free, nothing kept)')
    sweep = slot_sweep(templates)
    rows = [template_readiness(t, template_facts(t), sweep[t['template']], sources) for t in templates]
    table = cells(rows)
    pair_list = pairs(templates)
    blocks = seed_blocks(pair_list)
    reserved = sc.reserved_ranges() + [{'name': b['name'], 'low': b['low'], 'high': b['high']}
                                       for b in sc.files.read(SEALED_PLAN)['seed_contract']['reserved_by_this_plan']]
    seed_audit = sc.audit_seed_blocks(blocks, reserved)
    prior, later = prior_exposure(frozen_prior_paths)
    log('screening paired rosters')
    rosters, rejected, bound = build_rosters(pair_list, prior)
    audit = sc.n1.audit_disjointness(bound, prior)
    cand = candidates()
    cost = sc.load_cost()
    level_files = {}
    for member in bound:
        level_files[f"levels/{member['identity']}.xml"] = member['xml']
        level_files[f"levels/{member['identity']}.scenario.json"] = sc.json_text(member['scenario'])
    reasons = Counter(row['reason'] for row in rejected)
    supported = [cell['cell'] for cell in table if cell['v2_status'] == 'supported']
    plan = {
        'schema': SCHEMA, 'identity': IDENTITY, 'version': 1, 'issue': 110, 'consumer_issues': [104, 107],
        'frozen_before_any_capture': True, 'engine_seconds': 0, 'validation_command': VALIDATION_COMMAND,
        'contract': {
            'identity': contract.CONTRACT_IDENTITY, 'module': 'world_model/data/novelty_contract.py',
            'module_sha256': sc.file_sha256(ROOT / 'world_model/data/novelty_contract.py'),
            'vocabulary_v1': list(contract.VOCABULARY_V1), 'vocabulary_v2': list(contract.VOCABULARY_V2),
            'slot_capacity': contract.SLOT_CAPACITY, 'scene_fields': list(contract.SCENE_FIELDS),
            'carrier_dim_v1': contract.CARRIER_DIM_V1, 'carrier_dim_v2': contract.CARRIER_DIM_V2,
            'kind_vocabulary_v2': list(contract.KIND_VOCABULARY_V2),
            'novelty_visual_kind': contract.NOVELTY_VISUAL_KIND,
            'force_laws': contract.FORCE_LAWS,
            'action_bounds_v2': contract.ACTION_BOUNDS_V2, 'action_encoding': contract.ACTION_ENCODING,
            'task_objective': contract.TASK_OBJECTIVE,
        },
        'engine_sources': sources,
        'template_source': {'artifact': sc.relative(INVENTORY), 'sha256': sc.file_sha256(INVENTORY)},
        'slot_sweep': {'seed_rule': f'{SWEEP_SEED_BASE} + template index * 1000 + attempt (1..{SWEEP_SEEDS})',
                       'templates': sweep},
        'template_readiness': rows,
        'cells': table,
        'cell_counts': {
            'v1_supported': sum(c['v1_status'] == 'supported' for c in table),
            'v2_supported': len(supported), 'v2_unsupported': len(table) - len(supported),
            'captured': len(supported)},
        'capture_membership': {
            'decision': ('every cell supported under contract v2 is captured, through the 40 upstream normal/novel '
                         'template pairs; the 5 normal cells are covered by the normal sides of their 8 pairs'),
            'pairs': [{'pair': p['pair'], 'family': p['family'], 'novelty_level': p['novelty_level'],
                       'scenario': p['scenario'], 'normal_template': p['normal']['template'],
                       'novel_template': p['novel']['template']} for p in pair_list],
            'roster_sizes_per_pair': dict(SPLITS),
            'split_roles': {'smoke': 'pre-campaign rendered smoke only; never fitted or scored',
                            'fit': 'normal side: zero-shot training; novel side: few-shot adaptation',
                            'evaluation': 'never fitted; rollout error, identifiability and #107 scoring'},
        },
        'rosters': rosters,
        'counts': {'paired_levels': {split: sum(len(r) for r in by_pair.values()) for split, by_pair in rosters.items()},
                   'level_instances': len(bound), 'rejected_attempts': len(rejected),
                   'rejections_by_reason': {reason: reasons.get(reason, 0) for reason in REJECTION_REASONS}},
        'rejected_attempts': rejected,
        'screening_rules': {
            'declared_before_any_verdict': True, 'outcome_free': True,
            'order': 'smoke -> fit -> evaluation; pairs in level-major, scenario-minor order; ascending attempt',
            'rules': ['reserved seed collision', 'materialization error on either side',
                      'any generated slot outside the v2 vocabulary on either side', 'any bird not BirdRed',
                      'xml / scenario content or scenario identity equal to an earlier level or a prior exposure'],
            'pairing': 'both sides share the generation and engine seed; an attempt is accepted only if both pass',
            'restrictions': 'workbook distraction restrictions applied (Magnet levels: no wood circle distractions)',
        },
        'seed_contract': {'generation_seed_formula': f'{GENERATION_SEED_BASE} + split offset + pair index * {PAIR_SEED_BLOCK} + attempt',
                          'engine_seed_formula': f'{ENGINE_SEED_BASE} + split offset + pair index * {PAIR_SEED_BLOCK} + attempt',
                          'split_offsets': SPLIT_OFFSETS, 'reserved_by_this_plan': blocks,
                          'prior_reserved_ranges': reserved, 'audit': seed_audit},
        'disjointness': {'rosters': disjointness(rosters), 'prior_exposure_audit': audit},
        'prior_exposure': {'sources': sc.prior_summary(prior), 'count': len(prior),
                           'projection': 'scripts/prepare_sealed_cohort.prior_exposure + the #109 sealed cohort plan'},
        'candidates': cand,
        'expanded_branch_digests': branch_digests(rosters, cand),
        'smoke_branches': [f"{level[side]['identity']}-{suffix}" for level in
                           (roster[0] for roster in rosters['smoke'].values()) for side in SIDES
                           for suffix in SMOKE_CANDIDATES],
        'cost': cost,
        'budget': budget(cost, len(pair_list), cand['unique_candidates']),
        'protocols': protocols(),
        'claim_boundary': ('design only: readiness from source, prefab and generator evidence; no engine ran, no outcome '
                           'was read, no model was trained or scored. Runtime readiness is decided by the smoke gate. '
                           'Prior dispositions (#15-#112) are read-only inputs'),
    }
    return plan, level_files, rows, later


READINESS_COLUMNS = ('template', 'novelty_level', 'family', 'scenario', 'novelty_type', 'pig_type', 'birds',
                     'slingshot_side', 'fit_share_v1', 'fit_share_v2', 'force_law', 'visibility', 'representation_v2',
                     'identifiability', 'decision_verdict_expectation', 'v1_status', 'v1_reasons', 'v2_status',
                     'v2_reasons')


def readiness_csv(rows):
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=READINESS_COLUMNS, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: '; '.join(row[key]) if isinstance(row[key], list) else row[key] for key in READINESS_COLUMNS})
    return buffer.getvalue()


def findings(plan):
    lines = [f"# {IDENTITY}: per-cell readiness (45 cells)", '',
             f"Frozen design, zero engine seconds. v1 = frozen 18-slot contract; v2 = {contract.CONTRACT_IDENTITY}.",
             f"Supported: v1 {plan['cell_counts']['v1_supported']}/45, v2 {plan['cell_counts']['v2_supported']}/45.", '',
             '| cell | category | v1 | v1 reasons | v2 | min v2 fit share | visibility | representation v2 | decision verdict |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for cell in plan['cells']:
        lines.append('| ' + ' | '.join([
            cell['cell'], cell['category'], f"{cell['v1_status']} ({cell['v1_templates_supported']})",
            ', '.join(cell['v1_reasons']) or '-',
            f"{cell['v2_status']} ({cell['v2_templates_supported']})"
            + (f" ({'; '.join(cell['v2_reasons'])})" if cell['v2_reasons'] else ''),
            f"{cell['min_fit_share_v2']:.3f}", ', '.join(cell['visibility']),
            ' / '.join(cell['representation_v2']),
            ' / '.join(cell['decision_verdict_expectation'])]) + ' |')
    lines += ['', '## Budget (paired levels per split per side; grid 16 candidates per side)', '',
              '| prefix | branches | wall h (6 workers) | worker h | GiB |', '| --- | --- | --- | --- | --- |']
    for prefix, row in plan['budget']['prefixes'].items():
        total = row['total']
        lines.append(f"| {prefix} | {total['branches']} | {total['wall_hours']} | {total['worker_hours']} | {total['artifact_gib']} |")
    smoke = plan['budget']['smoke']
    lines += ['', f"Smoke gate: {smoke['branches']} branches, {smoke['wall_hours']} wall h, {smoke['artifact_gib']} GiB.",
              f"Recommended campaign prefix: {plan['budget']['recommended_prefix']} ({plan['budget']['recommendation_rationale']}).", '']
    return '\n'.join(lines)


def summary(plan):
    return {'identity': plan['identity'], 'cell_counts': plan['cell_counts'], 'counts': plan['counts'],
            'branches': plan['expanded_branch_digests'],
            'budget_recommended': plan['budget']['prefixes'][str(RECOMMENDED_PREFIX)]['total'],
            'unsupported_cells': [c['cell'] for c in plan['cells'] if c['v2_status'] != 'supported']}


def outputs(plan, rows):
    return {'readiness.csv': readiness_csv(rows), 'findings.md': findings(plan)}


def dry_run():
    plan, level_files, rows, _ = derive()
    value = summary(plan)
    value.update({'level_files': len(level_files), 'writes': 'none'})
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def prepare(output=OUTPUT):
    output = Path(output)
    if (output / 'plan.json').exists() or (output / 'levels').exists():
        raise ValueError(f'{output} already holds a frozen plan or level files; refusing to overwrite')
    plan, level_files, rows, _ = derive()
    plan['frozen_at'] = sc.utc_now()
    (output / 'levels').mkdir(parents=True)
    for name, text in sorted(level_files.items()):
        (output / name).write_text(text)
    (output / 'plan.json').write_text(sc.json_text(plan))
    for name, text in outputs(plan, rows).items():
        (output / name).write_text(text)
    value = summary(plan)
    value.update({'output': str(output), 'frozen_at': plan['frozen_at']})
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)
    return plan


def validate(output=OUTPUT):
    output = Path(output)
    text = (output / 'plan.json').read_text()
    saved = json.loads(text)
    if saved.get('identity') != IDENTITY or saved.get('schema') != SCHEMA:
        raise ValueError('plan identity/schema differs from this runner')
    plan, level_files, rows, later = derive([entry['path'] for entry in saved['prior_exposure']['sources']])
    plan['frozen_at'] = saved['frozen_at']
    problems = []
    if sc.json_text(plan) != text:
        differing = sorted(key for key in set(plan) | set(saved) if plan.get(key) != saved.get(key))
        problems.append(f'plan.json differs from the re-derivation in: {differing}')
    for name, content in outputs(plan, rows).items():
        if (output / name).read_text() != content:
            problems.append(f'{name} differs from the re-derivation')
    present = {str(path.relative_to(output)) for path in (output / 'levels').iterdir()}
    if present != set(level_files):
        problems.append(f'level file set differs: {len(present ^ set(level_files))} names')
    for name, content in level_files.items():
        if name in present and (output / name).read_text() != content:
            problems.append(f'{name} differs from the re-derivation')
    bound = [{'identity': level[side]['identity'], 'base_cluster': level[side]['identity'],
              'generation_seed': level['generation_seed'], 'engine_seed': level['engine_seed'],
              'scenario': json.loads((output / level[side]['scenario_path']).read_text()),
              'xml': (output / level[side]['xml_path']).read_text()}
             for by_pair in saved['rosters'].values() for roster in by_pair.values() for level in roster for side in SIDES]
    sc.n1.audit_disjointness(bound, later)  # post-freeze exposures must not collide either
    value = {'identity': IDENTITY, 'validated': not problems, 'problems': problems[:20], 'level_instances': len(bound),
             'post_freeze_prior_sources': [entry['path'] for entry in later], 'frozen_at': saved['frozen_at']}
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('dry-run', 'prepare', 'validate'):
        modes.add_argument(f'--{mode}', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='output root (default: the frozen artifact)')
    args = parser.parse_args()
    try:
        if args.dry_run:
            dry_run()
        elif args.prepare:
            prepare(args.output)
        elif not validate(args.output):
            sys.exit(1)
    except Exception as error:
        log(f'error: {type(error).__name__}: {error}')
        sys.exit(1)


if __name__ == '__main__':
    main()
