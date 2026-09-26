"""Issue #109 work item 4: freeze the sealed fit / policy / evaluation cohort plan for #104.

Pure planning: level XML is generated in-process (no engine, no capture, no
outcome is read). Families are the two normal-mechanics families the ICLR #96
contrasts evaluate (rolling ``type010103`` and sliding ``type010105``, novelty
level 0), bound to the same frozen templates as #77 N1.

Three level-disjoint splits per family, fixed before any capture:

* ``fit``        240 levels/family: predictor training;
* ``policy``      60 levels/family: controller training;
* ``evaluation`` 200 levels/family: held out, never fitted.

Each roster is ordered by generation seed; #104 captures an ordered PREFIX of
each roster whose depth it fixes from its power analysis before any verdict.

Outcome-free screening, declared before any verdict (applied in fixed order
fit -> policy -> evaluation, family order, ascending seed; a rejected seed is
recorded with its reason and the roster continues with the next seed):

1. static slot fit: every generated ``scenarioObjectId`` lies in the frozen
   18-slot checkpoint vocabulary and every bird is ``BirdRed``;
2. content dedup: xml sha256, scenario sha256 or scenario identity duplicating
   another roster level or any prior exposure is rejected;
3. candidate densification: evaluation levels carry the deduplicated union
   angle(13) + grid(16) + offset(11) + power(20); fit/policy levels carry the
   training behaviour inventory angle(13) + grid(16). Each unique action gets one
   identity ``<level-identity>-cNN``; inventories map to candidate ids.

Modes: --dry-run (inventory only, writes nothing), --prepare (materialize and
freeze plan.json once), --validate (re-derive and byte-compare, re-audit).
Validation: ``python -u -m scripts.prepare_sealed_cohort --validate``.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

from scripts import issue_76_expansion as files
from scripts import prepare_issue_77_n1 as n1
from scripts import prepare_issue_77_n2 as n2
from scripts import prepare_issue_77_n2n as n2n
from scripts import run_launch_power_probe as p94
from scripts import run_second_parameterization_probe as p93
from world_model.training.native_history_data import VOCABULARY

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'issue-109-sealed-cohort-v1'
SCHEMA = 'issue_109_sealed_cohort_plan_v1'
OUTPUT = ROOT / '.local-artifacts' / IDENTITY
VALIDATION_COMMAND = 'python -u -m scripts.prepare_sealed_cohort --validate'
SMOKE_SUMMARY = ROOT / '.local-artifacts/issue-109-capture-smoke-v1/summary.json'
ISSUE_96_PLAN = ROOT / '.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/plan.json'
N1_PLAN = n1.ROOT / 'plan.json'
N1_DYNAMICS_PLAN = ROOT / '.local-artifacts/issue-77-n1-dynamics-v1/plan.json'
ISSUE_71_PLAN = ROOT / '.local-artifacts/issue-71-hybrid-readiness-v1/plan.json'
LEDGER_DOC = ROOT / 'docs/issue-77-novelty-experiments-plan.md'
N1_RESULTS_DOC = 'docs/issue-77-novelty-experiments-results.md'

FAMILIES = n1.FAMILIES
NOVELTY_LEVEL = 0
SPLITS = (('fit', 240), ('policy', 60), ('evaluation', 200))
SPLIT_ROLES = {
    'fit': 'predictor training',
    'policy': 'controller (policy) training',
    'evaluation': 'held-out evaluation; never fitted into any predictor or controller',
}
SPLIT_OFFSETS = {'fit': 0, 'policy': 100_000, 'evaluation': 200_000}
FAMILY_OFFSETS = {'type010103': 0, 'type010105': 50_000}
GENERATION_SEED_BASE = 765_000_000
ENGINE_SEED_BASE = 765_500_000
SEED_BLOCK = 50_000          # seeds base+1 .. base+SEED_BLOCK-1 per (split, family)
MINIMUM_SEED = 764_700_000
LEDGER_OPEN_BLOCK = 100_000  # an open-ended ledger base 'X+' reserves [X, X + 99_999]
SLOT_VOCABULARY_SIZE = 18
ALLOWED_BIRDS = ('BirdRed',)
ACTION_FIELDS = ('drag_x', 'drag_y', 'tap_time_ms', 'release_time_ms')
INVENTORY_ORDER = ('angle', 'grid', 'offset', 'power')
INVENTORY_PREFIX = {'angle': 'a', 'grid': 'g', 'offset': 'o', 'power': 'p'}
CANDIDATE_SETS = {
    'training': ('angle', 'grid'),
    'evaluation': ('angle', 'grid', 'offset', 'power'),
}
SPLIT_CANDIDATE_SET = {'fit': 'training', 'policy': 'training', 'evaluation': 'evaluation'}
EVALUATION_SUBSETS = {
    'union': ('angle', 'grid', 'offset', 'power'),
    'grid_power': ('grid', 'power'),
    'grid': ('grid',),
}
EVALUATION_PREFIXES = (20, 40, 80, 120, 200)
COST_KEYS = ('amortized_seconds_per_branch', 'per_branch_wall_p50', 'per_branch_wall_p90',
             'per_branch_wall_mean', 'workers', 'artifact_bytes_per_branch', 'branches_per_hour')
REJECTION_REASONS = (
    'reserved_seed_collision', 'materialization_error', 'slot_outside_frozen_vocabulary',
    'bird_not_birdred', 'duplicate_xml_content', 'duplicate_scenario_content',
    'duplicate_scenario_identity',
)


def log(message):
    print(f'[{IDENTITY}] {message}', flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def file_sha256(path):
    return 'sha256:' + sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def xml_digest(xml):
    return sha256(xml.encode('utf-8')).hexdigest()


# ------------------------------------------------------------------ seeds

def ledger_ranges(path=LEDGER_DOC):
    """Parse every seed range of the #77 disjointness ledger (section 2)."""
    text = Path(path).read_text()
    match = re.search(r'^## 2\. Seed/lineage disjointness ledger.*?(?=^## 3\.)', text, re.M | re.S)
    if not match:
        raise ValueError(f'seed ledger section missing from {path}')
    ranges = []
    number = r'(\d[\d_]*\d)'
    for line in match.group(0).splitlines():
        if line.startswith('|') and not line.startswith(('| Range', '| ---')):
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            name, spec = f'#77 ledger: {cells[1]}', cells[0]
        elif 'Suggested new bases' in line or re.search(number + r'\+', line):
            name, spec = '#77 ledger: suggested N1/N2/adaptation bases', line
        else:
            continue
        for low, high, open_ended in re.findall(number + r'(?:\s*[–-]\s*' + number + r')?(\+)?', spec):
            low = int(low.replace('_', ''))
            if low < 1_000_000:
                continue
            if high:
                high = int(high.replace('_', ''))
            elif open_ended:
                high = low + LEDGER_OPEN_BLOCK - 1
            else:
                high = low
            ranges.append((name, low, high))
    if len(ranges) < 10:
        raise ValueError('seed ledger parse found too few ranges; the ledger format changed')
    return ranges


def reserved_ranges():
    """Union of every declared reserved/used seed range, deduplicated by bounds."""
    candidates = list(n2n.RESERVED_RANGES) + list(n1.RESERVED_RANGES) + list(n2.RESERVED_RANGES)
    for label, module in (('issue-77 N1', n1), ('issue-77 N2', n2), ('issue-77 N2-normal', n2n)):
        candidates.append((f'{label} generation seed block', module.GENERATION_SEED_BASE,
                           module.GENERATION_SEED_BASE + LEDGER_OPEN_BLOCK - 1))
        candidates.append((f'{label} engine seed block', module.ENGINE_SEED_BASE,
                           module.ENGINE_SEED_BASE + LEDGER_OPEN_BLOCK - 1))
    candidates += ledger_ranges()
    unique = {}
    for name, low, high in candidates:
        unique.setdefault((low, high), name)
    return [{'name': name, 'low': low, 'high': high} for (low, high), name in sorted(unique.items())]


def seed_blocks():
    """This cohort's generation/engine seed blocks, one per (split, family)."""
    blocks = []
    for split, _ in SPLITS:
        for family in FAMILIES:
            offset = SPLIT_OFFSETS[split] + FAMILY_OFFSETS[family]
            for kind, base in (('generation', GENERATION_SEED_BASE), ('engine', ENGINE_SEED_BASE)):
                blocks.append({'name': f'issue-109 sealed cohort {split} {family} {kind} seeds',
                               'split': split, 'family': family, 'kind': kind,
                               'low': base + offset + 1, 'high': base + offset + SEED_BLOCK - 1})
    return blocks


def seeds(split, family, attempt):
    if not 1 <= attempt < SEED_BLOCK:
        raise ValueError(f'{split}/{family} exhausted its seed block')
    offset = SPLIT_OFFSETS[split] + FAMILY_OFFSETS[family] + attempt
    return GENERATION_SEED_BASE + offset, ENGINE_SEED_BASE + offset


def audit_seed_blocks(blocks, reserved):
    """Blocks are pairwise disjoint, >= MINIMUM_SEED and disjoint from every reserved range."""
    for index, block in enumerate(blocks):
        if block['low'] < MINIMUM_SEED:
            raise ValueError(f"{block['name']} lies below {MINIMUM_SEED}")
        for other in blocks[index + 1:]:
            if block['low'] <= other['high'] and other['low'] <= block['high']:
                raise ValueError(f"{block['name']} overlaps {other['name']}")
        for entry in reserved:
            if block['low'] <= entry['high'] and entry['low'] <= block['high']:
                raise ValueError(f"{block['name']} overlaps reserved range {entry['name']}")
    return {'blocks': len(blocks), 'reserved_ranges_checked': len(reserved),
            'minimum_seed': MINIMUM_SEED, 'passed': True}


# ---------------------------------------------------------- prior exposure

def prior_exposure(frozen_paths=None):
    """#77 N1 prior-exposure projection plus the N1 plan itself (which N1 excludes as its own).

    With ``frozen_paths`` (validation), returns (frozen entries, post-freeze entries).
    """
    entries = {entry['path']: entry for entry in n1.prior_exposure()}
    path = relative(N1_PLAN)
    value = {key: item for key, item in files.read(N1_PLAN).items() if key != 'prior_exposure'}
    entries[path] = {'path': path, **files.exposure_projection(value), **n1._prior_details(value)}
    if frozen_paths is None:
        return [entries[key] for key in sorted(entries)], []
    missing = sorted(set(frozen_paths) - set(entries))
    if missing:
        raise ValueError(f'frozen prior-exposure sources disappeared: {missing}')
    return ([entries[key] for key in sorted(frozen_paths)],
            [entries[key] for key in sorted(set(entries) - set(frozen_paths))])


def prior_index(prior):
    return {
        'seeds': {seed for entry in prior for seed in entry['generation_or_reserved_seeds']},
        'scenario_identities': {item for entry in prior for item in entry['scenario_identities']},
        'scenario_content': {item for entry in prior for item in entry.get('scenario_content_sha256', ())},
        'xml_content': {item for entry in prior for item in entry.get('xml_content_sha256', ())},
    }


def prior_summary(prior):
    return [{'path': entry['path'], 'sha256': file_sha256(ROOT / entry['path']),
             'generation_or_reserved_seeds': len(entry['generation_or_reserved_seeds']),
             'member_identities': len(entry.get('member_identities', ())),
             'scenario_identities': len(entry['scenario_identities']),
             'scenario_content_sha256': len(entry.get('scenario_content_sha256', ())),
             'xml_content_sha256': len(entry.get('xml_content_sha256', ()))}
            for entry in prior]


# ------------------------------------------------------------ screening

def slot_vocabulary(path=ISSUE_71_PLAN):
    """The frozen 18-slot checkpoint vocabulary (issue-71 contract, used by N1 training)."""
    vocabulary = files.read(path)['contract']['vocabulary']
    if (len(vocabulary) != SLOT_VOCABULARY_SIZE or len(set(vocabulary)) != SLOT_VOCABULARY_SIZE
            or not set(vocabulary) <= set(VOCABULARY)):
        raise ValueError('the frozen checkpoint vocabulary is not an 18-slot subset of the native vocabulary')
    return list(vocabulary)


def materialize_level(family, generation_seed, template):
    """In-memory materialization (publish=False writes nothing): (xml text, scenario dict)."""
    member = {'identity': f'issue-109-sc-probe-{generation_seed}', 'generation_seed': generation_seed,
              'generator_family': family, 'novelty_level': NOVELTY_LEVEL}
    generated, scenario = files.materialize(member, template, OUTPUT / 'unpublished')
    return generated.xml_content.decode('utf-8'), scenario.to_dict()


def static_fit(xml, vocabulary):
    """Rule 1 (the N1 slot binder, against the frozen 18-slot vocabulary): (reason, detail, slots)."""
    tree = ET.fromstring(xml)
    slots = sorted(node.attrib['scenarioObjectId'] for node in tree.iter()
                   if 'scenarioObjectId' in node.attrib)
    outside = sorted(set(slots) - set(vocabulary))
    if outside:
        return 'slot_outside_frozen_vocabulary', {'slots': outside}, slots
    birds = sorted({bird.attrib.get('type') for bird in tree.findall('./Birds/Bird')} - set(ALLOWED_BIRDS))
    if birds:
        return 'bird_not_birdred', {'bird_types': birds}, slots
    return None, None, slots


def build_rosters(templates, prior, vocabulary, sizes=dict(SPLITS), materialize=materialize_level):
    """Screen seeds in fixed order; returns (splits, rejected, bound members for the audit)."""
    index = prior_index(prior)
    owners = {'xml': {}, 'scenario': {}, 'scenario_identity': {}}
    splits, rejected, bound = {}, [], []
    for split, _ in SPLITS:
        splits[split] = {}
        for family in FAMILIES:
            roster, attempt = [], 0
            while len(roster) < sizes[split]:
                attempt += 1
                generation_seed, engine_seed = seeds(split, family, attempt)
                base = {'split': split, 'family': family, 'attempt': attempt,
                        'generation_seed': generation_seed, 'engine_seed': engine_seed}
                reason, detail = None, None
                if {generation_seed, engine_seed} & index['seeds']:
                    reason, detail = 'reserved_seed_collision', {}
                else:
                    try:
                        xml, scenario = materialize(family, generation_seed, templates[family])
                    except Exception as error:  # generator failure is outcome-free; record it
                        reason, detail = 'materialization_error', {'error': f'{type(error).__name__}: {error}'}
                if reason is None:
                    reason, detail, slots = static_fit(xml, vocabulary)
                if reason is None:
                    xml_sha = xml_digest(xml)
                    scenario_sha = n1._digest(scenario)
                    identities = files.exposure_projection(scenario)['scenario_identities']
                    for key, digest, prior_set, name in (
                            ('xml', xml_sha, index['xml_content'], 'duplicate_xml_content'),
                            ('scenario', scenario_sha, index['scenario_content'], 'duplicate_scenario_content')):
                        if digest in owners[key] or digest in prior_set:
                            reason = name
                            detail = {'duplicates': owners[key].get(digest, 'prior exposure')}
                            break
                    if reason is None:
                        clash = [item for item in identities
                                 if item in owners['scenario_identity'] or item in index['scenario_identities']]
                        if clash:
                            reason = 'duplicate_scenario_identity'
                            detail = {'duplicates': owners['scenario_identity'].get(clash[0], 'prior exposure')}
                if reason is not None:
                    rejected.append({**base, 'reason': reason, 'detail': detail})
                    continue
                identity = f'issue-109-sc-{split}-{family}-{len(roster) + 1:03}'
                owners['xml'][xml_sha] = identity
                owners['scenario'][scenario_sha] = identity
                owners['scenario_identity'].update({item: identity for item in identities})
                roster.append({'identity': identity, 'ordinal': len(roster) + 1,
                               'generation_seed': generation_seed, 'engine_seed': engine_seed,
                               'generator_family': family, 'novelty_level': NOVELTY_LEVEL,
                               'split': split, 'candidate_set': SPLIT_CANDIDATE_SET[split],
                               'xml_sha256': xml_sha, 'scenario_sha256': scenario_sha,
                               'generated_slots': slots,
                               'xml_path': f'levels/{identity}.xml',
                               'scenario_path': f'levels/{identity}.scenario.json'})
                bound.append({'identity': identity, 'base_cluster': identity,
                              'generation_seed': generation_seed, 'engine_seed': engine_seed,
                              'scenario': scenario, 'xml': xml})
            splits[split][family] = roster
    return splits, rejected, bound


def split_disjointness(splits):
    """Every level identity, seed and content digest belongs to exactly one split."""
    fields = ('identity', 'generation_seed', 'engine_seed', 'xml_sha256', 'scenario_sha256')
    seen = {field: {} for field in fields}
    for split, rosters in splits.items():
        for family, roster in rosters.items():
            ordered = [level['generation_seed'] for level in roster]
            if ordered != sorted(ordered) or [level['ordinal'] for level in roster] != list(range(1, len(roster) + 1)):
                raise ValueError(f'{split}/{family} roster is not in generation-seed order')
            for level in roster:
                for field in fields:
                    owner = seen[field].setdefault(level[field], (split, level['identity']))
                    if owner != (split, level['identity']):
                        raise ValueError(f'{field} {level[field]} shared by {owner} and {level["identity"]}')
    generation = set(seen['generation_seed'])
    if generation & set(seen['engine_seed']):
        raise ValueError('generation and engine seeds overlap')
    return {'levels': len(seen['identity']), 'fields': list(fields), 'splits_level_disjoint': True}


# ------------------------------------------------------------ candidates

def action_key(action):
    return tuple(action[field] for field in ACTION_FIELDS)


def inventories():
    """The four frozen #87/#93/#94 inventories (power: nominal actions)."""
    offset, excluded = p93.offset_inventory()
    raw = {
        'angle': [{'ordinal': i, 'action': action} for i, action in enumerate(n1.candidate_actions())],
        'grid': p93.grid_inventory(),
        'offset': offset,
        'power': p94.design_inventory(),
    }
    expected = {'angle': 13, 'grid': 16, 'offset': 11, 'power': 20}
    if {name: len(items) for name, items in raw.items()} != expected or len(excluded) != 1:
        raise ValueError('candidate inventories differ from the frozen #87/#93/#94 designs')
    return {name: [{'label': f"{INVENTORY_PREFIX[name]}{item['ordinal']:02}", 'ordinal': item['ordinal'],
                    'action': {field: item['action'][field] for field in ACTION_FIELDS}}
                   for item in items]
            for name, items in raw.items()}


def candidate_set(names, inventory):
    """Deduplicate by exact action tuple; one suffix cNN per unique action, first occurrence order."""
    by_key, candidates, membership = {}, [], {}
    for name in names:
        membership[name] = []
        for item in inventory[name]:
            key = action_key(item['action'])
            if key not in by_key:
                by_key[key] = len(candidates)
                candidates.append({'suffix': f'c{len(candidates):02}', 'ordinal': len(candidates),
                                   'action': dict(zip(ACTION_FIELDS, key)), 'inventory_entries': []})
            candidate = candidates[by_key[key]]
            candidate['inventory_entries'].append({'inventory': name, 'label': item['label'],
                                                   'ordinal': item['ordinal']})
            membership[name].append(candidate['suffix'])
    if len(candidates) > 100:
        raise ValueError('candidate suffix cNN supports at most 100 candidates')
    entries = sum(len(inventory[name]) for name in names)
    return {'inventories': list(names), 'candidates': candidates, 'membership': membership,
            'inventory_entries': entries, 'unique_candidates': len(candidates),
            'duplicate_entries_collapsed': entries - len(candidates)}


def candidate_sets(inventory=None):
    inventory = inventory or inventories()
    sets = {name: candidate_set(names, inventory) for name, names in CANDIDATE_SETS.items()}
    training, evaluation = sets['training']['candidates'], sets['evaluation']['candidates']
    if [(c['suffix'], c['action']) for c in training] != [(c['suffix'], c['action']) for c in evaluation[:len(training)]]:
        raise ValueError('training candidates must be the identity-stable prefix of the evaluation set')
    return sets


def level_candidates(level, cset):
    """Expanded branch identities of one level, keyed by (level identity, engine seed, action)."""
    rows = []
    for candidate in cset['candidates']:
        key = {'level_identity': level['identity'], 'engine_seed': level['engine_seed'],
               'action': candidate['action']}
        rows.append({'identity': f"{level['identity']}-{candidate['suffix']}",
                     'level_identity': level['identity'], 'engine_seed': level['engine_seed'],
                     'action': candidate['action'], 'candidate_key_sha256': n1._digest(key),
                     'inventory_entries': candidate['inventory_entries']})
    return rows


def candidate_identity_digests(splits, sets):
    digests = {}
    for split, rosters in splits.items():
        rows = [level_candidates(level, sets[SPLIT_CANDIDATE_SET[split]])
                for family in FAMILIES for level in rosters[family]]
        identities = [row['identity'] for group in rows for row in group]
        if len(identities) != len(set(identities)):
            raise ValueError(f'{split} candidate identities are not unique')
        digests[split] = {'branches': len(identities), 'sha256': n1._digest(rows)}
    return digests


# ------------------------------------------------------- expected yield

def expected_yield():
    """Descriptive planning numbers from exposed #96 development evidence only."""
    universe = files.read(ISSUE_96_PLAN)['universe']
    family_of = {member['identity']: member['generator_family'] for member in files.read(N1_PLAN)['members']}

    def share(members, mixed):
        return {'members': len(members), 'mixed_members': len(mixed),
                'mixed_share': round(len(mixed) / len(members), 4) if members else None}

    rows = {}
    for name in INVENTORY_ORDER:
        members, mixed = universe['members'][name], set(universe['mixed_members'][name])
        per_family = {family: share([m for m in members if family_of[m] == family],
                                    [m for m in members if family_of[m] == family and m in mixed])
                      for family in FAMILIES}
        rows[name] = {**share(members, [m for m in members if m in mixed]), 'per_family': per_family}
    all_members = sorted({m for name in INVENTORY_ORDER for m in universe['members'][name]})
    any_mixed = {m for name in INVENTORY_ORDER for m in universe['mixed_members'][name]}
    rows['any_inventory'] = share(all_members, [m for m in all_members if m in any_mixed])
    projected = {
        str(prefix): {name: round(rows[name]['mixed_share'] * prefix * len(FAMILIES), 1)
                      for name in (*INVENTORY_ORDER, 'any_inventory')}
        for prefix in EVALUATION_PREFIXES}
    return {
        'label': 'DESCRIPTIVE planning numbers; not a prediction, not a verdict',
        'source': {'artifact': relative(ISSUE_96_PLAN), 'sha256': file_sha256(ISSUE_96_PLAN),
                   'fields': 'universe.members vs universe.mixed_members',
                   'mixed_rule': universe['mixed_rule'],
                   'family_source': f"{relative(N1_PLAN)} members[].generator_family"},
        'evidence_scope': ('exposed #77 N1 development members (14-15 per inventory; same families and '
                           'templates as this cohort); includes predictor/controller-training lineages; '
                           'no sealed or fresh level contributed'),
        'member_level_mixed_share': rows,
        'expected_mixed_evaluation_levels_by_prefix': {
            'rule': 'mixed_share x prefix levels per family x 2 families (both families pooled share)',
            'values': projected,
        },
        'caveat': ('shares were measured on development states that later informed method design; '
                   'fresh evaluation levels may yield fewer mixed members. #104 reports the realized '
                   'yield per level type.'),
    }


# ---------------------------------------------------------------- cost

def load_cost(cost_source=None):
    if cost_source is None:
        path, kind = SMOKE_SUMMARY, 'capture_smoke_summary'
        if not path.is_file():
            raise ValueError(f'{relative(path)} is not published; the plan freezes only with measured '
                             'throughput (use --cost-source for development runs)')
        throughput = files.read(path).get('throughput')
        field = 'throughput'
    else:
        path, kind = Path(cost_source), 'development_override'
        throughput = files.read(path)
        field = '<top level>'
    if not isinstance(throughput, dict):
        raise ValueError(f'{path} has no throughput section')
    missing = [key for key in COST_KEYS if not isinstance(throughput.get(key), (int, float))
               or isinstance(throughput.get(key), bool) or throughput[key] <= 0]
    if missing:
        raise ValueError(f'{path} throughput lacks positive numeric keys: {missing}')
    return {'source': {'kind': kind, 'path': relative(path), 'sha256': file_sha256(path), 'field': field},
            'throughput': {key: throughput[key] for key in COST_KEYS}}


def budget_row(levels, candidates_per_level, throughput):
    branches = levels * candidates_per_level
    return {
        'levels': levels,
        'candidates_per_level': candidates_per_level,
        'branches': branches,
        'wall_hours': round(branches * throughput['amortized_seconds_per_branch'] / 3600, 2),
        'wall_hours_p90_bound': round(branches * throughput['per_branch_wall_p90']
                                      / throughput['workers'] / 3600, 2),
        'worker_hours': round(branches * throughput['per_branch_wall_mean'] / 3600, 2),
        'artifact_bytes': round(branches * throughput['artifact_bytes_per_branch']),
        'artifact_gib': round(branches * throughput['artifact_bytes_per_branch'] / 2**30, 2),
    }


def budget(cost, sets, inventory):
    throughput = cost['throughput']
    full = {split: budget_row(size * len(FAMILIES), sets[SPLIT_CANDIDATE_SET[split]]['unique_candidates'],
                              throughput)
            for split, size in SPLITS}
    total = {key: round(sum(row[key] for row in full.values()), 2)
             for key in ('levels', 'branches', 'wall_hours', 'wall_hours_p90_bound', 'worker_hours',
                         'artifact_bytes', 'artifact_gib')}
    subsets = {name: candidate_set(names, inventory)['unique_candidates']
               for name, names in EVALUATION_SUBSETS.items()}
    prefixes = {name: {str(prefix): budget_row(prefix * len(FAMILIES), count, throughput)
                       for prefix in EVALUATION_PREFIXES}
                for name, count in subsets.items()}
    return {
        'units': ('wall_hours = branches x amortized_seconds_per_branch (campaign wall per branch at the '
                  'smoke worker count); wall_hours_p90_bound = branches x per_branch_wall_p90 / workers; '
                  'worker_hours = branches x per_branch_wall_mean; artifact bytes = branches x '
                  'artifact_bytes_per_branch'),
        'workers': throughput['workers'],
        'full_rosters': {**full, 'total': total},
        'evaluation_prefix_levels_per_family': list(EVALUATION_PREFIXES),
        'evaluation_candidate_subsets': {name: {'inventories': list(EVALUATION_SUBSETS[name]),
                                                'unique_candidates': count}
                                         for name, count in subsets.items()},
        'evaluation_prefixes': prefixes,
    }


# ------------------------------------------------------ training coverage

def training_coverage(sets):
    issue71 = files.read(ISSUE_71_PLAN)
    counts71 = Counter()
    for record in issue71['records']:
        found = set(re.findall(r'type\d{6}', record['scenario_lineage_identity']))
        if len(found) != 1:
            raise ValueError('issue-71 record does not name exactly one generator family')
        counts71[found.pop()] += 1
    dynamics = files.read(N1_DYNAMICS_PLAN)['n1_lineages']['records']
    n1_rows = Counter()
    for record in dynamics.values():
        n1_rows[(record['generator_family'], record['fit_partition'], 'lineages')] += 1
        n1_rows[(record['generator_family'], record['fit_partition'], 'branches')] += len(record['branches'])
    training_candidates = sets['training']['unique_candidates']
    families = sorted(set(counts71) | set(FAMILIES))
    table = {}
    for family in families:
        new = {split: size if family in FAMILIES else 0 for split, size in SPLITS if split != 'evaluation'}
        table[family] = {
            'issue71_lineages': counts71.get(family, 0),
            'n1_predictor_lineages': n1_rows[(family, 'predictor', 'lineages')],
            'n1_predictor_branches': n1_rows[(family, 'predictor', 'branches')],
            'n1_controller_lineages': n1_rows[(family, 'controller', 'lineages')],
            'n1_controller_branches': n1_rows[(family, 'controller', 'branches')],
            'sealed_fit_levels': new['fit'],
            'sealed_fit_branches': new['fit'] * training_candidates,
            'sealed_policy_levels': new['policy'],
            'sealed_policy_branches': new['policy'] * training_candidates,
            'evaluated_family': family in FAMILIES,
        }
    for family in FAMILIES:
        if not (table[family]['sealed_fit_levels'] and table[family]['sealed_policy_levels']):
            raise ValueError(f'evaluation family {family} lacks fit or policy coverage')
    return {
        'table': table,
        'existing_corpus': {
            'issue71': {'artifact': relative(ISSUE_71_PLAN), 'lineages': len(issue71['records']),
                        'predictor_lineages': issue71['data']['predictor_lineages'],
                        'controller_lineages': issue71['data']['controller_lineage_count'],
                        'families': sorted(counts71)},
            'n1_training_role': {'artifact': relative(N1_DYNAMICS_PLAN),
                                 'predictor_lineages': sum(v for (f, p, k), v in n1_rows.items()
                                                           if p == 'predictor' and k == 'lineages'),
                                 'controller_lineages': sum(v for (f, p, k), v in n1_rows.items()
                                                            if p == 'controller' and k == 'lineages'),
                                 'windows': 504, 'windows_source': f'{N1_RESULTS_DOC} section N1'},
        },
        'statement': ('both evaluation families (type010103, type010105) are represented in the sealed fit '
                      'and policy splits; the issue-71 corpus holds only type010101/type010102, and the prior '
                      'rolling/sliding training data was 12 N1 training-role lineages (504 windows)'),
        'training_support': ('fit/policy levels carry angle(13) + grid(16) = the training behaviour '
                             'inventory, so the primary evaluated grid inventory lies inside training '
                             'support; offset and power are evaluated only (power is out of training '
                             'support per #95: secondary)'),
    }


# ------------------------------------------------------------------ plan

def derive(cost_source=None, frozen_prior_paths=None, require_cost=True):
    """Deterministic derivation: (plan without frozen_at, level files, post-freeze prior entries)."""
    try:
        cost = load_cost(cost_source)
    except ValueError:
        if require_cost:
            raise
        cost = None  # dry run only: inventory without a budget
    reserved = reserved_ranges()
    blocks = seed_blocks()
    seed_audit = audit_seed_blocks(blocks, reserved)
    prior, later = prior_exposure(frozen_prior_paths)
    templates = n1.template_sources()
    vocabulary = slot_vocabulary()
    splits, rejected, bound = build_rosters(templates, prior, vocabulary)
    disjointness = split_disjointness(splits)
    audit = n1.audit_disjointness(bound, prior)
    inventory = inventories()
    sets = candidate_sets(inventory)
    level_files = {}
    for member in bound:
        level_files[f"levels/{member['identity']}.xml"] = member['xml']
        level_files[f"levels/{member['identity']}.scenario.json"] = json_text(member['scenario'])
    reasons = Counter(row['reason'] for row in rejected)
    plan = {
        'schema': SCHEMA,
        'identity': IDENTITY,
        'version': 1,
        'issue': 109,
        'consumer_issue': 104,
        'frozen_before_any_capture': True,
        'validation_command': VALIDATION_COMMAND,
        'families': [{'family': family, 'novelty_level': NOVELTY_LEVEL,
                      'mechanism': templates[family]['mechanism'],
                      'template_path': templates[family]['template'],
                      'template': templates[family]} for family in FAMILIES],
        'family_rationale': 'exactly the families the ICLR #96 contrasts evaluate (#77 N1 templates)',
        'template_source': {'artifact': relative(n1.NOVELTY_INVENTORY), 'sha256': file_sha256(n1.NOVELTY_INVENTORY),
                            'schema': 'issue_76_novelty_inventory_v1'},
        'split_roles': SPLIT_ROLES,
        'roster_sizes_per_family': dict(SPLITS),
        'splits': splits,
        'counts': {
            'levels': {split: {family: len(splits[split][family]) for family in FAMILIES} for split, _ in SPLITS},
            'rejected_seeds': len(rejected),
            'rejections_by_reason': {reason: reasons.get(reason, 0) for reason in REJECTION_REASONS},
            'rejections_by_split_family': {f'{split}/{family}': sum(1 for row in rejected
                                                                     if row['split'] == split and row['family'] == family)
                                           for split, _ in SPLITS for family in FAMILIES},
        },
        'rejected_seeds': rejected,
        'screening_rules': {
            'declared_before_any_verdict': True,
            'outcome_free': True,
            'order': 'fit -> policy -> evaluation; type010103 -> type010105; ascending seed attempt',
            'rules': [
                {'rule': 'reserved_seed', 'reject': 'generation or engine seed in any prior-exposure seed set'},
                {'rule': 'static_slot_fit', 'reject': ('any generated scenarioObjectId outside the frozen 18-slot '
                                                       'checkpoint vocabulary, or any bird not BirdRed'),
                 'vocabulary': vocabulary, 'vocabulary_source': f"{relative(ISSUE_71_PLAN)} contract.vocabulary",
                 'allowed_birds': list(ALLOWED_BIRDS)},
                {'rule': 'content_dedup', 'reject': ('xml sha256, scenario sha256 (canonical JSON) or scenario '
                                                     'identity equal to an earlier roster level or any prior exposure')},
                {'rule': 'materialization_error', 'reject': 'the generator raised for this seed'},
                {'rule': 'candidate_densification', 'effect': ('evaluation levels carry the deduplicated union '
                                                               'angle+grid+offset+power; fit/policy carry angle+grid')},
            ],
            'rejected_seed_policy': 'recorded with reason; the roster continues with the next seed in its block',
            'prefix_capture': ('#104 captures an ordered prefix (ordinal 1..k) of each roster; prefix '
                               'membership is fixed by roster order, never by outcomes'),
        },
        'seed_contract': {
            'generation_seed_formula': (f'{GENERATION_SEED_BASE} + split offset + family offset + attempt '
                                        f'(attempt 1..{SEED_BLOCK - 1})'),
            'engine_seed_formula': f'{ENGINE_SEED_BASE} + split offset + family offset + attempt',
            'split_offsets': SPLIT_OFFSETS,
            'family_offsets': FAMILY_OFFSETS,
            'reserved_by_this_plan': blocks,
            'prior_reserved_ranges': reserved,
            'prior_reserved_sources': ['scripts/prepare_issue_77_n2n.py RESERVED_RANGES',
                                       'scripts/prepare_issue_77_n1.py RESERVED_RANGES',
                                       'scripts/prepare_issue_77_n2.py RESERVED_RANGES',
                                       'issue-77 N1/N2/N2-normal module seed bases (100k blocks)',
                                       f'{relative(LEDGER_DOC)} section 2 ledger (open-ended bases = 100k blocks)'],
            'audit': seed_audit,
        },
        'disjointness': {
            'splits': disjointness,
            'prior_exposure_audit': audit,
            'collision_classes': ['reserved_seed', 'lineage_identity', 'scenario_identity',
                                  'scenario_content', 'xml_content'],
        },
        'prior_exposure': {'sources': prior_summary(prior), 'count': len(prior),
                           'projection': 'scripts/prepare_issue_77_n1.prior_exposure + the #77 N1 plan'},
        'candidate_inventories': {
            'action_fields': list(ACTION_FIELDS),
            'dedup_rule': 'exact action tuple (drag_x, drag_y, tap_time_ms, release_time_ms)',
            'identity_rule': ('<level-identity>-c<NN>; one identity per unique action of the level; keyed by '
                              '(level identity, engine seed, action); an inventory entry duplicating another '
                              "inventory's action maps to the same candidate id"),
            'inventories': inventory,
            'sources': {'angle': 'scripts/prepare_issue_77_n1.candidate_actions (#87 radius-80 sweep)',
                        'grid': 'scripts/run_second_parameterization_probe.grid_inventory (#93 Arm B)',
                        'offset': 'scripts/run_second_parameterization_probe.offset_inventory (#93 Arm A)',
                        'power': ('scripts/run_launch_power_probe.design_inventory (#94, nominal actions; '
                                  'out of training support per #95: secondary)')},
            'sets': sets,
            'split_candidate_set': SPLIT_CANDIDATE_SET,
            'expanded_identity_digests': candidate_identity_digests(splits, sets),
        },
        'expected_yield': expected_yield(),
        'cost': cost,
        'budget': budget(cost, sets, inventory) if cost else None,
        'training_coverage': training_coverage(sets),
        'inputs': [{'name': name, 'artifact': relative(path), 'sha256': file_sha256(path)}
                   for name, path in (('issue96_plan', ISSUE_96_PLAN), ('n1_plan', N1_PLAN),
                                      ('n1_dynamics_plan', N1_DYNAMICS_PLAN), ('issue71_plan', ISSUE_71_PLAN),
                                      ('seed_ledger', LEDGER_DOC), ('novelty_inventory', n1.NOVELTY_INVENTORY))],
        'handoff_104': {
            'fixed_here': ['split membership and roster order', 'seeds', 'level XML/scenario (hash-referenced)',
                           'candidate identities and inventory membership', 'screening rules'],
            'fixed_by_104_before_any_verdict': [
                'prefix depth per split and family from its #96-variance power analysis (delta 0.02 and 0.05)',
                'which evaluation candidate subset to capture (union / grid+power / grid)',
                'capture campaign limits and worker count'],
            'capture_pipeline': 'scripts/capture_pipeline_v2.py (issue-109 fixed pipeline)',
            'evaluation_rule': 'evaluation-split captures are never fitted; #96 contrasts rerun on the evaluation split only',
            'branch_expansion': 'scripts/prepare_sealed_cohort.level_candidates(level, sets[level.candidate_set])',
            'level_files': 'levels/<identity>.xml and levels/<identity>.scenario.json, verified by xml_sha256/scenario_sha256',
        },
        'claim_boundary': ('frozen design only: level-disjoint splits, outcome-free screening, deduplicated '
                           'candidate identities, descriptive yield and cost planning numbers; no engine ran, '
                           'no outcome was read, no model trained or scored; prior dispositions (#15-#98) '
                           'are read-only inputs'),
    }
    return plan, level_files, later


def summary(plan):
    evaluation = plan['candidate_inventories']['sets']
    return {
        'identity': plan['identity'],
        'levels': plan['counts']['levels'],
        'rejected_seeds': plan['counts']['rejected_seeds'],
        'rejections_by_reason': {k: v for k, v in plan['counts']['rejections_by_reason'].items() if v},
        'rejections_by_split_family': plan['counts']['rejections_by_split_family'],
        'unique_candidates': {name: value['unique_candidates'] for name, value in evaluation.items()},
        'branches': {split: value['branches']
                     for split, value in plan['candidate_inventories']['expanded_identity_digests'].items()},
        'prior_exposure_sources': plan['prior_exposure']['count'],
        'cost_source': plan['cost']['source'] if plan['cost'] else None,
        'budget_total': plan['budget']['full_rosters']['total'] if plan['budget'] else None,
    }


def dry_run():
    plan, level_files, _ = derive(require_cost=False)
    value = summary(plan)
    value['level_files'] = len(level_files)
    value['writes'] = 'none'
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def prepare(output=OUTPUT, cost_source=None):
    output = Path(output)
    if cost_source is not None and output.resolve() == OUTPUT.resolve():
        raise ValueError('the real sealed plan freezes only with the published capture-smoke throughput; '
                         '--cost-source is for development outputs')
    if (output / 'plan.json').exists() or (output / 'levels').exists():
        raise ValueError(f'{output} already holds a frozen plan or level files; refusing to overwrite')
    plan, level_files, _ = derive(cost_source=cost_source)
    plan['frozen_at'] = utc_now()
    (output / 'levels').mkdir(parents=True)
    for name, text in sorted(level_files.items()):
        (output / name).write_text(text)
    (output / 'plan.json').write_text(json_text(plan))
    value = summary(plan)
    value['output'] = str(output)
    value['frozen_at'] = plan['frozen_at']
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)
    return plan


def validate(output=OUTPUT):
    output = Path(output)
    text = (output / 'plan.json').read_text()
    saved = json.loads(text)
    if saved.get('identity') != IDENTITY or saved.get('schema') != SCHEMA:
        raise ValueError('plan identity/schema differs from this runner')
    source = saved['cost']['source']
    cost_source = None if source['kind'] == 'capture_smoke_summary' else ROOT / source['path']
    frozen_paths = [entry['path'] for entry in saved['prior_exposure']['sources']]
    plan, level_files, later = derive(cost_source=cost_source, frozen_prior_paths=frozen_paths)
    plan['frozen_at'] = saved['frozen_at']
    problems = []
    if json_text(plan) != text:
        differing = sorted(key for key in set(plan) | set(saved) if plan.get(key) != saved.get(key))
        problems.append(f'plan.json differs from the re-derivation in: {differing}')
    present = {str(path.relative_to(output)) for path in (output / 'levels').iterdir()}
    if present != set(level_files):
        problems.append(f'level file set differs: {len(present ^ set(level_files))} names')
    for name, content in level_files.items():
        if name in present and (output / name).read_text() != content:
            problems.append(f'{name} differs from the re-derivation')
    for split, rosters in saved['splits'].items():
        for roster in rosters.values():
            for level in roster:
                if xml_digest((output / level['xml_path']).read_text()) != level['xml_sha256']:
                    problems.append(f"{level['xml_path']} does not match xml_sha256")
    bound = [{'identity': level['identity'], 'base_cluster': level['identity'],
              'generation_seed': level['generation_seed'], 'engine_seed': level['engine_seed'],
              'scenario': json.loads((output / level['scenario_path']).read_text()),
              'xml': (output / level['xml_path']).read_text()}
             for rosters in saved['splits'].values() for roster in rosters.values() for level in roster]
    split_disjointness(saved['splits'])
    n1.audit_disjointness(bound, later)  # post-freeze exposures must not collide either
    value = {'identity': IDENTITY, 'validated': not problems, 'problems': problems[:20],
             'levels_checked': len(bound), 'post_freeze_prior_sources': [entry['path'] for entry in later],
             'frozen_at': saved['frozen_at']}
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('dry-run', 'prepare', 'validate'):
        modes.add_argument(f'--{mode}', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='output root (default: the frozen artifact)')
    parser.add_argument('--cost-source', type=Path, default=None,
                        help='development-only JSON with the throughput keys at top level')
    args = parser.parse_args()
    try:
        if args.dry_run:
            dry_run()
        elif args.prepare:
            prepare(args.output, args.cost_source)
        elif not validate(args.output):
            sys.exit(1)
    except Exception as error:
        log(f'error: {type(error).__name__}: {error}')
        sys.exit(1)


if __name__ == '__main__':
    main()
