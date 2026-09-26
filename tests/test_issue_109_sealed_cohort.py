"""Issue-109 item 4: sealed cohort invariants (split disjointness, screening, candidate dedup, seeds)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import prepare_sealed_cohort as sc

AVAILABLE = (sc.n1.NOVELTY_INVENTORY.is_file() and sc.ISSUE_71_PLAN.is_file()
             and sc.ISSUE_96_PLAN.is_file() and sc.N1_PLAN.is_file() and sc.N1_DYNAMICS_PLAN.is_file())
VOCABULARY = ['bird:0000', 'block:0000', 'pig:0000', 'slingshot:0000']
SMALL = {'fit': 3, 'policy': 2, 'evaluation': 3}
COST = {'amortized_seconds_per_branch': 90.0, 'per_branch_wall_p50': 450.0, 'per_branch_wall_p90': 700.0,
        'per_branch_wall_mean': 500.0, 'workers': 6, 'artifact_bytes_per_branch': 1000,
        'branches_per_hour': 40.0}


def fake_level(xml_key, slots=('bird:0000', 'block:0000', 'pig:0000'), bird='BirdRed'):
    objects = ''.join(f'<Block scenarioObjectId="{slot}" key="{xml_key}"/>' for slot in slots[1:])
    return (f'<Level><Birds><Bird type="{bird}" scenarioObjectId="{slots[0]}"/></Birds>'
            f'<GameObjects>{objects}</GameObjects></Level>'), {'content': xml_key}


class SeedTests(unittest.TestCase):
    def test_seed_blocks_are_disjoint_from_the_reserved_ledger(self):
        reserved = sc.reserved_ranges()
        bounds = {(entry['low'], entry['high']) for entry in reserved}
        # rows of the #77 ledger table and the pinned module ranges are all present
        for expected in ((761700001, 761700700), (20260908, 20260910), (764000000, 764099999),
                         (764400000, 764400099), (764500000, 764599999)):
            self.assertIn(expected, bounds)
        blocks = sc.seed_blocks()
        self.assertTrue(sc.audit_seed_blocks(blocks, reserved)['passed'])
        self.assertTrue(all(block['low'] >= sc.MINIMUM_SEED for block in blocks))
        for split, _ in sc.SPLITS:
            for family in sc.FAMILIES:
                generation, engine = sc.seeds(split, family, sc.SEED_BLOCK - 1)
                owners = [b for b in blocks if b['low'] <= generation <= b['high'] or b['low'] <= engine <= b['high']]
                self.assertEqual({(b['split'], b['family']) for b in owners}, {(split, family)})
        with self.assertRaises(ValueError):
            sc.seeds('fit', sc.FAMILIES[0], sc.SEED_BLOCK)

    def test_overlapping_reserved_range_is_rejected(self):
        blocks = sc.seed_blocks()
        clash = [{'name': 'someone else', 'low': blocks[-1]['high'], 'high': blocks[-1]['high'] + 5}]
        with self.assertRaises(ValueError):
            sc.audit_seed_blocks(blocks, clash)


class CandidateTests(unittest.TestCase):
    def test_duplicate_actions_share_one_identity(self):
        action = lambda x, y: {'drag_x': x, 'drag_y': y, 'tap_time_ms': 0, 'release_time_ms': 1000}
        inventory = {'first': [{'label': 'f00', 'ordinal': 0, 'action': action(-80, 10)},
                               {'label': 'f01', 'ordinal': 1, 'action': action(-40, 5)}],
                     'second': [{'label': 's00', 'ordinal': 0, 'action': action(-40, 5)},
                                {'label': 's01', 'ordinal': 1, 'action': action(-40, 6)}]}
        value = sc.candidate_set(('first', 'second'), inventory)
        self.assertEqual(value['unique_candidates'], 3)
        self.assertEqual(value['duplicate_entries_collapsed'], 1)
        self.assertEqual(value['membership'], {'first': ['c00', 'c01'], 'second': ['c01', 'c02']})
        rows = sc.level_candidates({'identity': 'lvl', 'engine_seed': 7}, value)
        self.assertEqual([row['identity'] for row in rows], ['lvl-c00', 'lvl-c01', 'lvl-c02'])
        other_seed = sc.level_candidates({'identity': 'lvl', 'engine_seed': 8}, value)
        self.assertNotEqual(rows[0]['candidate_key_sha256'], other_seed[0]['candidate_key_sha256'])

    def test_frozen_sets_map_every_inventory_entry_to_its_exact_action(self):
        inventory = sc.inventories()
        sets = sc.candidate_sets(inventory)
        self.assertEqual(sets['training']['unique_candidates'], 29)
        self.assertEqual(sets['evaluation']['unique_candidates'], 60)
        for cset in sets.values():
            by_suffix = {c['suffix']: c['action'] for c in cset['candidates']}
            self.assertEqual(len({sc.action_key(a) for a in by_suffix.values()}), len(by_suffix))
            for name in cset['inventories']:
                for item, suffix in zip(inventory[name], cset['membership'][name], strict=True):
                    self.assertEqual(by_suffix[suffix], item['action'])
        # training support: the primary grid inventory resolves to training candidate ids
        self.assertEqual(sets['training']['membership']['grid'], sets['evaluation']['membership']['grid'])


class ScreeningTests(unittest.TestCase):
    def roster(self, materialize, prior=()):
        templates = {family: {} for family in sc.FAMILIES}
        return sc.build_rosters(templates, list(prior), VOCABULARY, sizes=SMALL, materialize=materialize)

    def test_rejections_are_recorded_and_rosters_continue_with_the_next_seed(self):
        first_fit = sc.seeds('fit', sc.FAMILIES[0], 1)[0]

        def materialize(family, seed, template):
            offset = seed - first_fit
            if offset == 1:
                return fake_level('dup-of-first')
            if offset == 2:
                return fake_level('x', slots=('bird:0000', 'block:0005'))
            if offset == 3:
                return fake_level('dup-of-first')
            if offset == 4:
                return fake_level('y', bird='BirdBlue')
            if offset == 5:
                raise RuntimeError('generator failed')
            if offset == 6:
                return fake_level('prior-xml')
            return fake_level(f'level-{seed}')

        prior_xml = sc.xml_digest(fake_level('prior-xml')[0])
        prior = [{'path': 'p', 'generation_or_reserved_seeds': [sc.seeds('policy', sc.FAMILIES[1], 1)[1]],
                  'scenario_identities': [], 'scenario_content_sha256': [], 'xml_content_sha256': [prior_xml]}]
        splits, rejected, bound = self.roster(materialize, prior)
        reasons = [(row['split'], row['family'], row['attempt'], row['reason']) for row in rejected]
        family = sc.FAMILIES[0]
        self.assertEqual(reasons, [
            ('fit', family, 3, 'slot_outside_frozen_vocabulary'),
            ('fit', family, 4, 'duplicate_xml_content'),
            ('fit', family, 5, 'bird_not_birdred'),
            ('fit', family, 6, 'materialization_error'),
            ('fit', family, 7, 'duplicate_xml_content'),
            ('policy', sc.FAMILIES[1], 1, 'reserved_seed_collision'),
        ])
        self.assertEqual(rejected[1]['detail'], {'duplicates': f'issue-109-sc-fit-{family}-002'})
        self.assertEqual(rejected[4]['detail'], {'duplicates': 'prior exposure'})
        for split, size in SMALL.items():
            for fam in sc.FAMILIES:
                self.assertEqual(len(splits[split][fam]), size)
        self.assertEqual([level['ordinal'] for level in splits['fit'][family]], [1, 2, 3])
        self.assertEqual(splits['fit'][family][2]['generation_seed'], first_fit + 7)
        self.assertTrue(sc.split_disjointness(splits)['splits_level_disjoint'])
        self.assertEqual(len(bound), sum(SMALL.values()) * len(sc.FAMILIES))

    def test_level_shared_between_splits_breaks_disjointness(self):
        splits, _, _ = self.roster(lambda family, seed, template: fake_level(f'level-{seed}'))
        leaked = copy.deepcopy(splits)
        leaked['evaluation'][sc.FAMILIES[0]][0]['xml_sha256'] = leaked['fit'][sc.FAMILIES[0]][0]['xml_sha256']
        with self.assertRaises(ValueError):
            sc.split_disjointness(leaked)


@unittest.skipUnless(AVAILABLE, 'retained #71 / #77 / #96 artifacts not present')
class FrozenPlanTests(unittest.TestCase):
    def test_real_materialization_yields_level_disjoint_splits(self):
        prior, _ = sc.prior_exposure()
        splits, rejected, bound = sc.build_rosters(sc.n1.template_sources(), prior, sc.slot_vocabulary(),
                                                   sizes=SMALL)
        self.assertTrue(sc.split_disjointness(splits)['splits_level_disjoint'])
        self.assertTrue(sc.n1.audit_disjointness(bound, prior)['passed'])
        vocabulary = set(sc.slot_vocabulary())
        for rosters in splits.values():
            for roster in rosters.values():
                for level in roster:
                    self.assertTrue(set(level['generated_slots']) <= vocabulary)
        self.assertTrue(all(row['reason'] in sc.REJECTION_REASONS for row in rejected))

    def test_prepare_then_validate_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cost = directory / 'cost.json'
            cost.write_text(json.dumps(COST))
            output = directory / 'out'
            with mock.patch('builtins.print'):
                plan = sc.prepare(output, cost)
                self.assertEqual({family: len(roster) for family, roster in plan['splits']['evaluation'].items()},
                                 {family: 200 for family in sc.FAMILIES})
                with self.assertRaises(ValueError):
                    sc.prepare(output, cost)
                self.assertTrue(sc.validate(output))
                level = plan['splits']['evaluation'][sc.FAMILIES[1]][-1]
                path = output / level['xml_path']
                path.write_text(path.read_text().replace('BirdRed', 'BirdRed ', 1))
                self.assertFalse(sc.validate(output))

    def test_prepare_refuses_without_published_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(sc, 'SMOKE_SUMMARY', Path(directory) / 'missing.json'):
                with self.assertRaises(ValueError):
                    sc.prepare(Path(directory) / 'out')
            self.assertFalse((Path(directory) / 'out').exists())


if __name__ == '__main__':
    unittest.main()
