"""Issue-109 item 1: held-out cohort filter and the all-member #96 reproduction guard."""
import copy
import unittest

from scripts import run_heldout_rescore as h

AVAILABLE = (h.SOURCE / "compute.json").is_file() and h.DYNAMICS_PLAN.is_file()


@unittest.skipUnless(AVAILABLE, "retained #96 / #77 artifacts not present")
class HeldOutRescoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members, cls.universe, _ = h.load_universe()
        cls.fit = h.fit_lineages()
        cls.cohorts = h.cohort_members(cls.members, cls.fit)
        cls.compute96 = h.read_json(h.SOURCE / "compute.json")

    def test_held_out_rows_contain_no_fit_lineage(self):
        fitted = set(self.fit["predictor"]) | set(self.fit["controller"])
        pruned = h.prune(self.universe, self.cohorts["held_out"])
        for inventory in h.INVENTORIES:
            members = {row["member"] for row in h.t96.inventory_rows(h.SOURCE, pruned, inventory)}
            with self.subTest(inventory=inventory):
                self.assertTrue(members)
                self.assertTrue(members <= set(h.HELD_OUT))
                self.assertFalse(members & fitted)
        self.assertFalse(set(self.cohorts["exposed"]) & set(h.HELD_OUT))
        self.assertEqual(sorted(self.cohorts["held_out"] + self.cohorts["exposed"]), self.cohorts["all"])

    def test_held_out_member_that_is_a_fit_lineage_is_rejected(self):
        leaky = {"predictor": self.fit["predictor"] + [h.HELD_OUT[0]], "controller": self.fit["controller"]}
        with self.assertRaises(ValueError):
            h.cohort_members(self.members, leaky)

    def test_all_member_cohort_reproduces_published_96_and_detects_drift(self):
        choices = h.published_choices(self.compute96)
        rows = {i: h.t96.inventory_rows(h.SOURCE, self.universe, i) for i in h.INVENTORIES}
        blocks = {i: h.inventory_block(rows[i], self.universe[i], choices[i]) for i in h.INVENTORIES}
        tokens = h.cohort_tokens(blocks[h.PRIMARY], self.compute96["guards"])
        self.assertTrue(h.reproduction_check(blocks, rows, tokens, self.compute96)["pass"])

        drifted = copy.deepcopy(blocks)
        drifted[h.PRIMARY]["contrasts"]["J-F*"]["estimate"]["mean"] += 1e-3
        with self.assertRaises(ValueError):
            h.reproduction_check(drifted, rows, tokens, self.compute96)


if __name__ == "__main__":
    unittest.main()
