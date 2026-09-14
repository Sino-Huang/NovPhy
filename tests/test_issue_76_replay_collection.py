import unittest
from unittest.mock import patch

from scripts import validate_issue_76_replay_collection as collection


def test_failed_candidate_keeps_entire_group_inadmissible():
    members = [{"identity": str(i), "source_member_identity": "source",
                "original_action_reference": i == 0} for i in range(3)]
    cases = {str(i): i for i in range(3)}
    calls = []

    def compare(reference, candidate, limits):
        calls.append((reference, candidate))
        return {"comparable": candidate != 1, "failures": ["difference"] if candidate == 1 else []}

    with patch.object(collection.smoke.comparison, "compare", compare):
        group, = collection.compare_groups(members, cases, {})
    assert calls == [(0, 0), (0, 1), (0, 2)]
    assert not group["paired_ranking_admissible"]
    assert len(group["comparisons"]) == 3
    assert group["comparisons"][1]["failures"] == ["difference"]


def test_each_source_uses_its_own_reference():
    members = [{"identity": str(i), "source_member_identity": str(i // 2),
                "original_action_reference": i % 2 == 0} for i in range(4)]
    calls = []

    def compare(reference, candidate, limits):
        calls.append((reference, candidate))
        return {"comparable": True, "failures": []}

    with patch.object(collection.smoke.comparison, "compare", compare):
        groups = collection.compare_groups(members, {str(i): i for i in range(4)}, {})
    assert calls == [(0, 0), (0, 1), (2, 2), (2, 3)]
    assert all(group["paired_ranking_admissible"] for group in groups)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_failed_candidate_keeps_entire_group_inadmissible,
        test_each_source_uses_its_own_reference))
