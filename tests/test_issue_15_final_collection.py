from __future__ import annotations

import unittest

from scripts.final_evaluation_access import (
    audit_final_evaluation_workflow_access,
    authorize_final_evaluation_workflow_access,
)
from scripts.issue_15_final_collection import (
    DEFAULT_SEALED_ROOT,
    DEFAULT_ROOT,
    Issue15ConfirmatoryV2Reader,
    _assignment,
    _contract,
    _frozen,
    _observed_access,
)
from world_model.data import CohortV2OracleWindowDataset
from world_model.model import Abstraction, PredictionPair
from world_model.training.cohort_v2 import build_cohort_v2_transition_request


class Issue15FinalCollectionTests(unittest.TestCase):
    def test_frozen_collection_is_final_only_and_protocol_bound(self):
        _plan, protocol, collection, _partition, pending = _frozen(DEFAULT_ROOT)

        self.assertEqual(len(collection["attempt_ids"]), 6)
        self.assertEqual(_assignment(collection)["exposure_role"], "final_evaluation")
        self.assertEqual(pending.authorization_state, "pending")
        self.assertEqual(
            _contract(protocol).parameter_identity,
            protocol["artifact_identity"],
        )

    def test_authorized_access_record_passes_the_frozen_partition_audit(self):
        _plan, _protocol, collection, partition, pending = _frozen(DEFAULT_ROOT)
        authorized = authorize_final_evaluation_workflow_access(
            pending,
            authorization_identity="authorization:test",
            authorized_at="2026-08-27T07:00:00Z",
        )
        observed = _observed_access(
            authorized, _assignment(collection), "2026-08-27T07:00:00Z"
        )

        audit = audit_final_evaluation_workflow_access(
            partition, authorized, observed_accesses=[observed]
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["observed_access_count"], 1)

    def test_replacement_final_window_reaches_the_transition_adapter(self):
        reader = Issue15ConfirmatoryV2Reader(DEFAULT_SEALED_ROOT)
        window = CohortV2OracleWindowDataset(
            reader, requested_horizons=(1,)
        )[0]

        request = build_cohort_v2_transition_request(
            PredictionPair(1, Abstraction.CONTINUOUS), (window,)
        )

        self.assertEqual(request.pair, PredictionPair(1, Abstraction.CONTINUOUS))


if __name__ == "__main__":
    unittest.main()
