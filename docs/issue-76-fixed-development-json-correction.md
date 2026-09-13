# Technical correction: JSON policy-map ordering

The complete calibration matrix was scored under execution plan v1. The first
independent validation attempt then stopped before checking numerical curves:
the shared JSON writer sorts object keys, whereas the validator and calibration
selector expected the in-memory policy map's original insertion order. All nine
hybrid policy keys were present; their lexical order differed from the frozen
native pair order. The failed validation attempt consumed 4.689 seconds of its
existing 900-second cumulative allowance. It is not a predictor failure.

The correction restores the original `model.pairs` ordering when loading result
objects, after requiring the exact policy-key set. It neither rewrites result
files nor recomputes predictions. The selector still iterates the original
recipe and pair inventories to resolve ties. Models, members, metrics, numerical
tolerances, failure penalties, selection rules and access ordering are unchanged.

A sorted-JSON round-trip regression failed before the fix and passes afterward;
it checks that all serialized values remain equal and the original selection is
reproduced. A missing-policy regression still rejects incomplete input. All 28
tests pass. The diagnosing-bugs skill guided this reproducible regression fix;
the direct key-order comparison made further speculative instrumentation
unnecessary.

The original execution and independent-validation plans remain immutable.
The explicit [correction receipt](../data/issue-76-fixed-development/json-order-correction.json)
binds the old and corrected driver source. Validation resumes against the same
saved calibration predictions and cumulative budget. No model-selection or
fresh prediction is opened by this correction.
