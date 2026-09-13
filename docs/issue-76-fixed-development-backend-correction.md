# Technical correction: distinguish score verification from backend invariance

The original strict CPU reconstruction failed on a small floating-point
CPU/CUDA trajectory difference. The focused independent CUDA reconstruction
and separate NumPy field formulas matched the saved CUDA metrics under the
original tolerances. The original failure and all spent work remain recorded;
we do not claim CPU/CUDA backend invariance.

This amendment changes the arithmetic backend of the independent transition
check, not any saved prediction, target, metric, tolerance, policy-selection
rule or advancement criterion. It verifies the computation actually executed.
The unchanged independent host loop owns recursive state and native timing;
each transition executes on the original CUDA backend and returns to the host.
The transport receives only the current carrier, action and fixed pair, never
evaluation targets or stored predictions. NumPy independently computes all
field metrics on CPU. Neither the production recursive-loop helper nor its
metric helper is used in verification.

Every assigned member, model, fixed pair, available local context, common
elapsed time, field metric, failure, source and work record must still pass.
The original 0.001 relative and 0.00001 absolute metric tolerances stay fixed.
The complete calibrated choice is not frozen until all 54 cells pass this
explicitly qualified method. The original strict cross-backend check stays
false and is not replaced by a claim that it passed.

Keep the existing cumulative 900-second validation wall allowance, including
failed attempts. Also require each role's scoring charge plus **all** validation
wall time and the focused diagnostic charge to fit its 900-second allowance;
this conservatively includes CPU work and validation of the other role.
Memory, storage and access-order limits remain unchanged. No fitting, collection,
fresh access or inference to adaptive gameplay is authorized.

Freeze the new verifier, regression tests and this amendment before whole-matrix
verification. Preserve all original execution, smoke, validation and correction
plans. The original model-selection predictions remain unopened at amendment
time; the new verification method applies to the complete model-selection
matrix as well, without outcome-conditioned exclusions.
