# Therapeutic Program Economics engineering contract

Therapeutic Program Economics is a screening-grade simulator. It is decision support,
not medical, reimbursement, investment, legal, or patent advice.

## Non-negotiable model rules

- Keep list, public reimbursement, estimated net, and observed net prices distinct.
- Patient income may constrain access and out-of-pocket affordability; it must not mechanically
  reduce the clinical value of a treatment.
- A 20-year patent term starts at filing. Label expansion does not reset that clock.
- Every comparable and material assumption needs a source, an evidence grade, or an explicit
  `synthetic` marker.
- Unsupported critical inputs must produce `NOT_DECISION_GRADE`, not invented precision.
- Preserve separate payer, patient, and manufacturer outputs.
- Never imply that comparable prices prove an actual confidential manufacturer net price.

## Delivery rules

- Keep the calculation engine deterministic for a fixed random seed.
- Expose calculation steps, warnings, assumptions, matched comparables, and source provenance.
- Add a regression test for every change to economic logic.
- Run Ruff and pytest before committing.
