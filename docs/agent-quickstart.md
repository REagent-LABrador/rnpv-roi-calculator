# Agent quickstart

For multi-repository orchestration, use the versioned file-in/file-out module command instead of
importing internal calculation functions:

```bash
.venv/bin/rnpv-roi run --input examples/input.json --output ./work/roi-response.json
```

Validate the request and response against `schemas/input.schema.json` and
`schemas/output.schema.json`. A response with `status: "ok"` may intentionally contain
`payload.decision_grade: "NOT_DECISION_GRADE"`; transport completion and evidence sufficiency are
separate states.

The analyst CLI remains available for interactive validation, portfolio comparison, dashboard
inputs, and replay:

```bash
labrador example --output-dir ./work/inputs
labrador validate ./work/inputs/demo_program.json \
  --comparables ./work/inputs/demo_comparables.json --compact
labrador analyze ./work/inputs/demo_program.json \
  --comparables ./work/inputs/demo_comparables.json \
  --simulations 1000 --seed 42 --output ./work/result.json
labrador replay ./work/result.json
labrador portfolio ./work/inputs/demo_program.json \
  ./work/inputs/demo_program_b.json \
  --comparables ./work/inputs/demo_comparables.json \
  --simulations 1000 --seed 42 --sort-by p50_rnpv --descending
```

Agent rules:

1. Preserve every `price.basis`, evidence record, warning, and decision-grade field.
2. Do not change `synthetic`, evidence grade, or evidence type merely to silence a warning.
3. Never reinterpret list/public reimbursement data as an actual confidential net price.
4. Treat patient income as an access/affordability input, not a clinical-value multiplier.
5. Keep prevalent backlog, incident flow, adoption, access, persistence, overlap, and
   cannibalization distinct.
6. Do not reset the patent clock for an expansion indication.
7. Keep the result seed and simulation count with any quoted percentile.
8. Quote a KPI together with its currency, year/basis, warnings, and decision grade.
9. Supply explicit annual patient OOP evidence for decision-grade affordability; a net-price
   cost-share proxy must remain labeled and non-decision-grade.
10. Compare portfolio programs only when currency and valuation year match.
11. Treat `portfolio` ordering as a declared numeric screening sort, never an investment ranking.
12. Distinguish `MODEL_OUTPUT`, `CITED_REALITY_ANCHOR`, `CONFIGURATION_CHECK`, and
    `FALSIFICATION_CONTROL`. Do not combine their counts or call passing range checks validation.
13. Preserve `input_snapshot`, `input_digest`, engine/schema versions, simulation drivers, and
    internal reconciliation with every quoted result.
14. Treat a successful `replay` as deterministic consistency of engine-owned fields, not empirical
    validity. The interpretation presentation envelope is excluded from replay equality.
15. Report the shared filing/launch/expiry calendar and patient OOP basis explicitly.
16. Preserve the RNG contract: NumPy `default_rng`/PCG64, shared initial/expansion commercial
    shocks, sequential stage Bernoulli draws, implementation-dependent draw order, and the locked
    dependency environment needed for exact replay.
17. Use `indication.comparator_ids` as an explicit reviewed allowlist when the catalog contains
    products that must not anchor the indication's price.

Before using external evidence, follow [source-policy.md](source-policy.md). Before quoting an
analysis or anchor result, follow [interpretability-contract.md](interpretability-contract.md).
The bundled files are interface fixtures only and are always `SYNTHETIC` /
`NOT_DECISION_GRADE`.
