# rNPV/ROI module JSON contracts

Both schemas use JSON Schema Draft 2020-12 and are generated from the Pydantic transport/domain
models. Regenerate the schemas/input example, then the canonical response, with:

```bash
.venv/bin/python scripts/generate_contract_artifacts.py
.venv/bin/rnpv-roi run --input examples/input.json --output examples/output.json
```

CI fails if the generated input/schema artifacts drift from the checked-in files.

## Input

`input.schema.json` requires:

| Field | Meaning |
| --- | --- |
| `contract_version` | Transport contract; exactly `1.0.0`. |
| `module` | Routing identifier; exactly `rnpv_roi_calculator`. |
| `request_id` | Caller-provided correlation ID, returned unchanged. |
| `program` | Therapeutic program, initial indication, optional one-label expansion, patent, development, access, population, evidence, and assumptions. |
| `comparables` | Canonical comparable-therapy array. The outer metadata from legacy demo fixtures is not part of this contract. |
| `execution` | Explicit Monte Carlo draw count, seed, and optional triangular uncertainty ranges. |

Canonical modalities are `SMALL_MOLECULE`, `PEPTIDE`, and `ANTIBODY`. Prices retain explicit
currency, year, period, and basis (`LIST`, `PUBLIC_REIMBURSEMENT`, `ESTIMATED_NET`, or
`OBSERVED_NET`).

Some established economic inputs remain in the program/indication `assumptions` objects for engine
compatibility. The complete synthetic example demonstrates their units and placement. Important
keys include annual gross/net price assumptions, persistence, dose intensity, patient-level COGS,
commercial costs, epidemiology/adoption settings, QALY and willingness-to-pay inputs, payer budget,
and expansion interaction assumptions. JSON Schema cannot express every cross-field economic rule;
the runner always performs Pydantic and engine validation after transport validation.

## Output

`output.schema.json` accepts exactly one of:

- `status: "ok"` with a complete engine `AnalysisResult` in `payload`;
- `status: "error"` with `payload: null` and one or more structured `errors`.

Useful success paths for the orchestrator/UI:

| JSON path | Meaning |
| --- | --- |
| `payload.decision_grade` | `DECISION_GRADE` or `NOT_DECISION_GRADE`; separate from transport status. |
| `payload.recommendation` | Screening recommendation derived from the modeled uncertainty. |
| `payload.summary` | Headline deterministic and P10/P50/P90 rNPV metrics. |
| `payload.pricing` | Comparable selection, price corridor, and access estimates by indication. |
| `payload.access` | Treated-patient, affordability, payer-budget, and PMPM outputs. |
| `payload.cash_flow.annual_cash_flows` | Annual patient/revenue/cost/free-cash-flow ledger. |
| `payload.uncertainty` | Seeded rNPV, revenue, cash-at-risk, and protected-year distributions. |
| `payload.calculation_steps` | Formulas, inputs, outputs, units, and notes for interpretation. |
| `payload.critical_evidence_status` | Fail-closed evidence gates. |
| `interpretability` | Required shared UI contract for every `status: "ok"` response. |
| `warnings` | Envelope copy of engine warnings for generic pipeline display. |
| `provenance` | Envelope copy of field-level evidence references. |

`interpretability` schema version `1.0.0` contains required `headline`, `metrics`, `steps`,
`evidence`, `assumptions`, `uncertainty`, `limitations`, `counterfactuals`, `lineage`, and
`extensions` fields. Its adapter copies authoritative native outputs rather than recalculating them.
References are validated at runtime, numeric metrics carry units, Monte Carlo ranges are labeled as
scenario percentiles rather than confidence intervals, and unknown values remain `null` with a
limitation. The RNPV-specific extension retains the exact value decomposition and annual ledger,
plus currency/valuation-year/price-basis and RNG/correlation context.

The requirement applies only to successful/domain-degraded responses. Infrastructure and input
failures keep the existing `status: "error"` envelope and do not fabricate an interpretation.

Exact replay requires the recorded engine version and compatible locked dependencies. A seed and
input JSON alone are not a cross-version numerical reproducibility guarantee.

## Versioning

- `contract_version` changes when the module request/response envelope changes.
- `module_version` identifies the installed package.
- `engine_version` identifies calculation behavior.
- `engine_schema_version` and `payload.schema_version` identify the raw result schema.

Consumers should reject unsupported major contract versions and preserve unknown result fields when
forwarding an otherwise supported response.
