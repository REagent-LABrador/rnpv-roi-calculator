# Interpretability and evaluation contract

LABrador is a screening model whose outputs must be inspectable without being mistaken for facts.
Interpretability means that an analyst can recover the exact inputs, conventions, calculations,
uncertainty design, evidence gaps, and internal reconciliations behind an output. It does **not**
mean that a transparent output is accurate, calibrated, or appropriate for a decision.

## Four result types that must not be conflated

| Result type | What it means | What it does not mean |
|---|---|---|
| `MODEL_OUTPUT` | LABrador calculated a quantity from a submitted input snapshot and declared assumptions. | The quantity was observed, independently validated, or will occur. |
| `CITED_REALITY_ANCHOR` | An implementation output was compared with a sourced RA/I&I plausibility band. | The model was calibrated, back-tested, or validated for the submitted program. |
| `CONFIGURATION_CHECK` | A declared convention or rule is configured as intended. | The convention is empirically correct or predictively accurate. |
| `FALSIFICATION_CONTROL` | A deliberately perturbed case breached the expected band. | Other passing cases are correct. |

The reality-anchor harness reports these buckets separately. Never combine them into one pass rate.
In particular, configuration checks and falsification controls cannot inflate an empirical score.

## Reading one analysis artifact

Read fields in this order:

1. `decision_grade`, `recommendation`, `warnings`, and `critical_evidence_status` establish whether
   the implemented evidence gates passed. `DECISION_GRADE` still does not mean independently
   validated, decision-ready, or suitable for an investment or reimbursement submission.
2. `input_snapshot`, `input_digest`, `engine_version`, and `schema_version` identify exactly what
   was run. Do not quote a KPI without preserving this record.
3. `pricing` identifies the annual price amount, `basis`, `currency`, anchor price years, and
   valuation year. `ESTIMATED_NET` is a scenario; public list or reimbursement observations do not
   reveal confidential manufacturer net price.
4. `access_estimates[].patient_oop_basis` identifies whether patient liability is an explicit input
   or a manufacturer-net cost-share proxy. Only `EXPLICIT_ANNUAL_OOP` can support the implemented
   decision-grade affordability gate.
5. `cash_flow.patent_expiry_year`, launch years, and the patent input show one asset-level calendar.
   Expansion labels share the original filing clock and never receive a new 20-year term.
6. `seed`, `simulations`, and `simulation_assumptions` define the scenario distribution. P10/P50/P90
   are model percentiles, not confidence intervals or observed frequencies. The RNG contract uses
   NumPy `default_rng`/PCG64. Commercial shocks are shared across initial and expansion indications
   within a draw; clinical-stage Bernoulli draws are sequential. Draw order is implementation-
   dependent.
7. `calculation_steps`, `value_decomposition`, and `interpretability.output_reconciliation` expose
   formulas and check headline fields against their underlying output tables. Reconciliation is an
   internal arithmetic check, not external validation.
8. `interpretability.reality_anchors`, when run, reports cited plausibility comparisons independently
   of the submitted program result.

## Price and affordability labels

Every quoted price needs four labels: amount, currency, price basis, and year. State whether the year
is the source price year or the model valuation year. LABrador does not silently treat list,
reimbursement, estimated-net, and observed-net amounts as interchangeable.

Patient OOP is also a basis-sensitive quantity. A fallback calculated as manufacturer net price
times cost share is labeled `MANUFACTURER_NET_PROXY`; it is not an observed benefit-design liability
and must keep the result screening-only. Patient income affects modeled access and affordability,
not QALYs or clinical value.

## Reality-anchor interpretation

The bundled RA/I&I anchor collection is US- and source-period-specific. Its numeric bands do not
automatically transfer to another indication, payer, geography, population definition, route, or
year. A `PASS` means only that the measured output landed within that anchor's cited band. A `FAIL`
is diagnostic evidence requiring investigation; it does not identify the cause by itself. A `SKIP`
must stay visible and cannot be counted as a pass.

The harness preserves citation metadata but does not verify publications at run time. Checks that
inject source-derived values from the same claim are arithmetic regressions, not independent or
out-of-sample validation. Source scope, vintage, metric definition, and population still require
human review.

Ground rules carried forward from the red-team review:

- Net price is not WAC or list price.
- Franchise revenue is not single-indication revenue.
- Two values selected after observing the outcome are not an independent back-test.
- Modality probability comparisons must use the cited therapeutic-modality table and stage.
- IRA timing follows the regulatory approval pathway, not an informal molecule label.
- Revenue anchors retain the stated fiscal year.

## Replay

Persist an analysis to disk, then replay it:

```bash
labrador analyze fixtures/demo_program.json \
  --comparables fixtures/demo_comparables.json \
  --simulations 1000 --seed 42 --output analysis.json
labrador replay analysis.json
```

Replay reconstructs the validated inputs from `input_snapshot`, reruns the recorded seed and draw
count, and rejects incompatible engine versions, input digests, or engine-output mismatches. The
CLI/dashboard `interpretability` presentation envelope is excluded from replay equality. Exact
replay requires the recorded engine and a locked compatible dependency environment; seed plus JSON
alone is not a cross-version guarantee. A successful replay establishes deterministic consistency
of engine-owned fields. It does not establish empirical validity.

## Reporting template

When quoting a LABrador result, include:

```text
MODEL OUTPUT: [metric and value]
Program / indication: [...]
Currency / valuation year / price basis: [...]
Seed / draws / sampled drivers: [...]
Decision grade and material warnings: [...]
Input digest / engine version: [...]
Patient OOP basis: [...]
Patent filing, launch, and shared expiry: [...]
Internal reconciliation status: [...]
Reality-anchor status: NOT_RUN, or bucketed PASS/FAIL/SKIP counts (range checks, not validation)
```

Never collapse those fields into a single unexplained "validated ROI" score.
