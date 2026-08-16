# rNPV / ROI Calculator

This repository is the standalone rNPV/ROI module for the REagent-LABrador pipeline. It is an
interpretable, screening-grade simulator that connects a provenance-aware therapeutic program and
comparable-price evidence to access, affordability, protected cash flow, risk-adjusted NPV, and
seeded uncertainty.

> **Decision boundary:** LABrador is decision support, not medical, reimbursement, investment,
> legal, or patent advice. The bundled demo is **SYNTHETIC** and
> **NOT_DECISION_GRADE**. Public prices or comparable products do not reveal an actual
> confidential manufacturer net price.

Interpretability is not validation. LABrador labels four different result types and never treats
them as one score:

- `MODEL_OUTPUT` — a calculation from submitted inputs and assumptions;
- `CITED_REALITY_ANCHOR` — a sourced plausibility-band comparison, not validation or calibration;
- `CONFIGURATION_CHECK` — confirmation that a declared convention is configured as intended;
- `FALSIFICATION_CONTROL` — a perturbed case that should fail, proving only that the harness can.

See [the interpretability contract](docs/interpretability-contract.md) before quoting a result.

## Pipeline contract v1

The portable module boundary is one JSON request and one JSON response:

```bash
uv sync --locked --extra dev --no-editable

.venv/bin/rnpv-roi run \
  --input examples/input.json \
  --output /tmp/rnpv-roi-output.json
```

The command exits `0` after a completed calculation, including a valid
`NOT_DECISION_GRADE` result. Invalid input or execution failure exits `2` and still writes a
schema-valid `status: "error"` response. Standard output is empty; concise operational messages go
to standard error only on failure.

Published contracts:

- [`schemas/input.schema.json`](schemas/input.schema.json) — combined program, comparable catalog,
  and deterministic execution settings;
- [`schemas/output.schema.json`](schemas/output.schema.json) — discriminated success/error response
  envelope;
- [`examples/input.json`](examples/input.json) — complete synthetic request;
- [`examples/output.json`](examples/output.json) — complete synthetic response.

The transport `contract_version` is `1.0.0`. It is independent of module/engine version `0.4.0`
and engine result schema `1.3.0`.

### Input envelope

```json
{
  "contract_version": "1.0.0",
  "module": "rnpv_roi_calculator",
  "request_id": "program-001-roi",
  "program": {},
  "comparables": [],
  "execution": {
    "simulations": 1000,
    "seed": 42,
    "simulation_assumptions": {}
  }
}
```

The request is strict: unknown envelope fields, duplicate JSON keys, `NaN`, `Infinity`, more than
one label expansion, non-20-year modeled patent terms, invalid comparator catalogs, and invalid
simulation settings fail closed.

### Output envelope

Successful execution uses `status: "ok"`, returns the raw, replayable `AnalysisResult` under
`payload`, and always includes the required shared UI contract at top-level `interpretability`.
Engine warnings and evidence references are repeated at the envelope level as `warnings` and
`provenance` for a generic orchestrator.

The shared `interpretability` object is schema version `1.0.0`. It maps, without recalculation,
the native summary, calculation steps, evidence references, input assumptions, uncertainty,
warnings, failed evidence gates, launch-delay counterfactual, and lineage into stable common fields.
Module-specific value decomposition, annual ledger, financial context, RNG/correlation details, and
native evidence grades remain available under `interpretability.extensions`. Unknown values stay
JSON `null` and carry a structured limitation; transport completion never upgrades a
`NOT_DECISION_GRADE` result.

Failure uses `status: "error"`, `payload: null`, and structured `errors` with `type`, `path`, and
`message`. `NOT_DECISION_GRADE` is not a transport failure; downstream consumers must inspect both
`status` and `payload.decision_grade`.

See [`schemas/README.md`](schemas/README.md) for field semantics, versioning, and downstream paths.

## What it does

- Keeps list, public reimbursement, estimated net, and observed net price bases distinct.
- Separates clinical value from health-system access and patient out-of-pocket affordability.
- Models an eligible/prevalent population and incident flow without calling the result a market
  forecast.
- Applies the original asset patent clock to both an initial indication and any label expansion;
  an expansion does not restart the 20-year term.
- Produces annual patient, revenue, cost, free-cash-flow, and protected/post-LOE views.
- Runs deterministic Monte Carlo scenarios for a fixed seed.
- Returns assumptions, calculation steps, provenance, evidence grades, and warnings in JSON.
- Marks unsupported critical inputs or any synthetic demonstration as
  `NOT_DECISION_GRADE`.

## Analyst CLI and dashboard

Requires Python 3.11 or 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'

labrador example
labrador validate fixtures/demo_program.json \
  --comparables fixtures/demo_comparables.json
labrador analyze fixtures/demo_program.json \
  --comparables fixtures/demo_comparables.json \
  --simulations 1000 --seed 42 \
  --output analysis.json
labrador replay analysis.json
```

The CLI prints JSON by default. `--compact` produces one-line JSON for agent pipelines. A CSV
version of the comparable fixture is included to exercise the upload/import path:

```bash
labrador compare fixtures/demo_comparables.csv --compact
labrador example --output-dir ./starter-inputs
labrador portfolio fixtures/demo_program.json fixtures/demo_program_b.json \
  --comparables fixtures/demo_comparables.json \
  --simulations 1000 --seed 42 --sort-by p50_rnpv --descending
```

Add `--reality-checks` to `labrador analyze` to run the separate cited RA/I&I plausibility
harness. A passing band does not upgrade the submitted program's decision grade and must not be
reported as model validation.

Launch the dashboard:

```bash
streamlit run app.py
```

The dashboard provides five coordinated views:

1. **Executive** — screening recommendation, primary metrics, and material warnings.
2. **Price & Comparables** — explicit price bases, matched evidence, and provenance.
3. **Access & Affordability** — eligible and treated patients, payer budget impact, and PMPM.
4. **Cash Flow** — protected versus post-LOE revenue, costs, and discounted cash flow.
5. **Sensitivity / Audit** — declared simulation drivers, internal reconciliation, separately
   bucketed reality anchors, full input snapshot, and replayable JSON record.

## Engine input model

The Pydantic contracts in `src/labrador_roi/models.py` are authoritative. A `ProgramInput`
contains:

- asset identity, modality (`SMALL_MOLECULE`, `PEPTIDE`, or `ANTIBODY`), route,
  valuation/base years, and currency;
- an initial indication and optional expansion indications;
- population stock/flow, health-system access gates, and separately labeled income bands;
- patent filing term and any explicitly assumed extension;
- stage costs, durations, success probabilities, evidence, and analyst assumptions.

A comparable keeps product context and `PriceObservation` together. Every observation specifies
amount, currency, period, price year, price basis, and evidence metadata. Course or unit prices
also require annualization units. CSV upload is a transport convenience; it is normalized into
the same nested validated contract before analysis. When an indication supplies `comparator_ids`,
that list is an explicit analyst allowlist: unlisted catalog records cannot anchor its price.

Start from the bundled files:

- `fixtures/demo_program.json`
- `fixtures/demo_program_b.json` (a second synthetic program for portfolio comparison)
- `fixtures/demo_comparables.json`
- `fixtures/demo_comparables.csv`

Every value in these files is fictitious. Replace the values and the `SYNTHETIC` evidence records;
changing only the label is not sufficient to make an analysis decision-grade.

## Analyst CLI output

Successful CLI commands emit JSON. Validation failures also return structured JSON and exit with
code `2`:

```json
{
  "status": "error",
  "operation": "validate",
  "error_type": "ValidationError",
  "errors": [
    {"type": "missing", "loc": ["initial_indication"], "msg": "Field required"}
  ]
}
```

Each analyst-CLI analysis JSON includes its existing presentation-oriented `interpretability`
manifest with the status taxonomy, decision-grade warning, input digest/version context, price
currency/basis/year context, patient OOP basis, shared patent clock, declared simulation design,
internal output reconciliation, and optional reality-anchor report. This is distinct from the
portable module response's required shared contract. Do not scrape a displayed KPI and discard
either surface.

`labrador replay analysis.json` reconstructs the recorded inputs and verifies engine-owned fields.
The CLI/dashboard interpretation envelope is excluded from replay equality. A successful replay
establishes deterministic engine-artifact consistency for the recorded version and locked compatible
dependency environment; it is not empirical validation.

`labrador portfolio` accepts two or more program JSON paths plus one shared comparable catalog.
It returns standardized P10/P50/P90 rNPV, probability-positive, cash-at-risk, protected-years,
launch-delay-cost, decision-grade, and recommendation rows. Its explicit numeric sort is a
screening convenience, not an investment ranking. Programs must share one currency and valuation
year; LABrador will not silently perform FX or time-basis conversions. `NOT_DECISION_GRADE`
programs still require evidence replacement and review.

## Core assumptions and interpretation rules

- A comparable is evidence, not proof that two therapies deserve the same price.
- Public reimbursement, acquisition cost, wholesale/list price, and estimated net price answer
  different questions and are never silently pooled.
- Net-price scenarios are analyst inputs unless backed by authorized observed-net evidence. The
  repository contains no actual confidential manufacturer net-price data.
- Patient income can constrain coverage, initiation, cost sharing, and access. It cannot
  mechanically reduce clinical benefit or a QALY gain.
- For nonzero cost sharing, a decision-grade affordability result requires an explicit annual
  patient out-of-pocket amount and evidence. A manufacturer-net percentage is retained only as
  a labeled screening proxy and cannot clear the evidence gate.
- When income-band coverage is incomplete, any `patient_affordability_rate` is an explicit
  analyst fallback scenario, not evidence about patient wealth; the synthetic expansion fixture
  demonstrates this path and remains `NOT_DECISION_GRADE`.
- Cost-effectiveness, payer budget impact, patient affordability, and manufacturer cash flow are
  separate outputs. A favorable result in one does not establish another.
- Population estimates separate prevalent launch backlog from incident flow and apply explicit
  coverage, authorization, initiation, provider-capacity, adoption, persistence, overlap, and
  cannibalization assumptions.
- The simplified patent calculation is a screening model: base term starts at filing, not launch,
  and does not substitute for a product-specific legal/FDA exclusivity review.
- Results should be reported as ranges and scenarios. Synthetic demo precision is interface
  precision, not evidentiary precision.
- P10/P50/P90 are percentiles from the declared scenario model, not confidence intervals, observed
  frequencies, or independently validated forecasts. Always quote the seed, draw count, and sampled
  driver registry.
- Simulation uses NumPy `default_rng`/PCG64. Commercial shocks are shared across the initial and
  expansion indication within a draw; stage Bernoulli events are sequential. Draw order is an
  implementation detail, so seed plus JSON is not a cross-version or cross-dependency guarantee.
- Internal output reconciliation checks arithmetic consistency between headline fields and the
  underlying ledger. It does not establish that inputs or model structure are correct.
- Route is an explicit stratification field, not a hidden adherence multiplier. Encode
  route-specific administration burden, persistence, capacity, and costs as sourced assumptions.

## Source policy

Use bounded, auditable evidence lanes:

- FDA labels and Drugs@FDA for approved indication and regulatory context—not price.
- Orange Book records for listed patent/exclusivity context—not a legal conclusion or price.
- CMS Part B/Part D public data and NADAC for public reimbursement, utilization, formulary, or
  acquisition-cost signals—not confidential manufacturer net price.
- NICE appraisals for public HTA reasoning and disclosed prices, while preserving any
  confidential-discount caveat.
- Census/household-survey income bands or explicit user inputs for patient affordability.
- World Bank income groups only as country context, never as a mechanical WTP or price multiplier.
- Paperclip for bounded literature retrieval, followed by primary-source verification.

The complete rules and authoritative links are in [docs/source-policy.md](docs/source-policy.md).
Evaluation and reporting semantics are in
[docs/interpretability-contract.md](docs/interpretability-contract.md).
The Claude-review remediation map is in
[docs/red-team-hardening.md](docs/red-team-hardening.md).

## Build and verification plan

The implementation is intentionally layered:

1. **Typed contracts and provenance** — reject invalid or unlabeled precision at ingestion.
2. **Comparable and pricing analysis** — rank relevant evidence without collapsing price bases.
3. **Access, cash flow, and uncertainty** — deterministic engine plus seeded simulations.
4. **Human and agent surfaces** — the same validated inputs and engine behind CLI and Streamlit.
5. **Regression gates** — economic invariants, CLI contracts, lint, and deterministic smoke tests.

Run the local verification suite:

```bash
pytest
ruff check .
labrador analyze fixtures/demo_program.json \
  --comparables fixtures/demo_comparables.json --simulations 100 --seed 7 --compact
```

## Repository map

```text
app.py                         Streamlit dashboard
fixtures/                      Explicitly synthetic demo inputs
src/labrador_roi/cli.py        JSON/CSV adapters and CLI commands
src/labrador_roi/interpretability.py  Shared portable interpretability adapter
src/labrador_roi/models.py     Validated domain contracts
src/labrador_roi/engine.py     Analysis orchestrator
docs/source-policy.md          Evidence and price-basis rules
docs/interpretability-contract.md  Output, evaluation, and replay semantics
docs/red-team-hardening.md     Review finding to regression/limitation map
tests/                         Economic, provenance, and CLI regression tests
```

## Residual limitations

LABrador does not itself establish clinical efficacy, perform legal patent analysis, negotiate
coverage, retrieve confidential rebate contracts, predict competitor behavior, or replace a
jurisdiction-specific HEOR model. Missing, synthetic, low-grade, or internally asserted inputs
must remain visible in the output and may keep the result `NOT_DECISION_GRADE`.

- COGS is an evidenced program input. LABrador does not infer manufacturing cost from SMILES,
  peptide sequence, synthetic-accessibility scores, or an SPPS yield model.
- Development probabilities are program inputs; no hidden modality multiplier or therapeutic-area
  prior is applied. Antibody programs must supply the same explicit development, COGS, route, and
  commercial assumptions as every other modality.
- Antibody programs must also supply explicit LOE price/volume retention paths with supported
  evidence to qualify as `DECISION_GRADE`; LABrador does not infer a biologic erosion curve, BLA
  pathway, exclusivity period, or IRA clock from modality alone.
- Comparable records do not yet carry modality. Antibody analyses must use deliberate
  `comparator_ids` selection and inspect the exposed clinical-placement and route matches.
- The cash-flow MVP values one initial indication and one expansion. Additional expansions are
  rejected from decision-grade use instead of being silently ignored.
- IRA/MFP timing, tax-loss carryforwards, a configurable cross-driver correlation matrix, and
  stage-level Monte Carlo draw logs are not implemented.

The RA/I&I reality anchors are sourced plausibility checks for defined US scenarios. They are not a
blinded back-test, calibration set, prospective validation, or evidence that another indication or
geography is modeled correctly. Numeric bands must be re-grounded before transfer.

No repository license has been selected yet. The repository owner should choose one before
redistribution or external reuse.
