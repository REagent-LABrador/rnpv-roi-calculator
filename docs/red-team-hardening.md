# Red-team hardening record

This record maps the rNPV Copilot review and portable RA/I&I evaluation export (`fca049d`) to the
LABrador implementation. `CLOSED` means a regression now protects the stated behavior. It does not
mean the economic assumption is empirically validated.

| Review concern | LABrador disposition | Status |
|---|---|---|
| Monte Carlo scales one point revenue PV instead of rerunning the model | Every draw perturbs named inputs and recomputes patient cohorts, persistence, annual revenue, costs, tax, LOE, and discounted cash flow. | `CLOSED` |
| Binary success with the same expected development cost in every path | Stage Bernoulli paths pay a cost only when the stage is reached; a regression checks the simulated mean against expected-path NPV and preserves distinct failure losses. | `CLOSED` |
| Persistence uncertainty misses nonlinear cohort behavior | Persistence is sampled before the annual ledger is rebuilt; a regression proves the resulting revenue distribution is nonlinear. | `CLOSED` |
| Lifetime QALY value is reused as annual price | The value ceiling divides total incremental value and cost offsets by expected treatment years; a focused regression prevents lifetime-as-annual reuse. | `CLOSED` |
| Peak sales ignores the actual erosion/LOE ledger | Peak annual manufacturer net revenue and year are taken from the same annual ledger as NPV, with a simulated P50. | `CLOSED` |
| Expansion revenue is not conditioned on its development gateway | Expansion success, costs, overlap, spillover, and cannibalization are evaluated jointly; impossible conditional realizations fail validation. | `ALREADY_SAFE + REGRESSION` |
| Expansion receives a fresh patent term | Initial and expansion indications share one filing-based asset clock; launch delay shortens remaining protected life. | `ALREADY_SAFE + REGRESSION` |
| More labels are silently ignored | The MVP still values one expansion, but a second expansion now emits an error and forces `NOT_DECISION_GRADE`. | `FAIL-CLOSED LIMIT` |
| Overlap/cannibalization silently defaults to a decision-ready zero | The expansion interaction inputs and evidence are a critical gate; missing values cannot clear decision grade. | `CLOSED` |
| Arbitrary catalog products anchor price | `indication.comparator_ids`, when supplied, is an explicit allowlist; all unlisted records are excluded. | `CLOSED` |
| Seeded output cannot be reconstructed | Artifacts record the full input snapshot, seed, draw count, uncertainty ranges, engine/schema versions, NumPy version, bit generator, draw-contract version, and correlation semantics. `labrador replay` verifies engine-owned fields. | `CLOSED WITH VERSION BOUNDARY` |
| Passing checks are presented as validation | UI/CLI separate model output, cited range check, configuration check, falsification control, internal reconciliation, and replay. | `CLOSED` |
| Back-test is circular | The imported anchor harness is called a plausibility/arithmetic regression harness, never a back-test. Unsupported capabilities remain `SKIP`; source-derived inputs and citation-review limits are disclosed. | `CLOSED AS CLAIM BOUNDARY` |

## Portable RA/I&I checks

Run:

```bash
python -m labrador_roi.evaluation
```

The 14-row catalog keeps computation, cross-source, falsification-control, and configuration buckets
separate. The current adapter evaluates six model/control rows and one configuration row; unsupported
molecule-derived COGS, modality PoS, IRA timing, standard Phase-3 cost, and incompletely specified
headline eNPV rows remain visible as `SKIP`.

These checks are US RA/I&I-specific and reuse source-derived inputs for several arithmetic paths.
They are not blinded, prospective, out-of-sample, or sufficient to establish decision grade.

## Remaining modeling limits

- One expansion only; a portfolio of later labels needs a general multi-indication state model.
- No molecule/sequence-derived manufacturing-cost model.
- No IRA/MFP cash-flow module or jurisdiction-specific reimbursement model.
- Commercial shocks are shared by named driver across indications; there is no configurable
  correlation matrix.
- Exact per-draw stage traces are not stored; replay is bounded to the recorded engine and locked
  compatible dependency environment.
- Tax uses a simplified annual charge without NOL/tax-loss carryforward mechanics.
- Evidence gates and internal reconciliation improve inspectability; neither validates causal,
  clinical, epidemiologic, access, pricing, or commercial assumptions.
