# Source and evidence policy

LABrador treats provenance as part of the calculation. A number without a source, evidence grade,
or explicit `SYNTHETIC` marker is an unsupported assumption—not an observed fact.

## Evidence states

Every material input should be labeled as one of:

- **Observed:** transcribed from a primary or authoritative source with URL/identifier, date, and
  enough context to reproduce the value.
- **Modeled assumption:** an analyst choice with rationale and range; never relabeled as observed.
- **Estimated:** derived from observed inputs with the calculation retained.
- **Synthetic:** fictitious demonstration data. Any synthetic critical input forces
  `NOT_DECISION_GRADE`.
- **Unsupported:** missing usable evidence or rationale. Unsupported critical inputs also force
  `NOT_DECISION_GRADE`.

Evidence grade and evidence type are separate. A real URL does not automatically make a source
high quality or applicable to the target indication, payer, year, or geography.
Only `HIGH` or `MODERATE` non-synthetic source types can clear a critical evidence gate. `LOW`
grades and `ASSUMPTION`, `SYNTHETIC`, or `UNSUPPORTED` types stay visible but cannot confer
decision grade on their own.

## Price bases

Keep these separate through ingestion, matching, display, and sensitivity analysis:

| Price basis | What it can support | What it cannot establish |
|---|---|---|
| `LIST` | A public reference/list amount | Realized reimbursement or manufacturer net revenue |
| `PUBLIC_REIMBURSEMENT` | A published payer/payment signal | Confidential rebates or actual manufacturer net |
| `ESTIMATED_NET` | A transparent analyst scenario | An observed contract price |
| `OBSERVED_NET` | Authorized evidence of a realized net amount | Transferability to a different payer/product |

Do not reverse-engineer or claim a confidential manufacturer net price from public comparables.
When an authorized observed-net input is unavailable, model an explicit estimated-net range and
retain that label in every downstream result.

Until an explicit price-indexing method is supplied, every selected anchor must already use the
program base year. LABrador will warn on other price years and block decision grade instead of
silently inflating or mixing nominal amounts.

## Approved practical data lanes

### Regulatory indication and patent context

- [Drugs@FDA](https://www.accessdata.fda.gov/scripts/cder/daf/) and
  [FDA labeling resources](https://labels.fda.gov/) can support approved indication, population,
  dosing, warnings, approval history, and regulatory context. They are not price sources.
- The [FDA Orange Book](https://www.accessdata.fda.gov/scripts/cder/ob/) can support listed patent
  and exclusivity context for approved small-molecule products. It is not a freedom-to-operate,
  validity, enforceability, or product-specific legal conclusion, and it is not a price source.
- Preserve the filing-date evidence and any extension assumption. A label expansion does not
  reset the asset's patent clock.

### Public price, reimbursement, utilization, and access signals

- [CMS Part B Average Sales Price files](https://www.cms.gov/medicare/payment/fee-for-service-providers/part-b-drugs/average-drug-sales-price)
  can support public Medicare Part B payment calculations and price-period context.
- [CMS Medicare Part D data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers)
  can support public utilization/spending analyses with the dataset's suppression and scope
  limitations retained.
- [Medicaid NADAC](https://www.medicaid.gov/medicaid/prescription-drugs/retail-price-survey/national-average-drug-acquisition-cost)
  can support a public pharmacy-acquisition-cost signal for covered outpatient drugs.

These sources do not disclose the complete confidential manufacturer rebate, contracting, or
realized net-revenue stack. Label each imported amount using its actual public basis.

### HTA and value context

- [NICE technology appraisal guidance](https://www.nice.org.uk/guidance/ta) can support public HTA
  reasoning, comparators, modeled outcomes, and disclosed prices. Preserve statements that a
  commercial arrangement or discount is confidential; do not infer its value.
- A cited appraisal remains geography-, indication-, comparator-, and date-specific.

### Population and patient affordability

- Use indication-specific epidemiology from primary literature, registries, or authoritative
  public-health sources, with definitions and years retained.
- Use [US Census American Community Survey](https://www.census.gov/programs-surveys/acs) or another
  named household survey for income distributions, or identify values as user assumptions.
- Use [World Bank country and lending groups](https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups)
  only as descriptive country context. Income group or GNI per capita is not a mechanical
  willingness-to-pay, QALY, clinical-value, or medicine-price multiplier.
- Patient income may inform out-of-pocket affordability, initiation, or access scenarios. It must
  not mechanically alter clinical benefit.

### Literature retrieval

- [Paperclip](https://paperclip.gxl.ai/) can help an agent search and retrieve bounded biomedical
  literature. Record the query, filters, result identifiers, and access date.
- Retrieval is discovery, not validation. Verify decision-critical claims against the primary
  paper, registry, regulator, payer, or HTA record before upgrading the evidence grade.

## Minimum source record

For every critical input, retain when available:

- source ID or stable URL;
- title/citation and publisher;
- publication/source date and access date;
- geography, payer, population, indication, and price year;
- value, units, currency, period, and price basis;
- extraction or derivation method;
- evidence type and grade;
- synthetic flag;
- applicability caveats and analyst notes.

Never place API keys, cookies, bearer tokens, signed URLs, or confidential contract text in fixture,
result, audit, or source fields.

## Decision-grade gate

`DECISION_GRADE` means only that required fields cleared the implemented evidence checks. It does
not turn a screening model into a reimbursement submission, investment recommendation, or legal
opinion. A result remains `NOT_DECISION_GRADE` when critical inputs are synthetic, unsupported,
materially stale, inapplicable to the target geography/indication, or missing essential context.

`DECISION_GRADE` is not a claim of model validation. Cited reality anchors, configuration checks,
falsification controls, internal arithmetic reconciliation, and replay each answer different
questions and must remain separately labeled. See
[interpretability-contract.md](interpretability-contract.md).
