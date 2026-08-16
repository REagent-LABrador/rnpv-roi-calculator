# RA / I&I reality-anchor adapter

This package ports the 14 implementation-independent anchors from the rNPV Copilot export at
commit `fca049d` without changing its claims or expected bands. Every row retains its original
source attribution.

Run it with:

```bash
python -m labrador_roi.evaluation
```

Or use the typed API:

```python
from labrador_roi.evaluation import evaluate_reality_anchors

report = evaluate_reality_anchors()
```

The adapter uses only public LABrador pricing and cash-flow contracts. An unsupported capability is
reported as `SKIP` with a reason; it never receives the expected value. Unexpected adapter errors are
`FAIL`. Configuration rows are counted separately and cannot inflate the model-reality score.

The deterministic `RAEvaluationScenario` makes all adapter inputs visible. It injects cited RA/BIO
scenario inputs into the engine; this validates engine arithmetic, price-basis handling, and cohort
behavior, not an internal RA evidence database. The pricing scenario deliberately keeps auxiliary
ceiling/floor inputs synthetic, so it does not become decision-grade merely by passing a price band.

The evaluator preserves citation metadata but does not retrieve, version, or independently validate
the underlying publications at run time. Several checks reuse source-derived inputs from the same
claim they compare against, so they are transparent arithmetic regression checks—not blinded or
out-of-sample evidence. Audit the source, population, year, and metric before using a band in a demo.

Numeric bands are US RA/I&I-specific. Re-ground them before using this structure for another
indication, geography, currency, or valuation year.
