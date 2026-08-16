"""Portable, honestly bucketed RA/I&I reality checks for LABrador.

The source anchors are implementation-independent.  This adapter deliberately distinguishes a
live LABrador calculation from a capability the engine does not currently implement.  Unsupported
anchors are explicit ``SKIP`` results with a reason; they are never filled with the expected value.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Any

from labrador_roi.cashflow import (
    DevelopmentStage,
    IndicationCommercialAssumptions,
    ProgramCashFlowInputs,
    calculate_cashflow,
)
from labrador_roi.cashflow import (
    PatentAssumptions as CashFlowPatentAssumptions,
)
from labrador_roi.comparables import ComparableSet
from labrador_roi.models import (
    AccessAssumptions,
    ComparableTherapy,
    DevelopmentAssumptions,
    EvidenceGrade,
    EvidenceMetadata,
    EvidenceType,
    IndicationInput,
    Modality,
    PatentAssumptions,
    PopulationInput,
    PriceBasis,
    PriceObservation,
    ProgramInput,
    RouteOfAdministration,
)
from labrador_roi.pricing import PricingInputs, calculate_pricing_corridor


class AnchorBucket(StrEnum):
    """Evaluation roles; configuration never inflates model-reality counts."""

    COMPUTATION = "computation"
    CROSS_SOURCE = "cross_source"
    CONTROL = "control"
    CONFIG = "config"


class AnchorStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class RealityAnchor:
    """One externally attributed measurement contract."""

    anchor_id: str
    bucket: AnchorBucket
    measure: str
    scenario: Mapping[str, Any]
    claim: str
    expected: tuple[float, float]
    unit: str
    source: str
    source_url: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RealityAnchor:
        required = {
            "id",
            "bucket",
            "measure",
            "scenario",
            "claim",
            "expected",
            "unit",
            "source",
            "source_url",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"reality anchor is missing fields: {', '.join(missing)}")
        expected = raw["expected"]
        if not isinstance(expected, list) or len(expected) != 2:
            raise ValueError(f"anchor {raw['id']!r} expected must be a two-value list")
        low, high = (float(expected[0]), float(expected[1]))
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"anchor {raw['id']!r} has an invalid expected band")
        scenario = raw["scenario"]
        if not isinstance(scenario, dict):
            raise ValueError(f"anchor {raw['id']!r} scenario must be an object")
        return cls(
            anchor_id=str(raw["id"]),
            bucket=AnchorBucket(str(raw["bucket"])),
            measure=str(raw["measure"]),
            scenario=dict(scenario),
            claim=str(raw["claim"]),
            expected=(low, high),
            unit=str(raw["unit"]),
            source=str(raw["source"]),
            source_url=str(raw["source_url"]),
        )


@dataclass(frozen=True)
class AdapterMeasurement:
    value: float
    reason: str


@dataclass(frozen=True)
class AnchorResult:
    anchor_id: str
    bucket: AnchorBucket
    status: AnchorStatus
    actual: float | None
    expected: tuple[float, float]
    unit: str
    reason: str
    claim: str
    source: str
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bucket"] = self.bucket.value
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class BucketCounts:
    total: int
    passed: int
    failed: int
    skipped: int

    @property
    def evaluated(self) -> int:
        return self.passed + self.failed


@dataclass(frozen=True)
class RealityReport:
    results: tuple[AnchorResult, ...]
    bucket_counts: Mapping[AnchorBucket, BucketCounts]

    @property
    def model_counts(self) -> BucketCounts:
        model_buckets = {
            AnchorBucket.COMPUTATION,
            AnchorBucket.CROSS_SOURCE,
            AnchorBucket.CONTROL,
        }
        counts = [self.bucket_counts[bucket] for bucket in model_buckets]
        return BucketCounts(
            total=sum(item.total for item in counts),
            passed=sum(item.passed for item in counts),
            failed=sum(item.failed for item in counts),
            skipped=sum(item.skipped for item in counts),
        )

    @property
    def exit_code(self) -> int:
        """Only a measured failure is fatal; an explicit capability gap remains visible."""

        return int(any(result.status == AnchorStatus.FAIL for result in self.results))

    def result(self, anchor_id: str) -> AnchorResult:
        return next(item for item in self.results if item.anchor_id == anchor_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [item.to_dict() for item in self.results],
            "bucket_counts": {
                bucket.value: {
                    **asdict(counts),
                    "evaluated": counts.evaluated,
                }
                for bucket, counts in self.bucket_counts.items()
            },
            "model_counts": {
                **asdict(self.model_counts),
                "evaluated": self.model_counts.evaluated,
            },
        }


@dataclass(frozen=True)
class RAEvaluationScenario:
    """Transparent inputs used only to exercise existing APIs against the cited anchors."""

    valuation_year: int = 2024
    adult_population: float = 262_000_000
    ra_prevalence: float = 0.0053
    diagnosed_fraction: float = 0.90
    treated_fraction: float = 0.70
    advanced_eligible_fraction: float = 0.45
    annual_persistence_rate: float = 0.73
    persistence_horizon_years: int = 15
    autoimmune_transitions: tuple[float, ...] = (0.552, 0.314, 0.653, 0.941)
    estimated_net_comparables: tuple[tuple[str, float, RouteOfAdministration], ...] = (
        ("adalimumab", 27_000.0, RouteOfAdministration.SUBCUTANEOUS_SELF),
        ("tofacitinib", 30_000.0, RouteOfAdministration.ORAL),
        ("upadacitinib", 40_000.0, RouteOfAdministration.ORAL),
    )


class UnsupportedAnchor(Exception):
    """A capability gap, distinct from an evaluated result outside its band."""


class LabradorRealityAdapter:
    """Map the portable anchor semantics to LABrador's existing public APIs."""

    def __init__(self, scenario: RAEvaluationScenario | None = None) -> None:
        self.scenario = scenario or RAEvaluationScenario()

    def actual(self, anchor: RealityAnchor) -> AdapterMeasurement:
        if anchor.anchor_id in {"pos_loa_chain", "autoimmune_loa_bracket"}:
            return AdapterMeasurement(
                self._autoimmune_loa(),
                "calculate_cashflow chained the four BIO-sourced transition inputs; this "
                "checks stage chaining, not an internal therapeutic-area prior library",
            )
        if anchor.anchor_id == "addressable_us":
            return AdapterMeasurement(
                self._addressable_population(),
                "calculate_cashflow applied the cited RA prevalence and successive diagnosis, "
                "treatment, and advanced-therapy gates",
            )
        if anchor.anchor_id == "prevalence_10x_breaches":
            return AdapterMeasurement(
                self._addressable_population(prevalence_multiplier=10.0),
                "the identical cash-flow population path was rerun with prevalence multiplied "
                "by ten; no expected value was substituted",
            )
        if anchor.anchor_id == "mean_time_on_therapy":
            return AdapterMeasurement(
                self._mean_time_on_therapy(),
                "summed LABrador's annual active-patient occupancy for one starter over a "
                "15-year horizon using 73% annual persistence",
            )
        if anchor.anchor_id == "net_price_oral":
            return AdapterMeasurement(
                self._oral_net_price(anchor),
                "calculate_pricing_corridor selected the median of two route-matched oral "
                "ESTIMATED_NET comparables; the SC comparable remained secondary and no "
                "list/WAC price was admitted",
            )
        if anchor.anchor_id == "config.discount_rate":
            return AdapterMeasurement(
                self._default_discount_rate(),
                "read from an instantiated ProgramCashFlowInputs default, not copied from the band",
            )

        unsupported = {
            "smallmol_cogs_pct": (
                "LABrador accepts annual manufacturer COGS as an evidenced input but does not "
                "estimate small-molecule COGS from SMILES"
            ),
            "peptide_cogs": (
                "LABrador accepts annual manufacturer COGS as an evidenced input but has no "
                "sequence/dose SPPS cost model"
            ),
            "modality_pos_ratio": (
                "development probabilities are explicit program inputs; LABrador has no "
                "peptide-versus-small-molecule PoS factor"
            ),
            "enpv_headline_order": (
                "the portable Phase-3 scenario omits the patient, access, adoption, cost, timing, "
                "and patent inputs LABrador requires; the engine intentionally has no RA defaults"
            ),
            "config.ira_smallmol": "LABrador does not yet model IRA/MFP eligibility or timing",
            "config.ira_biologic": "LABrador does not yet model IRA/MFP eligibility or timing",
            "config.phase3_cost": (
                "stage costs are supplied per program and LABrador has no small-molecule Phase-3 "
                "cost default"
            ),
        }
        reason = unsupported.get(anchor.anchor_id)
        if reason is None:
            raise RuntimeError(f"adapter has no mapping for anchor {anchor.anchor_id!r}")
        raise UnsupportedAnchor(reason)

    def _base_indication(
        self,
        *,
        indication_id: str,
        backlog_patients: float,
        persistence: float = 0.0,
        coverage: float = 1.0,
        authorization: float = 1.0,
        initiation: float = 1.0,
    ) -> IndicationCommercialAssumptions:
        return IndicationCommercialAssumptions(
            indication_id=indication_id,
            launch_year=self.scenario.valuation_year,
            route="ORAL",
            backlog_patients=backlog_patients,
            annual_incident_patients=0,
            coverage_rate=coverage,
            authorization_rate=authorization,
            patient_affordability_rate=1,
            initiation_rate=initiation,
            provider_capacity_rate=1,
            adoption_by_year={0: 1},
            annual_persistence_rate=persistence,
            dose_intensity=1,
            annual_gross_price=0,
        )

    def _cashflow_inputs(
        self,
        indication: IndicationCommercialAssumptions,
        *,
        forecast_end_year: int,
        development_stages: tuple[DevelopmentStage, ...] = (),
    ) -> ProgramCashFlowInputs:
        return ProgramCashFlowInputs(
            program_id=f"reality-{indication.indication_id}",
            valuation_year=self.scenario.valuation_year,
            forecast_end_year=forecast_end_year,
            patent=CashFlowPatentAssumptions(filing_year=self.scenario.valuation_year),
            initial_indication=indication,
            initial_development_stages=development_stages,
        )

    def _autoimmune_loa(self) -> float:
        stage_names = ("phase_1", "phase_2", "phase_3", "filing")
        stages = tuple(
            DevelopmentStage(
                name=name,
                year=self.scenario.valuation_year,
                cost=0,
                success_probability=probability,
            )
            for name, probability in zip(
                stage_names,
                self.scenario.autoimmune_transitions,
                strict=True,
            )
        )
        inputs = self._cashflow_inputs(
            self._base_indication(indication_id="ra-pos", backlog_patients=0),
            forecast_end_year=self.scenario.valuation_year,
            development_stages=stages,
        )
        return calculate_cashflow(
            inputs,
            calculate_delay_cost=False,
        ).initial_approval_probability

    def _addressable_population(self, prevalence_multiplier: float = 1.0) -> float:
        prevalent_ra = (
            self.scenario.adult_population * self.scenario.ra_prevalence * prevalence_multiplier
        )
        indication = self._base_indication(
            indication_id="ra-addressable",
            backlog_patients=prevalent_ra,
            coverage=self.scenario.diagnosed_fraction,
            authorization=self.scenario.treated_fraction,
            initiation=self.scenario.advanced_eligible_fraction,
        )
        inputs = self._cashflow_inputs(
            indication,
            forecast_end_year=self.scenario.valuation_year,
        )
        result = calculate_cashflow(inputs, calculate_delay_cost=False)
        return result.annual_cash_flows[0].initial_new_starts

    def _mean_time_on_therapy(self) -> float:
        indication = self._base_indication(
            indication_id="ra-persistence",
            backlog_patients=1,
            persistence=self.scenario.annual_persistence_rate,
        )
        inputs = self._cashflow_inputs(
            indication,
            forecast_end_year=(
                self.scenario.valuation_year + self.scenario.persistence_horizon_years
            ),
        )
        result = calculate_cashflow(inputs, calculate_delay_cost=False)
        return sum(row.initial_active_patients for row in result.annual_cash_flows)

    def _oral_net_price(self, anchor: RealityAnchor) -> float:
        source = EvidenceMetadata(
            source_id="ra-net-comparable-anchor",
            source_url=anchor.source_url,
            citation=anchor.source,
            evidence_type=EvidenceType.SECONDARY_RESEARCH,
            grade=EvidenceGrade.MODERATE,
        )
        synthetic = EvidenceMetadata(
            evidence_type=EvidenceType.SYNTHETIC,
            grade=EvidenceGrade.SYNTHETIC,
            synthetic=True,
            notes="Auxiliary evaluation fixture; it must not make the scenario decision-grade.",
        )
        population = self._addressable_population()
        access = AccessAssumptions(
            coverage_fraction=1,
            prior_authorization_pass_fraction=1,
            initiation_fraction=1,
            provider_capacity_fraction=1,
            patient_cost_share_fraction=0,
            adoption_by_year={0: 1},
            evidence={
                field: synthetic
                for field in (
                    "coverage_fraction",
                    "prior_authorization_pass_fraction",
                    "initiation_fraction",
                    "provider_capacity_fraction",
                    "patient_cost_share_fraction",
                )
            },
        )
        indication = IndicationInput(
            indication_id="ra",
            name="Rheumatoid arthritis",
            therapeutic_area="Autoimmune",
            target_population="Adults eligible for advanced therapy",
            line_of_therapy="Advanced therapy",
            geography="United States",
            currency="USD",
            launch_year=2030,
            route=RouteOfAdministration.ORAL,
            population=PopulationInput(
                eligible_patients=population,
                evidence={"eligible_patients": source},
            ),
            access=access,
        )
        program = ProgramInput(
            program_id="ra-net-price-evaluation",
            program_name="RA oral net-price evaluation fixture",
            target="evaluation target",
            modality=Modality.SMALL_MOLECULE,
            route=RouteOfAdministration.ORAL,
            base_year=self.scenario.valuation_year,
            valuation_year=self.scenario.valuation_year,
            currency="USD",
            initial_indication=indication,
            patent=PatentAssumptions(filing_year=self.scenario.valuation_year),
            development=DevelopmentAssumptions(current_stage="evaluation"),
        )
        comparables = ComparableSet(
            comparables=[
                ComparableTherapy(
                    comparable_id=f"ra-net-{name}",
                    name=name,
                    therapeutic_area=indication.therapeutic_area,
                    indication=indication.name,
                    target_population=indication.target_population,
                    line_of_therapy=indication.line_of_therapy,
                    geography=indication.geography,
                    route=route,
                    price=PriceObservation(
                        amount=amount,
                        currency="USD",
                        basis=PriceBasis.ESTIMATED_NET,
                        price_year=self.scenario.valuation_year,
                        evidence=source,
                    ),
                    notes=(
                        "Evaluation-only estimated-net anchor; never represented as an observed "
                        "confidential manufacturer net."
                    ),
                )
                for name, amount, route in self.scenario.estimated_net_comparables
            ]
        )
        auxiliary_inputs = {
            "incremental_qalys": 0.6,
            "willingness_to_pay_per_qaly": 150_000,
            "comparator_total_cost": 45_000,
            "new_non_drug_total_cost": 15_000,
            "expected_treatment_years": 3,
            "annual_comparator_drug_cost": 30_000,
            "annual_non_drug_cost_offsets": 5_000,
            "annual_payer_budget_limit": 10_000_000_000,
            "annual_manufacturer_cost": 3_000,
            "required_gross_margin_fraction": 0.8,
            "candidate_list_price": 80_000,
        }
        pricing = calculate_pricing_corridor(
            PricingInputs(
                program=program,
                comparables=comparables,
                evidence={key: synthetic for key in auxiliary_inputs},
                **auxiliary_inputs,
            )
        )
        if pricing.selected_annual_net_price is None:
            raise RuntimeError("RA estimated-net pricing fixture did not produce a corridor")
        return pricing.selected_annual_net_price

    def _default_discount_rate(self) -> float:
        inputs = self._cashflow_inputs(
            self._base_indication(indication_id="config", backlog_patients=0),
            forecast_end_year=self.scenario.valuation_year,
        )
        return inputs.discount_rate


def load_reality_anchors(
    anchors_path: str | Path | None = None,
) -> tuple[RealityAnchor, ...]:
    """Load and validate the portable anchor catalog."""

    if anchors_path is None:
        text = files(__package__).joinpath("reality_anchors.json").read_text(encoding="utf-8")
    else:
        text = Path(anchors_path).read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict) or not isinstance(raw.get("anchors"), list):
        raise ValueError("reality-anchor document must contain an anchors list")
    anchors = tuple(RealityAnchor.from_mapping(item) for item in raw["anchors"])
    identifiers = [anchor.anchor_id for anchor in anchors]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("reality-anchor IDs must be unique")
    return anchors


def evaluate_reality_anchors(
    *,
    anchors_path: str | Path | None = None,
) -> RealityReport:
    """Evaluate every portable anchor through the deterministic LABrador adapter."""

    adapter = LabradorRealityAdapter()
    results: list[AnchorResult] = []
    for anchor in load_reality_anchors(anchors_path):
        try:
            measurement = adapter.actual(anchor)
            actual = float(measurement.value)
            low, high = anchor.expected
            passed = math.isfinite(actual) and low <= actual <= high
            status = AnchorStatus.PASS if passed else AnchorStatus.FAIL
            reason = measurement.reason
            if not math.isfinite(actual):
                reason = f"{reason}; adapter returned a non-finite value"
        except UnsupportedAnchor as error:
            actual = None
            status = AnchorStatus.SKIP
            reason = str(error)
        except Exception as error:  # unexpected errors must be visible failures, never skips
            actual = None
            status = AnchorStatus.FAIL
            reason = f"adapter error: {type(error).__name__}: {error}"
        results.append(
            AnchorResult(
                anchor_id=anchor.anchor_id,
                bucket=anchor.bucket,
                status=status,
                actual=actual,
                expected=anchor.expected,
                unit=anchor.unit,
                reason=reason,
                claim=anchor.claim,
                source=anchor.source,
                source_url=anchor.source_url,
            )
        )
    counts: dict[AnchorBucket, BucketCounts] = {}
    for bucket in AnchorBucket:
        bucket_results = [item for item in results if item.bucket == bucket]
        counts[bucket] = BucketCounts(
            total=len(bucket_results),
            passed=sum(item.status == AnchorStatus.PASS for item in bucket_results),
            failed=sum(item.status == AnchorStatus.FAIL for item in bucket_results),
            skipped=sum(item.status == AnchorStatus.SKIP for item in bucket_results),
        )
    return RealityReport(results=tuple(results), bucket_counts=counts)


def format_reality_report(report: RealityReport) -> str:
    """Render a concise human-readable report without changing scoring semantics."""

    lines = ["LABrador RA/I&I reality anchors"]
    for bucket in AnchorBucket:
        counts = report.bucket_counts[bucket]
        lines.append(
            f"\n[{bucket.value}] {counts.passed}/{counts.evaluated} evaluated pass; "
            f"{counts.skipped} skipped"
        )
        for result in (item for item in report.results if item.bucket == bucket):
            low, high = result.expected
            band = f"{low:g}-{high:g}" if high < 1e18 else f">={low:g}"
            actual = "not evaluated" if result.actual is None else f"{result.actual:.6g}"
            lines.append(
                f"  {result.status.value:<4} {result.anchor_id:<28} band {band:<18} actual {actual}"
            )
            lines.append(f"       {result.reason}")
    model = report.model_counts
    lines.append(
        f"\nMODEL checks: {model.passed}/{model.evaluated} evaluated pass; "
        f"{model.skipped} unsupported (config excluded)"
    )
    return "\n".join(lines)
