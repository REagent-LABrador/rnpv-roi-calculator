"""Auditable four-layer payer price and access corridor calculations."""

from __future__ import annotations

from enum import StrEnum
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, model_validator

from labrador_roi.comparables import (
    ComparableSelection,
    ComparableSet,
    ComparableTier,
    select_comparables,
)
from labrador_roi.models import (
    CalculationStep,
    DecisionGrade,
    EvidenceMetadata,
    PriceBasis,
    ProgramInput,
    WarningRecord,
    WarningSeverity,
)


class ComparablePriceAnchor(BaseModel):
    """An annualized anchor for exactly one price basis and comparable tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    basis: PriceBasis
    tier: ComparableTier
    currency: str
    lower: float
    median: float
    upper: float
    comparable_ids: list[str]
    price_years: list[int]


class PatientOOPBasis(StrEnum):
    """How an access estimate obtained its annual patient out-of-pocket amount."""

    EXPLICIT_ANNUAL_PATIENT_OOP = "EXPLICIT_ANNUAL_PATIENT_OOP"
    ZERO_PATIENT_COST_SHARE = "ZERO_PATIENT_COST_SHARE"
    MANUFACTURER_NET_PROXY = "MANUFACTURER_NET_PROXY"
    UNKNOWN = "UNKNOWN"


class AccessEstimate(BaseModel):
    """Access at one annual net price, with unknown income data kept explicit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    annual_net_price: float
    expected_patient_oop: float | None
    patient_oop_basis: PatientOOPBasis
    income_affordable_share: float | None
    known_affordable_lower_bound: float
    unknown_income_share: float
    system_access_fraction: float | None
    total_access_fraction: float | None
    accessible_patients: float | None
    payer_paid_per_patient: float | None
    warnings: list[WarningRecord] = Field(default_factory=list)


class PriceCorridor(BaseModel):
    """Feasible and comparable-calibrated annual net-price bounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    basis: PriceBasis
    currency: str
    feasible_lower: float
    feasible_upper: float
    likely_lower: float
    likely_midpoint: float
    likely_upper: float
    selected_annual_net_price: float
    value_based_ceiling: float
    payer_affordability_ceiling: float | None
    commercial_floor: float
    comparable_anchor_lower: float | None
    comparable_anchor_median: float | None
    comparable_anchor_upper: float | None
    candidate_list_price: float | None
    estimated_gross_to_net_fraction: float | None


class PricingInputs(BaseModel):
    """Inputs for one program-indication payer corridor.

    Costs and health outcomes are per patient over the health-economic horizon unless explicitly
    named annual.  Every critical value has a same-named entry in ``evidence``.
    """

    model_config = ConfigDict(extra="forbid")

    program: ProgramInput
    comparables: ComparableSet
    indication_id: str | None = None
    incremental_qalys: float | None = None
    willingness_to_pay_per_qaly: float | None = Field(default=None, gt=0)
    comparator_total_cost: float | None = Field(default=None, ge=0)
    new_non_drug_total_cost: float | None = Field(default=None, ge=0)
    expected_treatment_years: float | None = Field(default=None, gt=0)
    annual_comparator_drug_cost: float | None = Field(default=None, ge=0)
    annual_non_drug_cost_offsets: float | None = Field(default=None, ge=0)
    annual_payer_budget_limit: float | None = Field(default=None, ge=0)
    annual_manufacturer_cost: float | None = Field(default=None, ge=0)
    required_gross_margin_fraction: float | None = Field(default=None, ge=0, lt=1)
    selected_net_anchor_basis: PriceBasis = PriceBasis.ESTIMATED_NET
    candidate_list_price: float | None = Field(default=None, gt=0)
    access_year_offset: int = Field(default=0, ge=0)
    evidence: dict[str, EvidenceMetadata] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selected_basis_and_indication(self) -> PricingInputs:
        if self.selected_net_anchor_basis not in {
            PriceBasis.ESTIMATED_NET,
            PriceBasis.OBSERVED_NET,
        }:
            raise ValueError("the obtainable net corridor requires an explicit net price basis")
        indication = self.program.indication(self.indication_id)
        if indication.indication_id not in {
            self.program.initial_indication.indication_id,
            *(item.indication_id for item in self.program.expansion_indications),
        }:
            raise ValueError("pricing indication is not part of the program")
        return self


class PricingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    program_id: str
    indication_id: str
    decision_grade: DecisionGrade
    annual_net_price_corridor: PriceCorridor | None
    comparable_selection: ComparableSelection
    comparable_anchors: list[ComparablePriceAnchor]
    access_estimates: list[AccessEstimate]
    calculation_steps: list[CalculationStep]
    warnings: list[WarningRecord]
    critical_evidence_status: dict[str, bool]

    @property
    def selected_annual_net_price(self) -> float | None:
        if self.annual_net_price_corridor is None:
            return None
        return self.annual_net_price_corridor.selected_annual_net_price


_CRITICAL_INPUT_FIELDS = (
    "incremental_qalys",
    "willingness_to_pay_per_qaly",
    "comparator_total_cost",
    "new_non_drug_total_cost",
    "expected_treatment_years",
    "annual_comparator_drug_cost",
    "annual_non_drug_cost_offsets",
    "annual_payer_budget_limit",
    "annual_manufacturer_cost",
    "required_gross_margin_fraction",
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def build_comparable_anchors(selection: ComparableSelection) -> list[ComparablePriceAnchor]:
    """Build one anchor per basis, preferring primary without blending secondary records."""

    anchors: list[ComparablePriceAnchor] = []
    for basis in PriceBasis:
        matching = [
            item for item in selection.anchor_candidates() if item.comparable.price.basis == basis
        ]
        primary = [item for item in matching if item.tier == ComparableTier.PRIMARY]
        selected = primary or [item for item in matching if item.tier == ComparableTier.SECONDARY]
        if not selected:
            continue
        values = [item.comparable.price.annualized_amount() for item in selected]
        anchors.append(
            ComparablePriceAnchor(
                basis=basis,
                tier=ComparableTier.PRIMARY if primary else ComparableTier.SECONDARY,
                currency=selected[0].comparable.price.currency,
                lower=_percentile(values, 0.25),
                median=median(values),
                upper=_percentile(values, 0.75),
                comparable_ids=[item.comparable.comparable_id for item in selected],
                price_years=sorted({item.comparable.price.price_year for item in selected}),
            )
        )
    return anchors


def evaluate_access_at_price(
    program: ProgramInput,
    annual_net_price: float,
    indication_id: str | None = None,
    year_offset: int = 0,
) -> AccessEstimate:
    """Apply income only to patient OOP affordability and access, never clinical value."""

    indication = program.indication(indication_id)
    access = indication.access
    warnings: list[WarningRecord] = []
    system_access = access.system_access_fraction(year_offset)
    eligible = indication.population.eligible_patients

    cost_share = access.patient_cost_share_fraction
    if cost_share is None and access.universal_or_public_coverage:
        cost_share = 0.0
    if access.annual_patient_oop is not None:
        expected_oop = access.annual_patient_oop
        patient_oop_basis = PatientOOPBasis.EXPLICIT_ANNUAL_PATIENT_OOP
    elif cost_share == 0:
        expected_oop = 0.0
        patient_oop_basis = PatientOOPBasis.ZERO_PATIENT_COST_SHARE
    elif cost_share is None:
        expected_oop = None
        patient_oop_basis = PatientOOPBasis.UNKNOWN
    else:
        expected_oop = annual_net_price * cost_share
        patient_oop_basis = PatientOOPBasis.MANUFACTURER_NET_PROXY
        warnings.append(
            WarningRecord(
                code="PATIENT_OOP_NET_PRICE_PROXY",
                field="annual_patient_oop",
                severity=WarningSeverity.ERROR,
                message=(
                    "Annual patient OOP was not supplied; the displayed fallback applies "
                    "cost share to manufacturer net price and is not decision-grade."
                ),
            )
        )
    if patient_oop_basis == PatientOOPBasis.EXPLICIT_ANNUAL_PATIENT_OOP:
        payer_paid = max(0.0, annual_net_price - expected_oop)
    elif cost_share is None:
        payer_paid = None
    else:
        payer_paid = annual_net_price * (1 - cost_share)

    known_affordable = 0.0
    unknown_share = 0.0
    affordable_share: float | None
    if expected_oop == 0:
        affordable_share = 1.0
        known_affordable = 1.0
    elif expected_oop is None:
        affordable_share = None
        unknown_share = 1.0
        warnings.append(
            WarningRecord(
                code="UNKNOWN_PATIENT_COST_SHARE",
                field="patient_cost_share_fraction",
                message="Patient OOP burden and income-mediated access are unknown.",
            )
        )
    elif not indication.income_bands:
        affordable_share = None
        unknown_share = 1.0
        warnings.append(
            WarningRecord(
                code="MISSING_INCOME_BANDS",
                field="income_bands",
                message="Income-mediated access is unknown; no wealth default was applied.",
            )
        )
    else:
        for band in indication.income_bands:
            if band.annual_income is None or band.maximum_oop_share is None:
                unknown_share += band.population_share
                continue
            maximum_oop = band.annual_income * band.maximum_oop_share
            if expected_oop is not None and expected_oop <= maximum_oop:
                known_affordable += band.population_share
        covered_share = sum(band.population_share for band in indication.income_bands)
        unknown_share += max(0.0, 1.0 - covered_share)
        affordable_share = known_affordable if unknown_share <= 1e-9 else None
        if affordable_share is None:
            warnings.append(
                WarningRecord(
                    code="PARTIAL_INCOME_DATA",
                    field="income_bands",
                    message=(
                        "Affordable share is unknown because income-band coverage is incomplete."
                    ),
                )
            )

    total_access = None
    if system_access is not None and affordable_share is not None:
        total_access = system_access * affordable_share
    accessible_patients = None
    if total_access is not None and eligible is not None:
        accessible_patients = eligible * total_access
    if system_access is None:
        warnings.append(
            WarningRecord(
                code="INCOMPLETE_SYSTEM_ACCESS",
                field="access",
                message="Coverage, authorization, initiation, capacity, or adoption is unknown.",
            )
        )
    if eligible is None:
        warnings.append(
            WarningRecord(
                code="MISSING_ELIGIBLE_POPULATION",
                field="eligible_patients",
                message="Accessible patient count cannot be calculated without eligible patients.",
            )
        )

    return AccessEstimate(
        annual_net_price=annual_net_price,
        expected_patient_oop=expected_oop,
        patient_oop_basis=patient_oop_basis,
        income_affordable_share=affordable_share,
        known_affordable_lower_bound=known_affordable,
        unknown_income_share=unknown_share,
        system_access_fraction=system_access,
        total_access_fraction=total_access,
        accessible_patients=accessible_patients,
        payer_paid_per_patient=payer_paid,
        warnings=warnings,
    )


def _supported_input_status(inputs: PricingInputs) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for field_name in _CRITICAL_INPUT_FIELDS:
        metadata = inputs.evidence.get(field_name)
        status[field_name] = getattr(inputs, field_name) is not None and bool(
            metadata and metadata.supports_decision
        )
    indication = inputs.program.indication(inputs.indication_id)
    population_evidence = indication.population.evidence.get("eligible_patients")
    status["eligible_patients"] = indication.population.eligible_patients is not None and bool(
        population_evidence and population_evidence.supports_decision
    )
    access_fields = (
        "coverage_fraction",
        "prior_authorization_pass_fraction",
        "initiation_fraction",
        "provider_capacity_fraction",
    )
    for field_name in access_fields:
        metadata = indication.access.evidence.get(field_name)
        status[f"access.{field_name}"] = getattr(
            indication.access, field_name
        ) is not None and bool(metadata and metadata.supports_decision)
    oop_metadata = indication.access.evidence.get("annual_patient_oop")
    if indication.access.annual_patient_oop is not None:
        status["access.annual_patient_oop"] = bool(oop_metadata and oop_metadata.supports_decision)
    else:
        uses_zero_oop_public_coverage = (
            indication.access.universal_or_public_coverage
            and indication.access.patient_cost_share_fraction is None
        )
        if uses_zero_oop_public_coverage:
            metadata = indication.access.evidence.get("universal_or_public_coverage")
            status["access.universal_or_public_coverage"] = bool(
                metadata and metadata.supports_decision
            )
        else:
            metadata = indication.access.evidence.get("patient_cost_share_fraction")
            status["access.patient_cost_share_fraction"] = (
                indication.access.patient_cost_share_fraction is not None
                and bool(metadata and metadata.supports_decision)
            )
        if (indication.access.patient_cost_share_fraction or 0) > 0:
            status["access.annual_patient_oop"] = False
    return status


def calculate_pricing_corridor(inputs: PricingInputs) -> PricingResult:
    """Calculate value, market, payer-affordability, and commercial-floor layers."""

    indication = inputs.program.indication(inputs.indication_id)
    selection = select_comparables(inputs.program, inputs.comparables, indication.indication_id)
    anchors = build_comparable_anchors(selection)
    anchor = next(
        (item for item in anchors if item.basis == inputs.selected_net_anchor_basis),
        None,
    )
    warnings: list[WarningRecord] = []
    steps: list[CalculationStep] = []
    evidence_status = _supported_input_status(inputs)

    value_ceiling: float | None = None
    value_inputs = (
        inputs.incremental_qalys,
        inputs.willingness_to_pay_per_qaly,
        inputs.comparator_total_cost,
        inputs.new_non_drug_total_cost,
        inputs.expected_treatment_years,
    )
    if all(item is not None for item in value_inputs):
        assert inputs.incremental_qalys is not None
        assert inputs.willingness_to_pay_per_qaly is not None
        assert inputs.comparator_total_cost is not None
        assert inputs.new_non_drug_total_cost is not None
        assert inputs.expected_treatment_years is not None
        raw_ceiling = (
            inputs.willingness_to_pay_per_qaly * inputs.incremental_qalys
            + inputs.comparator_total_cost
            - inputs.new_non_drug_total_cost
        ) / inputs.expected_treatment_years
        value_ceiling = max(0.0, raw_ceiling)
        if raw_ceiling < 0:
            warnings.append(
                WarningRecord(
                    code="NEGATIVE_VALUE_CEILING",
                    message=(
                        "The calculated maximum acquisition price was negative "
                        "and is shown as zero."
                    ),
                )
            )
    steps.append(
        CalculationStep(
            step_id="value_ceiling",
            label="Value-based annual net-price ceiling",
            formula=(
                "max(0, (WTP * incremental QALYs + comparator total cost "
                "- new non-drug total cost) / treatment years)"
            ),
            inputs={
                "incremental_qalys": inputs.incremental_qalys,
                "wtp_per_qaly": inputs.willingness_to_pay_per_qaly,
                "comparator_total_cost": inputs.comparator_total_cost,
                "new_non_drug_total_cost": inputs.new_non_drug_total_cost,
                "expected_treatment_years": inputs.expected_treatment_years,
            },
            result=value_ceiling,
            unit=f"{indication.currency}/patient-year",
            evidence_keys=list(_CRITICAL_INPUT_FIELDS[:5]),
            notes=["Patient income is deliberately absent from this formula."],
        )
    )

    system_access = indication.access.system_access_fraction(inputs.access_year_offset)
    eligible = indication.population.eligible_patients
    cost_share = indication.access.patient_cost_share_fraction
    if cost_share is None and indication.access.universal_or_public_coverage:
        cost_share = 0.0
    affordability_ceiling: float | None = None
    affordability_values = (
        inputs.annual_comparator_drug_cost,
        inputs.annual_non_drug_cost_offsets,
        inputs.annual_payer_budget_limit,
        system_access,
        eligible,
    )
    if all(item is not None for item in affordability_values):
        assert inputs.annual_comparator_drug_cost is not None
        assert inputs.annual_non_drug_cost_offsets is not None
        assert inputs.annual_payer_budget_limit is not None
        assert system_access is not None
        assert eligible is not None
        system_accessible_patients = eligible * system_access
        payer_budget_per_patient = (
            inputs.annual_comparator_drug_cost
            + inputs.annual_non_drug_cost_offsets
            + inputs.annual_payer_budget_limit / system_accessible_patients
            if system_accessible_patients > 0
            else None
        )
        if payer_budget_per_patient is not None:
            if indication.access.annual_patient_oop is not None:
                affordability_ceiling = (
                    payer_budget_per_patient + indication.access.annual_patient_oop
                )
            elif cost_share is not None and cost_share < 1:
                affordability_ceiling = payer_budget_per_patient / (1 - cost_share)
    steps.append(
        CalculationStep(
            step_id="payer_affordability",
            label="Payer affordability ceiling",
            formula=(
                "payer budget per patient + explicit annual patient OOP; when OOP is missing, "
                "the non-decision-grade fallback divides payer budget by (1 - cost share)"
            ),
            inputs={
                "annual_comparator_drug_cost": inputs.annual_comparator_drug_cost,
                "annual_non_drug_cost_offsets": inputs.annual_non_drug_cost_offsets,
                "annual_payer_budget_limit": inputs.annual_payer_budget_limit,
                "eligible_patients": eligible,
                "system_access_fraction": system_access,
                "patient_cost_share_fraction": cost_share,
                "annual_patient_oop": indication.access.annual_patient_oop,
            },
            result=affordability_ceiling,
            unit=f"{indication.currency}/patient-year",
            evidence_keys=[
                "annual_comparator_drug_cost",
                "annual_non_drug_cost_offsets",
                "annual_payer_budget_limit",
                "eligible_patients",
            ],
            notes=[
                "Income-mediated non-initiation is reported separately, not used to raise price."
            ],
        )
    )

    commercial_floor: float | None = None
    if (
        inputs.annual_manufacturer_cost is not None
        and inputs.required_gross_margin_fraction is not None
    ):
        commercial_floor = inputs.annual_manufacturer_cost / (
            1 - inputs.required_gross_margin_fraction
        )
    steps.append(
        CalculationStep(
            step_id="commercial_floor",
            label="Commercial viability floor",
            formula="annual manufacturer cost / (1 - required gross margin)",
            inputs={
                "annual_manufacturer_cost": inputs.annual_manufacturer_cost,
                "required_gross_margin_fraction": inputs.required_gross_margin_fraction,
            },
            result=commercial_floor,
            unit=f"{indication.currency}/patient-year",
            evidence_keys=["annual_manufacturer_cost", "required_gross_margin_fraction"],
            notes=["This is a manufacturer constraint, not payer willingness to pay."],
        )
    )

    for price_anchor in anchors:
        price_year_mismatch = any(
            year != inputs.program.base_year for year in price_anchor.price_years
        )
        if len(price_anchor.price_years) > 1 or price_year_mismatch:
            warnings.append(
                WarningRecord(
                    code="UNADJUSTED_PRICE_YEAR",
                    field=(
                        "selected_net_comparable_anchor"
                        if price_anchor is anchor
                        else "comparable_anchors"
                    ),
                    severity=(
                        WarningSeverity.ERROR if price_anchor is anchor else WarningSeverity.WARNING
                    ),
                    message=(
                        f"{price_anchor.basis} anchor retains source price years "
                        f"{price_anchor.price_years}; every selected anchor price must equal "
                        f"base year {inputs.program.base_year} until an explicit adjustment "
                        "method is supplied."
                    ),
                )
            )
    if anchor is None:
        warnings.append(
            WarningRecord(
                code="MISSING_SELECTED_NET_ANCHOR",
                field="selected_net_anchor_basis",
                severity=WarningSeverity.ERROR,
                message=(
                    f"No usable {inputs.selected_net_anchor_basis} primary or secondary "
                    "comparable was available; other price bases were not substituted."
                ),
            )
        )
    else:
        used_comparable_ids = set(anchor.comparable_ids)
        evidence_status["selected_net_anchor_price_year"] = all(
            year == inputs.program.base_year for year in anchor.price_years
        )
        for assessment in selection.anchor_candidates():
            price = assessment.comparable.price
            if assessment.comparable.comparable_id in used_comparable_ids:
                key = f"comparable:{assessment.comparable.comparable_id}:price"
                evidence_status[key] = price.evidence.supports_decision

    if inputs.candidate_list_price is not None:
        candidate_list_evidence = inputs.evidence.get("candidate_list_price")
        evidence_status["candidate_list_price"] = bool(
            candidate_list_evidence and candidate_list_evidence.supports_decision
        )

    corridor: PriceCorridor | None = None
    if value_ceiling is not None and commercial_floor is not None:
        hard_ceilings = [value_ceiling]
        if affordability_ceiling is not None:
            hard_ceilings.append(affordability_ceiling)
        if inputs.candidate_list_price is not None:
            hard_ceilings.append(inputs.candidate_list_price)
        feasible_upper = min(hard_ceilings)
        if commercial_floor <= feasible_upper:
            if anchor is None:
                likely_lower = commercial_floor
                likely_upper = feasible_upper
                likely_midpoint = (commercial_floor + feasible_upper) / 2
                anchor_lower = anchor_median = anchor_upper = None
            else:

                def clip(value: float) -> float:
                    return min(feasible_upper, max(commercial_floor, value))

                likely_lower = clip(anchor.lower)
                likely_midpoint = clip(anchor.median)
                likely_upper = clip(anchor.upper)
                anchor_lower = anchor.lower
                anchor_median = anchor.median
                anchor_upper = anchor.upper
            gross_to_net: float | None = None
            if inputs.candidate_list_price is not None:
                gross_to_net = 1 - likely_midpoint / inputs.candidate_list_price
            corridor = PriceCorridor(
                basis=inputs.selected_net_anchor_basis,
                currency=indication.currency,
                feasible_lower=commercial_floor,
                feasible_upper=feasible_upper,
                likely_lower=likely_lower,
                likely_midpoint=likely_midpoint,
                likely_upper=likely_upper,
                selected_annual_net_price=likely_midpoint,
                value_based_ceiling=value_ceiling,
                payer_affordability_ceiling=affordability_ceiling,
                commercial_floor=commercial_floor,
                comparable_anchor_lower=anchor_lower,
                comparable_anchor_median=anchor_median,
                comparable_anchor_upper=anchor_upper,
                candidate_list_price=inputs.candidate_list_price,
                estimated_gross_to_net_fraction=gross_to_net,
            )
        else:
            warnings.append(
                WarningRecord(
                    code="NO_VIABLE_PRICE_OVERLAP",
                    severity=WarningSeverity.ERROR,
                    message="Commercial floor exceeds the payer/value ceiling.",
                )
            )

    access_estimates: list[AccessEstimate] = []
    access_prices = {
        item
        for item in (
            value_ceiling,
            affordability_ceiling,
            commercial_floor,
            corridor.selected_annual_net_price if corridor else None,
        )
        if item is not None
    }
    for price in sorted(access_prices):
        estimate = evaluate_access_at_price(
            inputs.program,
            price,
            indication.indication_id,
            inputs.access_year_offset,
        )
        access_estimates.append(estimate)
        warnings.extend(estimate.warnings)

    requires_income_evidence = (
        indication.access.annual_patient_oop is not None
        and indication.access.annual_patient_oop > 0
    ) or (
        indication.access.annual_patient_oop is None
        and (indication.access.patient_cost_share_fraction or 0) > 0
    )
    if requires_income_evidence:
        income_complete = bool(indication.income_bands)
        income_complete = (
            income_complete
            and abs(sum(item.population_share for item in indication.income_bands) - 1) <= 1e-6
        )
        income_complete = income_complete and all(
            item.annual_income is not None
            and item.maximum_oop_share is not None
            and item.evidence.supports_decision
            for item in indication.income_bands
        )
        evidence_status["income_bands"] = income_complete

    if anchor is None:
        evidence_status["selected_net_comparable_anchor"] = False
    else:
        evidence_status["selected_net_comparable_anchor"] = all(
            evidence_status.get(f"comparable:{item_id}:price", False)
            for item_id in anchor.comparable_ids
        )

    for field_name, supported in evidence_status.items():
        if not supported:
            warnings.append(
                WarningRecord(
                    code="UNSUPPORTED_CRITICAL_INPUT",
                    field=field_name,
                    severity=WarningSeverity.ERROR,
                    message=f"Critical input '{field_name}' lacks decision-supporting evidence.",
                )
            )
    decision_grade = (
        DecisionGrade.DECISION_GRADE
        if corridor is not None and all(evidence_status.values())
        else DecisionGrade.NOT_DECISION_GRADE
    )

    return PricingResult(
        program_id=inputs.program.program_id,
        indication_id=indication.indication_id,
        decision_grade=decision_grade,
        annual_net_price_corridor=corridor,
        comparable_selection=selection,
        comparable_anchors=anchors,
        access_estimates=access_estimates,
        calculation_steps=steps,
        warnings=warnings,
        critical_evidence_status=evidence_status,
    )
