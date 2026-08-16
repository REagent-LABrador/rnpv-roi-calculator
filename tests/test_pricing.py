import pytest

from labrador_roi.comparables import ComparableSet
from labrador_roi.models import (
    AccessAssumptions,
    ComparableTherapy,
    DecisionGrade,
    DevelopmentAssumptions,
    EvidenceGrade,
    EvidenceMetadata,
    EvidenceType,
    IncomeBandAssumptions,
    IndicationInput,
    Modality,
    PatentAssumptions,
    PopulationInput,
    PriceBasis,
    PriceObservation,
    ProgramInput,
    RouteOfAdministration,
)
from labrador_roi.pricing import (
    PatientOOPBasis,
    PricingInputs,
    calculate_pricing_corridor,
)


def supported(source_id: str = "test-source") -> EvidenceMetadata:
    return EvidenceMetadata(
        source_id=source_id,
        evidence_type=EvidenceType.PAYER_OR_HTA,
        grade=EvidenceGrade.MODERATE,
    )


def synthetic() -> EvidenceMetadata:
    return EvidenceMetadata(
        evidence_type=EvidenceType.SYNTHETIC,
        grade=EvidenceGrade.SYNTHETIC,
        synthetic=True,
    )


def make_program(
    *,
    income_bands: list[IncomeBandAssumptions] | None = None,
    universal: bool = False,
    cost_share: float | None = 0.1,
    annual_oop: float | None = 2_200,
) -> ProgramInput:
    source = supported()
    access_evidence = {
        "coverage_fraction": source,
        "prior_authorization_pass_fraction": source,
        "initiation_fraction": source,
        "provider_capacity_fraction": source,
    }
    if cost_share is not None:
        access_evidence["patient_cost_share_fraction"] = source
    elif universal:
        access_evidence["universal_or_public_coverage"] = source
    if annual_oop is not None:
        access_evidence["annual_patient_oop"] = source
    indication = IndicationInput(
        indication_id="psoriasis",
        name="Plaque psoriasis",
        therapeutic_area="Inflammatory dermatology",
        target_population="Adults with moderate to severe disease",
        line_of_therapy="After systemic therapy failure",
        geography="United States",
        currency="USD",
        launch_year=2030,
        population=PopulationInput(
            eligible_patients=100,
            evidence={"eligible_patients": source},
        ),
        access=AccessAssumptions(
            universal_or_public_coverage=universal,
            coverage_fraction=0.8,
            prior_authorization_pass_fraction=0.5,
            initiation_fraction=0.5,
            provider_capacity_fraction=1.0,
            patient_cost_share_fraction=cost_share,
            annual_patient_oop=annual_oop,
            evidence=access_evidence,
        ),
        income_bands=income_bands or [],
    )
    return ProgramInput(
        program_id="p1",
        program_name="TYK2 program",
        target="TYK2",
        modality=Modality.SMALL_MOLECULE,
        route=RouteOfAdministration.ORAL,
        base_year=2026,
        valuation_year=2026,
        currency="USD",
        initial_indication=indication,
        patent=PatentAssumptions(filing_year=2025),
        development=DevelopmentAssumptions(current_stage="PRECLINICAL"),
    )


def make_comparable(
    comparable_id: str,
    amount: float,
    basis: PriceBasis,
    *,
    evidence: EvidenceMetadata | None = None,
    price_year: int = 2026,
) -> ComparableTherapy:
    return ComparableTherapy(
        comparable_id=comparable_id,
        name=comparable_id,
        therapeutic_area="Inflammatory dermatology",
        indication="Plaque psoriasis",
        target_population="Adults with moderate to severe disease",
        line_of_therapy="After systemic therapy failure",
        geography="United States",
        route=RouteOfAdministration.ORAL,
        price=PriceObservation(
            amount=amount,
            currency="USD",
            basis=basis,
            price_year=price_year,
            evidence=evidence or supported(f"price-{comparable_id}"),
        ),
    )


def complete_evidence() -> dict[str, EvidenceMetadata]:
    return {
        name: supported(name)
        for name in (
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
            "candidate_list_price",
        )
    }


def pricing_inputs(program: ProgramInput, comparables: list[ComparableTherapy]) -> PricingInputs:
    return PricingInputs(
        program=program,
        comparables=ComparableSet(comparables=comparables),
        incremental_qalys=0.5,
        willingness_to_pay_per_qaly=100_000,
        comparator_total_cost=20_000,
        new_non_drug_total_cost=10_000,
        expected_treatment_years=2,
        annual_comparator_drug_cost=10_000,
        annual_non_drug_cost_offsets=5_000,
        annual_payer_budget_limit=1_000_000,
        annual_manufacturer_cost=8_000,
        required_gross_margin_fraction=0.5,
        candidate_list_price=30_000,
        evidence=complete_evidence(),
    )


def test_four_layer_corridor_is_auditable_and_price_bases_do_not_mix() -> None:
    income = [
        IncomeBandAssumptions(
            name="lower",
            population_share=0.5,
            annual_income=20_000,
            maximum_oop_share=0.05,
            evidence=supported("income-lower"),
        ),
        IncomeBandAssumptions(
            name="higher",
            population_share=0.5,
            annual_income=80_000,
            maximum_oop_share=0.05,
            evidence=supported("income-higher"),
        ),
    ]
    program = make_program(income_bands=income)
    result = calculate_pricing_corridor(
        pricing_inputs(
            program,
            [
                make_comparable("list", 100_000, PriceBasis.LIST),
                make_comparable("net-a", 20_000, PriceBasis.ESTIMATED_NET),
                make_comparable("net-b", 24_000, PriceBasis.ESTIMATED_NET),
            ],
        )
    )

    assert result.decision_grade == DecisionGrade.DECISION_GRADE
    assert {anchor.basis for anchor in result.comparable_anchors} == {
        PriceBasis.LIST,
        PriceBasis.ESTIMATED_NET,
    }
    corridor = result.annual_net_price_corridor
    assert corridor is not None
    assert corridor.value_based_ceiling == 30_000
    assert corridor.commercial_floor == 16_000
    assert corridor.comparable_anchor_median == 22_000
    assert corridor.selected_annual_net_price == 22_000
    assert corridor.estimated_gross_to_net_fraction == 1 - 22_000 / 30_000
    selected_access = next(
        item for item in result.access_estimates if item.annual_net_price == 22_000
    )
    assert selected_access.expected_patient_oop == 2_200
    assert selected_access.patient_oop_basis == PatientOOPBasis.EXPLICIT_ANNUAL_PATIENT_OOP
    assert [step.step_id for step in result.calculation_steps] == [
        "value_ceiling",
        "payer_affordability",
        "commercial_floor",
    ]


def test_income_changes_oop_access_but_not_clinical_value_ceiling() -> None:
    low_income = [
        IncomeBandAssumptions(
            name="all",
            population_share=1,
            annual_income=20_000,
            maximum_oop_share=0.01,
            evidence=supported("low-income"),
        )
    ]
    high_income = [
        IncomeBandAssumptions(
            name="all",
            population_share=1,
            annual_income=200_000,
            maximum_oop_share=0.10,
            evidence=supported("high-income"),
        )
    ]
    comparable = make_comparable("net", 22_000, PriceBasis.ESTIMATED_NET)
    low_result = calculate_pricing_corridor(
        pricing_inputs(make_program(income_bands=low_income), [comparable])
    )
    high_result = calculate_pricing_corridor(
        pricing_inputs(make_program(income_bands=high_income), [comparable])
    )

    assert low_result.annual_net_price_corridor is not None
    assert high_result.annual_net_price_corridor is not None
    assert (
        low_result.annual_net_price_corridor.value_based_ceiling
        == high_result.annual_net_price_corridor.value_based_ceiling
    )
    low_selected = next(
        item
        for item in low_result.access_estimates
        if item.annual_net_price == low_result.selected_annual_net_price
    )
    high_selected = next(
        item
        for item in high_result.access_estimates
        if item.annual_net_price == high_result.selected_annual_net_price
    )
    assert low_selected.income_affordable_share == 0
    assert high_selected.income_affordable_share == 1


def test_missing_income_or_synthetic_critical_price_is_not_decision_grade() -> None:
    result = calculate_pricing_corridor(
        pricing_inputs(
            make_program(income_bands=[]),
            [
                make_comparable(
                    "synthetic-net",
                    22_000,
                    PriceBasis.ESTIMATED_NET,
                    evidence=synthetic(),
                )
            ],
        )
    )

    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.critical_evidence_status["income_bands"] is False
    assert result.critical_evidence_status["selected_net_comparable_anchor"] is False
    assert any(item.code == "MISSING_INCOME_BANDS" for item in result.warnings)


def test_unused_secondary_price_evidence_does_not_downgrade_primary_anchor() -> None:
    program = make_program(
        income_bands=[
            IncomeBandAssumptions(
                name="all",
                population_share=1,
                annual_income=80_000,
                maximum_oop_share=0.05,
                evidence=supported("income"),
            )
        ]
    )
    primary = make_comparable("primary", 22_000, PriceBasis.ESTIMATED_NET)
    secondary = make_comparable(
        "unused-secondary",
        5_000,
        PriceBasis.ESTIMATED_NET,
        evidence=synthetic(),
    ).model_copy(update={"route": RouteOfAdministration.INTRAVENOUS})

    result = calculate_pricing_corridor(pricing_inputs(program, [primary, secondary]))

    assert result.decision_grade == DecisionGrade.DECISION_GRADE
    assert result.selected_annual_net_price == 22_000
    assert "comparable:unused-secondary:price" not in result.critical_evidence_status


def test_public_universal_coverage_does_not_require_income_default() -> None:
    result = calculate_pricing_corridor(
        pricing_inputs(
            make_program(income_bands=[], universal=True, cost_share=None, annual_oop=None),
            [make_comparable("net", 22_000, PriceBasis.ESTIMATED_NET)],
        )
    )

    assert result.decision_grade == DecisionGrade.DECISION_GRADE
    selected = next(item for item in result.access_estimates if item.annual_net_price == 22_000)
    assert selected.expected_patient_oop == 0
    assert selected.patient_oop_basis == PatientOOPBasis.ZERO_PATIENT_COST_SHARE
    assert selected.income_affordable_share == 1
    assert selected.known_affordable_lower_bound == 1
    assert "income_bands" not in result.critical_evidence_status


def test_public_coverage_with_cost_share_still_applies_income_affordability() -> None:
    income = [
        IncomeBandAssumptions(
            name="all",
            population_share=1,
            annual_income=20_000,
            maximum_oop_share=0.01,
            evidence=supported("public-plan-income"),
        )
    ]
    result = calculate_pricing_corridor(
        pricing_inputs(
            make_program(income_bands=income, universal=True, cost_share=0.1),
            [make_comparable("net", 22_000, PriceBasis.ESTIMATED_NET)],
        )
    )

    selected = next(item for item in result.access_estimates if item.annual_net_price == 22_000)
    assert selected.expected_patient_oop == 2_200
    assert selected.income_affordable_share == 0


def test_system_access_interpolates_and_holds_supplied_adoption_curve() -> None:
    access = AccessAssumptions(
        coverage_fraction=1,
        prior_authorization_pass_fraction=1,
        initiation_fraction=1,
        provider_capacity_fraction=1,
        adoption_by_year={0: 0.1, 2: 0.5},
    )

    assert access.system_access_fraction(0) == pytest.approx(0.1)
    assert access.system_access_fraction(1) == pytest.approx(0.3)
    assert access.system_access_fraction(2) == pytest.approx(0.5)
    assert access.system_access_fraction(3) == pytest.approx(0.5)


def test_selected_net_price_cannot_exceed_candidate_list_price() -> None:
    income = [
        IncomeBandAssumptions(
            name="all",
            population_share=1,
            annual_income=100_000,
            maximum_oop_share=0.1,
            evidence=supported("income"),
        )
    ]
    inputs = pricing_inputs(
        make_program(income_bands=income),
        [make_comparable("high-net", 50_000, PriceBasis.ESTIMATED_NET)],
    ).model_copy(update={"candidate_list_price": 18_000})

    result = calculate_pricing_corridor(inputs)

    corridor = result.annual_net_price_corridor
    assert corridor is not None
    assert corridor.feasible_upper == 18_000
    assert corridor.selected_annual_net_price == 18_000
    assert corridor.estimated_gross_to_net_fraction == 0
    assert not any(item.code == "NET_ABOVE_LIST" for item in result.warnings)


def test_unadjusted_selected_anchor_price_year_blocks_decision_grade() -> None:
    income = [
        IncomeBandAssumptions(
            name="all",
            population_share=1,
            annual_income=100_000,
            maximum_oop_share=0.1,
            evidence=supported("income"),
        )
    ]
    result = calculate_pricing_corridor(
        pricing_inputs(
            make_program(income_bands=income),
            [make_comparable("stale-net", 22_000, PriceBasis.ESTIMATED_NET, price_year=2025)],
        )
    )

    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.critical_evidence_status["selected_net_anchor_price_year"] is False
    assert any(
        item.code == "UNADJUSTED_PRICE_YEAR" and item.severity.value == "ERROR"
        for item in result.warnings
    )


def test_manufacturer_net_oop_proxy_is_explicit_and_not_decision_grade() -> None:
    income = [
        IncomeBandAssumptions(
            name="all",
            population_share=1,
            annual_income=100_000,
            maximum_oop_share=0.1,
            evidence=supported("income"),
        )
    ]
    result = calculate_pricing_corridor(
        pricing_inputs(
            make_program(income_bands=income, annual_oop=None),
            [make_comparable("net", 22_000, PriceBasis.ESTIMATED_NET)],
        )
    )

    selected = next(item for item in result.access_estimates if item.annual_net_price == 22_000)
    assert selected.expected_patient_oop == 2_200
    assert selected.patient_oop_basis == PatientOOPBasis.MANUFACTURER_NET_PROXY
    assert result.critical_evidence_status["access.annual_patient_oop"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
