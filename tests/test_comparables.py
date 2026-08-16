import pytest
from pydantic import ValidationError

from labrador_roi.comparables import (
    ComparableSet,
    ComparableTier,
    select_comparables,
)
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


def supported(source_id: str = "test-source") -> EvidenceMetadata:
    return EvidenceMetadata(
        source_id=source_id,
        evidence_type=EvidenceType.PAYER_OR_HTA,
        grade=EvidenceGrade.MODERATE,
    )


def program() -> ProgramInput:
    source = supported()
    access = AccessAssumptions(
        universal_or_public_coverage=True,
        coverage_fraction=0.8,
        prior_authorization_pass_fraction=0.7,
        initiation_fraction=0.6,
        provider_capacity_fraction=1.0,
        evidence={
            "coverage_fraction": source,
            "prior_authorization_pass_fraction": source,
            "initiation_fraction": source,
            "provider_capacity_fraction": source,
        },
    )
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
            eligible_patients=100_000,
            evidence={"eligible_patients": source},
        ),
        access=access,
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


def comparable(
    comparable_id: str,
    *,
    indication: str = "Plaque psoriasis",
    line: str = "After systemic therapy failure",
    geography: str = "United States",
    route: RouteOfAdministration = RouteOfAdministration.ORAL,
    target_population: str = "Adults with moderate to severe disease",
    therapeutic_area: str = "Inflammatory dermatology",
) -> ComparableTherapy:
    return ComparableTherapy(
        comparable_id=comparable_id,
        name=comparable_id,
        therapeutic_area=therapeutic_area,
        indication=indication,
        target_population=target_population,
        line_of_therapy=line,
        geography=geography,
        route=route,
        target_or_mechanism="TYK2",
        price=PriceObservation(
            amount=25_000,
            currency="USD",
            basis=PriceBasis.ESTIMATED_NET,
            price_year=2026,
            evidence=supported(f"price-{comparable_id}"),
        ),
    )


def test_noncompensatory_tiers_keep_pico_and_route_visible() -> None:
    catalog = ComparableSet(
        comparables=[
            comparable("exact"),
            comparable("route-mismatch", route=RouteOfAdministration.INTRAVENOUS),
            comparable(
                "target-only",
                indication="Ulcerative colitis",
                line="First line",
                geography="United Kingdom",
                target_population="Adults",
                therapeutic_area="Gastroenterology",
            ),
        ]
    )

    selection = select_comparables(program(), catalog)

    assert [item.tier for item in selection.assessments] == [
        ComparableTier.PRIMARY,
        ComparableTier.SECONDARY,
        ComparableTier.CONTEXT,
    ]
    assert selection.assessments[1].components.route == 0
    assert "route is not an exact match" in selection.assessments[1].reasons
    assert selection.assessments[2].components.target_or_mechanism == 1
    assert not selection.assessments[2].usable_for_same_basis_anchor


def test_currency_mismatch_is_retained_but_not_anchor_eligible() -> None:
    candidate = comparable("wrong-currency")
    candidate = candidate.model_copy(
        update={
            "price": candidate.price.model_copy(update={"currency": "GBP"}),
        }
    )

    selection = select_comparables(program(), ComparableSet(comparables=[candidate]))

    assessment = selection.assessments[0]
    assert assessment.tier == ComparableTier.PRIMARY
    assert not assessment.usable_for_same_basis_anchor
    assert any("no FX conversion" in reason for reason in assessment.reasons)


def test_duplicate_comparable_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="comparable_id values must be unique"):
        ComparableSet(comparables=[comparable("same"), comparable("same")])


def test_duplicate_product_price_fingerprints_are_rejected() -> None:
    first = comparable("first").model_copy(update={"name": "Same product"})
    second = comparable("second").model_copy(update={"name": "same PRODUCT"})

    with pytest.raises(ValidationError, match="duplicate comparable product/price"):
        ComparableSet(comparables=[first, second])

    with pytest.raises(ValidationError, match="duplicate comparable product/price"):
        select_comparables(program(), [first, second])
