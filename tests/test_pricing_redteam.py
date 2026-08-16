import json
from pathlib import Path

import pytest

from labrador_roi.comparables import ComparableSet, ComparableTier, select_comparables
from labrador_roi.models import PriceBasis, ProgramInput
from labrador_roi.pricing import PricingInputs, calculate_pricing_corridor

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _inputs() -> tuple[ProgramInput, ComparableSet]:
    program = ProgramInput.model_validate_json((FIXTURES / "demo_program.json").read_text())
    raw = json.loads((FIXTURES / "demo_comparables.json").read_text())
    return program, ComparableSet.model_validate({"comparables": raw["comparables"]})


def test_lifetime_qaly_budget_is_annualized_over_treated_years() -> None:
    """Regression for the Claude prototype's lifetime-value-as-annual-price defect."""

    program, comparables = _inputs()
    total_discounted_qaly_budget = 150_000 * 0.2594404
    discounted_treatment_years = 4.7171
    result = calculate_pricing_corridor(
        PricingInputs(
            program=program,
            comparables=comparables,
            incremental_qalys=0.2594404,
            willingness_to_pay_per_qaly=150_000,
            comparator_total_cost=0,
            new_non_drug_total_cost=0,
            expected_treatment_years=discounted_treatment_years,
            annual_manufacturer_cost=100,
            required_gross_margin_fraction=0.5,
            selected_net_anchor_basis=PriceBasis.ESTIMATED_NET,
        )
    )

    assert result.annual_net_price_corridor is not None
    expected_annual_ceiling = total_discounted_qaly_budget / discounted_treatment_years
    assert result.annual_net_price_corridor.value_based_ceiling == pytest.approx(
        expected_annual_ceiling
    )
    assert result.selected_annual_net_price <= expected_annual_ceiling
    assert result.annual_net_price_corridor.value_based_ceiling < total_discounted_qaly_budget


def test_unmatched_indication_cannot_reuse_an_ra_like_price_anchor() -> None:
    """A target/mechanism match cannot compensate for a failed indication/PICO gate."""

    program, comparables = _inputs()
    unrelated = program.initial_indication.model_copy(
        update={
            "name": "Toenail fungus",
            "therapeutic_area": "Dermatology",
            "target_population": "Adults with distal subungual onychomycosis",
            "line_of_therapy": "First line",
        }
    )
    program = program.model_copy(update={"initial_indication": unrelated})

    selection = select_comparables(program, comparables)

    assert selection.anchor_candidates() == []
    assert all(not item.usable_for_same_basis_anchor for item in selection.assessments)


def test_explicit_comparator_allowlist_excludes_unreviewed_catalog_records() -> None:
    program, comparables = _inputs()
    selected_id = "syn-comp-net-anchor"
    indication = program.initial_indication.model_copy(update={"comparator_ids": [selected_id]})
    program = program.model_copy(update={"initial_indication": indication})

    selection = select_comparables(program, comparables)

    selected = next(
        item for item in selection.assessments if item.comparable.comparable_id == selected_id
    )
    excluded = [
        item for item in selection.assessments if item.comparable.comparable_id != selected_id
    ]
    assert selected.usable_for_same_basis_anchor
    assert all(item.tier == ComparableTier.EXCLUDED for item in excluded)
    assert all(any("allowlist" in reason for reason in item.reasons) for item in excluded)
