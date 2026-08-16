from __future__ import annotations

import pytest
from pydantic import ValidationError

from labrador_roi.cashflow import (
    DevelopmentStage,
    ExpansionAssumptions,
    IndicationCommercialAssumptions,
    PatentAssumptions,
    ProgramCashFlowInputs,
    calculate_cashflow,
    delay_launch,
)


def _indication(
    indication_id: str,
    launch_year: int,
    *,
    gross_to_net_rate: float = 0.2,
    annual_gross_price: float = 20_000,
) -> IndicationCommercialAssumptions:
    return IndicationCommercialAssumptions(
        indication_id=indication_id,
        launch_year=launch_year,
        route="ORAL",
        backlog_patients=10_000,
        backlog_release_years=2,
        annual_incident_patients=1_000,
        coverage_rate=0.8,
        authorization_rate=0.8,
        patient_affordability_rate=1.0,
        initiation_rate=0.75,
        provider_capacity_rate=0.95,
        peak_adoption_rate=0.4,
        adoption_ramp_years=3,
        annual_persistence_rate=0.75,
        dose_intensity=0.9,
        annual_gross_price=annual_gross_price,
        gross_to_net_rate=gross_to_net_rate,
        cogs_per_full_dose_patient=1_500,
        variable_commercial_cost_per_patient=200,
        fixed_commercial_cost_per_year=1_000_000,
    )


def program_inputs(
    *,
    gross_to_net_rate: float = 0.2,
    expansion: bool = True,
) -> ProgramCashFlowInputs:
    expansion_inputs = None
    if expansion:
        expansion_inputs = ExpansionAssumptions(
            indication=_indication("expansion", 2034, annual_gross_price=16_000),
            development_stages=(
                DevelopmentStage(
                    name="expansion_phase_3",
                    year=2032,
                    cost=50_000_000,
                    success_probability=0.7,
                ),
            ),
            population_overlap_rate=0.15,
            initial_indication_cannibalization_rate=0.1,
            franchise_price_spillover_rate=0.12,
            shared_commercial_cost_savings_rate=0.25,
        )
    return ProgramCashFlowInputs(
        program_id="program-1",
        valuation_year=2026,
        forecast_end_year=2045,
        patent=PatentAssumptions(filing_year=2020),
        initial_indication=_indication("initial", 2030, gross_to_net_rate=gross_to_net_rate),
        initial_development_stages=(
            DevelopmentStage(name="phase_2", year=2027, cost=30_000_000, success_probability=0.6),
            DevelopmentStage(name="phase_3", year=2029, cost=100_000_000, success_probability=0.65),
        ),
        expansion=expansion_inputs,
        critical_inputs_supported=True,
        evidence_references=("synthetic-test",),
    )


def test_no_prelaunch_revenue() -> None:
    result = calculate_cashflow(program_inputs())

    assert all(row.net_revenue == 0 for row in result.annual_cash_flows if row.year < 2030)


def test_no_protected_revenue_after_patent_expiry() -> None:
    result = calculate_cashflow(program_inputs())

    assert result.patent_expiry_year == 2040
    assert all(
        row.protected_net_revenue == 0
        for row in result.annual_cash_flows
        if row.year >= result.patent_expiry_year
    )
    assert any(
        row.post_loe_net_revenue > 0
        for row in result.annual_cash_flows
        if row.year >= result.patent_expiry_year
    )


def test_higher_gross_to_net_lowers_value() -> None:
    lower_gtn = calculate_cashflow(program_inputs(gross_to_net_rate=0.1))
    higher_gtn = calculate_cashflow(program_inputs(gross_to_net_rate=0.5))

    assert higher_gtn.npv < lower_gtn.npv


def test_lower_patient_affordability_lowers_starts_and_revenue() -> None:
    inputs = program_inputs(expansion=False)
    lower_affordability = inputs.model_copy(
        update={
            "initial_indication": inputs.initial_indication.model_copy(
                update={"patient_affordability_rate": 0.4}
            )
        }
    )
    full = calculate_cashflow(inputs)
    constrained = calculate_cashflow(lower_affordability)

    assert sum(row.initial_new_starts for row in constrained.annual_cash_flows) < sum(
        row.initial_new_starts for row in full.annual_cash_flows
    )
    assert constrained.value_decomposition.gross_revenue < full.value_decomposition.gross_revenue


def test_launch_delay_cannot_add_protected_years() -> None:
    inputs = program_inputs()
    base = calculate_cashflow(inputs)
    delayed = calculate_cashflow(delay_launch(inputs))

    assert delayed.effective_protected_years == base.effective_protected_years - 1
    assert delayed.patent_expiry_year == base.patent_expiry_year


def test_label_expansion_does_not_reset_patent_clock() -> None:
    with_expansion = calculate_cashflow(program_inputs(expansion=True))
    without_expansion = calculate_cashflow(program_inputs(expansion=False))

    assert with_expansion.patent_expiry_year == without_expansion.patent_expiry_year == 2040
    assert with_expansion.expansion_effective_protected_years == 6
    assert all(
        row.protected_net_revenue == 0
        for row in with_expansion.annual_cash_flows
        if row.year >= 2040
    )


def test_patent_extension_is_capped_at_five_years() -> None:
    with pytest.raises(ValidationError):
        PatentAssumptions(filing_year=2020, extension_years=5.1)


def test_expansion_is_joint_and_price_spillover_reduces_initial_value() -> None:
    inputs = program_inputs(expansion=True)
    assert inputs.expansion is not None
    no_spillover = inputs.model_copy(
        update={
            "expansion": inputs.expansion.model_copy(
                update={
                    "franchise_price_spillover_rate": 0.0,
                    "initial_indication_cannibalization_rate": 0.0,
                }
            )
        }
    )
    base = calculate_cashflow(inputs)
    cleaner = calculate_cashflow(no_spillover)

    assert base.npv < cleaner.npv


def test_failed_expansion_does_not_cannibalize_or_spill_over_initial_value() -> None:
    inputs = program_inputs(expansion=True)
    assert inputs.expansion is not None
    failed_stage = inputs.expansion.development_stages[0].model_copy(
        update={"success_probability": 0.0, "cost": 0.0}
    )
    failed_expansion = inputs.expansion.model_copy(
        update={
            "development_stages": (failed_stage,),
            "initial_indication_cannibalization_rate": 0.5,
            "franchise_price_spillover_rate": 0.5,
        }
    )
    no_commercial_effect = failed_expansion.model_copy(
        update={
            "initial_indication_cannibalization_rate": 0.0,
            "franchise_price_spillover_rate": 0.0,
        }
    )

    failed = calculate_cashflow(
        inputs.model_copy(update={"expansion": failed_expansion}),
        calculate_delay_cost=False,
    )
    control = calculate_cashflow(
        inputs.model_copy(update={"expansion": no_commercial_effect}),
        calculate_delay_cost=False,
    )

    assert failed.expansion_approval_probability == 0
    assert max(row.expansion_active_patients for row in failed.annual_cash_flows) == 0
    assert failed.npv == pytest.approx(control.npv)
    assert failed.annual_cash_flows == control.annual_cash_flows


def test_supplied_adoption_curve_is_used_and_held_after_its_last_year() -> None:
    indication = IndicationCommercialAssumptions(
        indication_id="curve",
        launch_year=2030,
        backlog_patients=0,
        annual_incident_patients=100,
        coverage_rate=1,
        authorization_rate=1,
        patient_affordability_rate=1,
        initiation_rate=1,
        provider_capacity_rate=1,
        adoption_by_year={0: 0.1, 1: 0.1, 2: 0.9},
        annual_persistence_rate=0,
        dose_intensity=1,
        annual_gross_price=1,
    )
    inputs = ProgramCashFlowInputs(
        program_id="curve",
        valuation_year=2030,
        forecast_end_year=2033,
        patent=PatentAssumptions(filing_year=2025),
        initial_indication=indication,
    )

    result = calculate_cashflow(inputs, calculate_delay_cost=False)

    assert [row.initial_new_starts for row in result.annual_cash_flows] == pytest.approx(
        [10, 19, 243.9, 114.39]
    )


def test_regulatory_exclusivity_can_extend_the_shared_protected_window() -> None:
    inputs = program_inputs(expansion=False)
    inputs = inputs.model_copy(
        update={
            "patent": inputs.patent.model_copy(update={"regulatory_exclusivity_end_year": 2043})
        }
    )

    result = calculate_cashflow(inputs, calculate_delay_cost=False)

    assert result.patent_expiry_year == 2040
    assert result.effective_exclusivity_end_year == 2043
    assert result.effective_protected_years == 13
    assert all(row.protected for row in result.annual_cash_flows if row.year < 2043)
    assert all(not row.protected for row in result.annual_cash_flows if row.year >= 2043)


def test_unstarted_backlog_and_incident_cohorts_carry_forward_once() -> None:
    indication = IndicationCommercialAssumptions(
        indication_id="carry",
        launch_year=2030,
        backlog_patients=100,
        annual_incident_patients=10,
        coverage_rate=1,
        authorization_rate=1,
        patient_affordability_rate=1,
        initiation_rate=1,
        provider_capacity_rate=1,
        adoption_by_year={0: 0.25, 1: 1.0},
        annual_persistence_rate=0,
        dose_intensity=1,
        annual_gross_price=1,
    )
    inputs = ProgramCashFlowInputs(
        program_id="carry",
        valuation_year=2030,
        forecast_end_year=2031,
        tax_rate=0,
        patent=PatentAssumptions(filing_year=2025),
        initial_indication=indication,
    )

    result = calculate_cashflow(inputs, calculate_delay_cost=False)

    starts = [row.initial_new_starts for row in result.annual_cash_flows]
    assert starts == pytest.approx([27.5, 92.5])
    assert sum(starts) == pytest.approx(100 + 2 * 10)


def test_fractional_exclusivity_prorates_transition_year_revenue() -> None:
    indication = IndicationCommercialAssumptions(
        indication_id="fractional",
        launch_year=2030,
        backlog_patients=100,
        annual_incident_patients=0,
        coverage_rate=1,
        authorization_rate=1,
        patient_affordability_rate=1,
        initiation_rate=1,
        provider_capacity_rate=1,
        adoption_by_year={0: 1},
        annual_persistence_rate=1,
        dose_intensity=1,
        annual_gross_price=100,
    )
    inputs = ProgramCashFlowInputs(
        program_id="fractional",
        valuation_year=2030,
        forecast_end_year=2041,
        tax_rate=0,
        patent=PatentAssumptions(filing_year=2020, extension_years=0.5),
        initial_indication=indication,
        loe_price_retention=(0.5, 0.25),
        loe_volume_retention=(1.0, 1.0),
    )

    result = calculate_cashflow(inputs, calculate_delay_cost=False)
    transition = next(row for row in result.annual_cash_flows if row.year == 2040)
    following = next(row for row in result.annual_cash_flows if row.year == 2041)

    assert transition.protected_fraction == pytest.approx(0.5)
    assert transition.protected is False
    assert transition.protected_net_revenue == pytest.approx(5_000)
    assert transition.post_loe_net_revenue == pytest.approx(2_500)
    assert transition.net_revenue == pytest.approx(7_500)
    assert following.net_revenue == pytest.approx(3_750)


def test_value_decomposition_reconciles_to_npv() -> None:
    result = calculate_cashflow(program_inputs())

    decomposed_npv = (
        result.value_decomposition.initial_indication_discounted_fcf
        + result.value_decomposition.expansion_increment_discounted_fcf
    )
    assert decomposed_npv == pytest.approx(result.npv)
