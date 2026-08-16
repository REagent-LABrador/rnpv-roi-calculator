from __future__ import annotations

import pytest
from pydantic import ValidationError
from test_engine import _decision_ready_fixture

from labrador_roi.cashflow import (
    DevelopmentRealization,
    DevelopmentStage,
    ExpansionAssumptions,
    IndicationCommercialAssumptions,
    PatentAssumptions,
    ProgramCashFlowInputs,
    calculate_cashflow,
)
from labrador_roi.engine import ENGINE_VERSION, SCHEMA_VERSION, Recommendation, analyze_program
from labrador_roi.models import DecisionGrade, ProgramInput
from labrador_roi.provenance import sha256_digest
from labrador_roi.simulation import SimulationAssumptions, TriangularRange, simulate_program


def _point(value: float) -> TriangularRange:
    return TriangularRange(low=value, mode=value, high=value)


FIXED_ASSUMPTIONS = SimulationAssumptions(
    price_multiplier=_point(1),
    patient_multiplier=_point(1),
    gross_to_net_shift=_point(0),
    persistence_multiplier=_point(1),
    development_cost_multiplier=_point(1),
    launch_delay_years=_point(0),
    loe_retention_multiplier=_point(1),
)


def _indication(
    indication_id: str,
    launch_year: int,
    *,
    backlog_patients: float = 100,
    annual_persistence_rate: float = 0.5,
    annual_gross_price: float = 100,
) -> IndicationCommercialAssumptions:
    return IndicationCommercialAssumptions(
        indication_id=indication_id,
        launch_year=launch_year,
        route="ORAL",
        backlog_patients=backlog_patients,
        annual_incident_patients=0,
        coverage_rate=1,
        authorization_rate=1,
        patient_affordability_rate=1,
        initiation_rate=1,
        provider_capacity_rate=1,
        adoption_by_year={0: 1},
        annual_persistence_rate=annual_persistence_rate,
        dose_intensity=1,
        annual_gross_price=annual_gross_price,
    )


def _program(
    *,
    launch_year: int = 2026,
    forecast_end_year: int = 2030,
    stages: tuple[DevelopmentStage, ...] = (),
    expansion: ExpansionAssumptions | None = None,
) -> ProgramCashFlowInputs:
    return ProgramCashFlowInputs(
        program_id="redteam-program",
        valuation_year=2026,
        forecast_end_year=forecast_end_year,
        discount_rate=0,
        tax_rate=0,
        patent=PatentAssumptions(filing_year=2020),
        initial_indication=_indication("initial", launch_year),
        initial_development_stages=stages,
        expansion=expansion,
        critical_inputs_supported=True,
    )


def test_monte_carlo_uses_pathwise_failure_costs_and_preserves_the_expected_mean() -> None:
    stages = (
        DevelopmentStage(name="phase_2", year=2026, cost=10, success_probability=0.25),
        DevelopmentStage(name="phase_3", year=2027, cost=90, success_probability=1),
    )
    inputs = _program(launch_year=2027, forecast_end_year=2027, stages=stages)
    inputs = inputs.model_copy(
        update={
            "initial_indication": inputs.initial_indication.model_copy(
                update={"backlog_patients": 0, "annual_gross_price": 0}
            )
        }
    )

    deterministic = calculate_cashflow(inputs, calculate_delay_cost=False)
    simulated = simulate_program(
        inputs,
        simulations=2_000,
        seed=91,
        assumptions=FIXED_ASSUMPTIONS,
    )

    assert deterministic.npv == pytest.approx(-(10 + 0.25 * 90))
    assert simulated.rnpv.p10 == pytest.approx(-100)
    assert simulated.rnpv.p50 == pytest.approx(-10)
    assert simulated.rnpv.p90 == pytest.approx(-10)
    assert simulated.rnpv.mean == pytest.approx(deterministic.npv, abs=3)


def test_uncertainty_recomputes_the_nonlinear_persistence_ledger_per_draw() -> None:
    inputs = _program()
    deterministic = calculate_cashflow(inputs, calculate_delay_cost=False)
    persistence_uncertainty = FIXED_ASSUMPTIONS.model_copy(
        update={"persistence_multiplier": TriangularRange(low=0.5, mode=1, high=1.5)}
    )

    simulated = simulate_program(
        inputs,
        simulations=2_000,
        seed=19,
        assumptions=persistence_uncertainty,
    )

    assert simulated.protected_net_revenue.p10 < simulated.protected_net_revenue.p90
    assert simulated.protected_net_revenue.mean > (
        deterministic.value_decomposition.protected_net_revenue * 1.01
    )
    assert simulated.peak_annual_net_revenue.p50 == pytest.approx(10_000)


def test_invalid_development_realizations_fail_closed() -> None:
    with pytest.raises(ValidationError, match="finite and non-negative"):
        DevelopmentRealization(
            initial_success=False,
            initial_development_costs_by_year={2026: -1},
        )

    expansion = ExpansionAssumptions(
        indication=_indication("expansion", 2028),
        conditional_on_initial_success=True,
    )
    inputs = _program(expansion=expansion)
    impossible = DevelopmentRealization(
        initial_success=False,
        expansion_success=True,
        expansion_development_costs_by_year={2027: 1},
    )

    with pytest.raises(ValueError, match="cannot proceed after the initial program fails"):
        calculate_cashflow(inputs, realization=impossible, calculate_delay_cost=False)


def test_expansion_overlap_and_launch_delay_use_one_shared_patent_clock() -> None:
    expansion = ExpansionAssumptions(
        indication=_indication("expansion", 2034),
        population_overlap_rate=0,
    )
    inputs = _program(launch_year=2030, forecast_end_year=2042, expansion=expansion)
    base = calculate_cashflow(inputs, calculate_delay_cost=False)

    fully_overlapping = inputs.model_copy(
        update={"expansion": expansion.model_copy(update={"population_overlap_rate": 1})}
    )
    overlap_result = calculate_cashflow(fully_overlapping, calculate_delay_cost=False)
    delayed_expansion = inputs.model_copy(
        update={
            "expansion": expansion.model_copy(
                update={"indication": expansion.indication.model_copy(update={"launch_year": 2036})}
            )
        }
    )
    delayed = calculate_cashflow(delayed_expansion, calculate_delay_cost=False)

    assert max(row.expansion_active_patients for row in base.annual_cash_flows) > 0
    assert max(row.expansion_active_patients for row in overlap_result.annual_cash_flows) == 0
    assert base.patent_expiry_year == delayed.patent_expiry_year == 2040
    assert base.expansion_effective_protected_years == 6
    assert delayed.expansion_effective_protected_years == 4
    assert (
        delayed.value_decomposition.protected_net_revenue
        < base.value_decomposition.protected_net_revenue
    )


def test_peak_revenue_is_derived_from_the_annual_ledger_and_exposed_with_rng_metadata() -> None:
    inputs = _program()
    result = analyze_program(
        inputs,
        simulations=20,
        seed=7,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )
    ledger_peak = max(result.cash_flow.annual_cash_flows, key=lambda row: row.net_revenue)

    assert result.schema_version == "1.3.0"
    assert result.engine_version == "0.4.0"
    assert result.cash_flow.peak_annual_net_revenue == ledger_peak.net_revenue
    assert result.cash_flow.peak_annual_net_revenue_year == ledger_peak.year
    assert result.summary.peak_annual_net_revenue == ledger_peak.net_revenue
    assert result.summary.peak_annual_net_revenue_year == ledger_peak.year
    assert result.summary.peak_annual_net_revenue_p50 == pytest.approx(ledger_peak.net_revenue)
    assert result.uncertainty.rng_bit_generator == "PCG64"
    assert result.uncertainty.numpy_version
    assert result.uncertainty.draw_order_contract_version == "1.0.0"
    assert "shared draw" in result.uncertainty.commercial_driver_correlation
    assert any(step.step_id == "peak_annual_net_revenue" for step in result.calculation_steps)

    expected_digest = sha256_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "inputs": {"cashflow_inputs": inputs},
            "seed": 7,
            "simulations": 20,
            "simulation_assumptions": FIXED_ASSUMPTIONS,
        }
    )
    assert result.input_digest == expected_digest


def test_unmodeled_third_indication_forces_not_decision_grade() -> None:
    program, comparables = _decision_ready_fixture()
    first_expansion = program.expansion_indications[0]
    second_expansion = first_expansion.model_copy(
        update={
            "indication_id": "second-expansion",
            "name": "Second expansion",
            "launch_year": first_expansion.launch_year + 1,
        }
    )
    program = program.model_copy(
        update={"expansion_indications": [first_expansion, second_expansion]}
    )

    result = analyze_program(program, comparables, simulations=2, seed=3)

    assert result.critical_evidence_status["cashflow.expansion_scope"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.recommendation == Recommendation.NOT_DECISION_GRADE
    assert any(
        warning.code == "EXPANSION_SCOPE_LIMIT" and warning.severity.value == "ERROR"
        for warning in result.warnings
    )


def test_missing_expansion_interaction_inputs_force_not_decision_grade() -> None:
    program, comparables = _decision_ready_fixture()
    expansion = program.expansion_indications[0]
    expansion = expansion.model_copy(
        update={
            "population": expansion.population.model_copy(
                update={
                    "overlap_with_initial_fraction": None,
                    "cannibalization_fraction": None,
                }
            )
        }
    )
    program = program.model_copy(update={"expansion_indications": [expansion]})

    result = analyze_program(program, comparables, simulations=2, seed=3)

    assert result.critical_evidence_status["cashflow.expansion.population_interaction"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.recommendation == Recommendation.NOT_DECISION_GRADE


def test_program_contract_rejects_impossible_patent_and_expansion_dates() -> None:
    program, _ = _decision_ready_fixture()
    payload = program.model_dump(mode="json")
    payload["patent"]["filing_year"] = program.initial_indication.launch_year + 1
    with pytest.raises(ValidationError, match="patent filing cannot occur after initial launch"):
        ProgramInput.model_validate(payload)

    payload = program.model_dump(mode="json")
    payload["expansion_indications"][0]["launch_year"] = program.initial_indication.launch_year - 1
    with pytest.raises(ValidationError, match="expansion launch cannot precede"):
        ProgramInput.model_validate(payload)
