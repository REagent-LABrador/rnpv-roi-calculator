from __future__ import annotations

import json
from copy import deepcopy
from math import prod
from pathlib import Path

import pytest
from test_cashflow import program_inputs

from labrador_roi.comparables import ComparableSet
from labrador_roi.engine import (
    ENGINE_VERSION,
    SCHEMA_VERSION,
    Recommendation,
    analyze_program,
)
from labrador_roi.models import (
    DecisionGrade,
    EvidenceGrade,
    EvidenceMetadata,
    EvidenceType,
    Modality,
    ProgramInput,
)
from labrador_roi.simulation import SimulationAssumptions, TriangularRange

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"

FIXED_ASSUMPTIONS = SimulationAssumptions(
    price_multiplier=TriangularRange(low=1, mode=1, high=1),
    patient_multiplier=TriangularRange(low=1, mode=1, high=1),
    gross_to_net_shift=TriangularRange(low=0, mode=0, high=0),
    persistence_multiplier=TriangularRange(low=1, mode=1, high=1),
    development_cost_multiplier=TriangularRange(low=1, mode=1, high=1),
    launch_delay_years=TriangularRange(low=0, mode=0, high=0),
    loe_retention_multiplier=TriangularRange(low=1, mode=1, high=1),
)


def _evidence_payload(
    source_id: str = "test-source",
    *,
    evidence_type: str = "PAYER_OR_HTA",
    grade: str = "MODERATE",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "evidence_type": evidence_type,
        "grade": grade,
        "synthetic": False,
    }


def _replace_fixture_evidence(
    value,
    *,
    evidence_type: str = "PAYER_OR_HTA",
    grade: str = "MODERATE",
):
    if isinstance(value, dict):
        if {"evidence_type", "grade", "synthetic"} <= value.keys():
            return _evidence_payload(
                str(value.get("source_id") or "test-source"),
                evidence_type=evidence_type,
                grade=grade,
            )
        return {
            key: _replace_fixture_evidence(item, evidence_type=evidence_type, grade=grade)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_fixture_evidence(item, evidence_type=evidence_type, grade=grade)
            for item in value
        ]
    return value


def _decision_ready_fixture(
    *,
    evidence_type: str = "PAYER_OR_HTA",
    grade: str = "MODERATE",
) -> tuple[ProgramInput, ComparableSet]:
    raw_program = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    raw_program = _replace_fixture_evidence(raw_program, evidence_type=evidence_type, grade=grade)
    group = _evidence_payload(
        evidence_type=evidence_type,
        grade=grade,
    )
    raw_program["patent"]["evidence"]["patent_inputs"] = group
    raw_program["development"]["program_probability_of_approval"] = prod(
        raw_program["development"]["stage_success_probabilities"].values()
    )
    raw_program["expansion_indications"][0]["access"]["patient_cost_share_fraction"] = 0
    for indication in [
        raw_program["initial_indication"],
        *raw_program["expansion_indications"],
    ]:
        indication["population"]["evidence"]["population_inputs"] = group
        indication["evidence"]["commercial_inputs"] = group
        indication["evidence"]["candidate_list_price"] = group
        if indication["access"].get("patient_cost_share_fraction", 0) > 0:
            indication["access"]["annual_patient_oop"] = 2_000
            indication["access"]["evidence"]["annual_patient_oop"] = group
    raw_comparables = json.loads((FIXTURE_ROOT / "demo_comparables.json").read_text())
    raw_comparables = _replace_fixture_evidence(
        raw_comparables, evidence_type=evidence_type, grade=grade
    )
    return (
        ProgramInput.model_validate(raw_program),
        ComparableSet.model_validate({"comparables": raw_comparables["comparables"]}),
    )


def test_seeded_analysis_is_reproducible() -> None:
    first = analyze_program(
        program_inputs(), simulations=200, seed=77, simulation_assumptions=FIXED_ASSUMPTIONS
    )
    second = analyze_program(
        program_inputs(), simulations=200, seed=77, simulation_assumptions=FIXED_ASSUMPTIONS
    )

    assert first.input_digest == second.input_digest
    assert first.run_id == second.run_id
    assert first.uncertainty == second.uncertainty
    assert first.summary == second.summary


def test_analysis_has_stable_audit_contract_and_decomposition() -> None:
    result = analyze_program(
        program_inputs(), simulations=100, seed=1, simulation_assumptions=FIXED_ASSUMPTIONS
    )

    assert result.schema_version == SCHEMA_VERSION
    assert result.engine_version == ENGINE_VERSION
    assert result.run_id.startswith("run_")
    assert result.input_digest.startswith("sha256:")
    assert result.seed == 1
    assert result.simulations == 100
    assert result.simulation_assumptions == FIXED_ASSUMPTIONS
    assert result.input_snapshot["cashflow_inputs"]["program_id"] == "program-1"
    assert result.calculation_steps
    assert result.value_decomposition.protected_net_revenue > 0
    assert result.summary.value_lost_per_launch_delay_year != 0
    assert result.critical_evidence_status == {"cashflow_inputs": True}


def test_standardized_recommendation_and_decision_grade() -> None:
    result = analyze_program(
        program_inputs(), simulations=300, seed=4, simulation_assumptions=FIXED_ASSUMPTIONS
    )

    assert result.decision_grade == DecisionGrade.DECISION_GRADE
    assert result.recommendation in set(Recommendation)
    assert result.recommendation != Recommendation.NOT_DECISION_GRADE


def test_unsupported_inputs_never_receive_decision_grade() -> None:
    unsupported = program_inputs().model_copy(update={"critical_inputs_supported": False})
    result = analyze_program(
        unsupported, simulations=50, seed=2, simulation_assumptions=FIXED_ASSUMPTIONS
    )

    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.recommendation == Recommendation.NOT_DECISION_GRADE


def test_result_contains_no_secret_from_evidence_reference() -> None:
    secret_url = "https://example.test/data?api_" + "key=secret-value"
    inputs = program_inputs().model_copy(update={"evidence_references": (secret_url,)})
    result = analyze_program(
        inputs, simulations=10, seed=5, simulation_assumptions=FIXED_ASSUMPTIONS
    )

    rendered = result.model_dump_json()
    assert "secret-value" not in rendered
    assert result.input_snapshot["cashflow_inputs"]["evidence_references"] == [
        "https://example.test/data?api_" + "key=%5BREDACTED%5D"
    ]


def test_zero_oop_public_coverage_maps_to_full_affordability() -> None:
    program = ProgramInput.model_validate_json((FIXTURE_ROOT / "demo_program.json").read_text())
    public_access = program.initial_indication.access.model_copy(
        update={"universal_or_public_coverage": True, "patient_cost_share_fraction": 0.0}
    )
    program = program.model_copy(
        update={
            "initial_indication": program.initial_indication.model_copy(
                update={"access": public_access}
            )
        }
    )
    comparable_payload = json.loads((FIXTURE_ROOT / "demo_comparables.json").read_text())
    comparables = ComparableSet.model_validate({"comparables": comparable_payload["comparables"]})

    result = analyze_program(program, comparables, simulations=5, seed=3)

    assert result.access[0].patient_affordability_rate == 1.0


@pytest.mark.parametrize("fixture_name", ["demo_program.json", "demo_program_b.json"])
def test_demo_fixture_approval_probability_reconciles_with_stage_path(
    fixture_name: str,
) -> None:
    program = ProgramInput.model_validate_json((FIXTURE_ROOT / fixture_name).read_text())
    comparable_payload = json.loads((FIXTURE_ROOT / "demo_comparables.json").read_text())
    comparables = ComparableSet.model_validate({"comparables": comparable_payload["comparables"]})

    stage_product = prod(program.development.stage_success_probabilities.values())
    assert program.development.program_probability_of_approval == pytest.approx(
        stage_product,
        rel=1e-6,
        abs=1e-9,
    )

    result = analyze_program(
        program,
        comparables,
        simulations=2,
        seed=1,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert result.critical_evidence_status["cashflow.development_probability_reconciliation"]
    assert not any(
        warning.code == "INCONSISTENT_PROGRAM_APPROVAL_PROBABILITY" for warning in result.warnings
    )


@pytest.mark.parametrize(
    ("serialized_modality", "expected"),
    [
        ("SMALL_MOLECULE", Modality.SMALL_MOLECULE),
        ("PEPTIDE", Modality.PEPTIDE),
        ("ANTIBODY", Modality.ANTIBODY),
        ("antibody", Modality.ANTIBODY),
    ],
)
def test_program_input_supports_antibody_and_existing_modalities(
    serialized_modality: str,
    expected: Modality,
) -> None:
    raw_program = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    raw_program["modality"] = serialized_modality

    program = ProgramInput.model_validate(raw_program)

    assert program.modality is expected


def test_antibody_program_runs_without_hidden_modality_adjustments() -> None:
    raw_program = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    antibody_payload = {**raw_program, "modality": "antibody"}
    peptide = ProgramInput.model_validate(raw_program)
    antibody = ProgramInput.model_validate(antibody_payload)
    comparable_payload = json.loads((FIXTURE_ROOT / "demo_comparables.json").read_text())
    comparables = ComparableSet.model_validate({"comparables": comparable_payload["comparables"]})

    peptide_result = analyze_program(
        peptide,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )
    antibody_result = analyze_program(
        antibody,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert antibody_result.input_snapshot["program"]["modality"] == "ANTIBODY"
    assert antibody_result.cash_flow == peptide_result.cash_flow
    assert antibody_result.uncertainty == peptide_result.uncertainty


def test_antibody_loe_retention_requires_supported_evidence_for_decision_grade() -> None:
    program, comparables = _decision_ready_fixture()
    antibody = program.model_copy(update={"modality": Modality.ANTIBODY})

    unsupported = analyze_program(
        antibody,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert unsupported.critical_evidence_status["cashflow.antibody_loe_retention"] is False
    assert unsupported.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert any(
        warning.field == "cashflow.antibody_loe_retention" for warning in unsupported.warnings
    )

    loe_evidence = EvidenceMetadata(
        source_id="antibody-loe-source",
        evidence_type=EvidenceType.REAL_WORLD,
        grade=EvidenceGrade.MODERATE,
    )
    supported_antibody = antibody.model_copy(
        update={
            "assumptions": {
                **antibody.assumptions,
                "loe_price_retention": [0.55, 0.35, 0.25, 0.18, 0.12, 0.12],
                "loe_volume_retention": [0.75, 0.55, 0.4, 0.3, 0.2, 0.2],
            },
            "evidence": {**antibody.evidence, "loe_retention": loe_evidence},
        }
    )
    supported = analyze_program(
        supported_antibody,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert supported.critical_evidence_status["cashflow.antibody_loe_retention"] is True
    assert supported.decision_grade == DecisionGrade.DECISION_GRADE
    assert not any(
        warning.field == "cashflow.antibody_loe_retention" for warning in supported.warnings
    )


@pytest.mark.parametrize(
    ("mapping_name", "stage_name"),
    [
        ("stage_costs", "phase_2"),
        ("stage_durations_years", "phase_2"),
        ("stage_success_probabilities", "phase_2"),
    ],
)
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_development_inputs_are_rejected(
    mapping_name: str,
    stage_name: str,
    invalid_value: float,
) -> None:
    raw_program = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    raw_program["development"][mapping_name][stage_name] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        ProgramInput.model_validate(raw_program)


@pytest.mark.parametrize(
    ("mapping_name", "stage_name"),
    [
        ("stage_costs", "phase_2"),
        ("stage_durations_years", "phase_2"),
        ("stage_success_probabilities", "phase_2"),
    ],
)
def test_boolean_development_mapping_inputs_are_rejected(
    mapping_name: str,
    stage_name: str,
) -> None:
    raw_program = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    raw_program["development"][mapping_name][stage_name] = True

    with pytest.raises(ValueError, match="boolean"):
        ProgramInput.model_validate(raw_program)


def test_boolean_aggregate_approval_probability_is_rejected() -> None:
    raw_program = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    raw_program["development"]["program_probability_of_approval"] = True

    with pytest.raises(ValueError, match="boolean"):
        ProgramInput.model_validate(raw_program)


def test_antibody_requires_explicit_comparator_allowlists_for_decision_grade() -> None:
    program, comparables = _decision_ready_fixture()
    loe_evidence = EvidenceMetadata(
        source_id="antibody-loe-source",
        evidence_type=EvidenceType.REAL_WORLD,
        grade=EvidenceGrade.MODERATE,
    )
    initial = program.initial_indication.model_copy(update={"comparator_ids": []})
    expansions = [
        indication.model_copy(update={"comparator_ids": []})
        for indication in program.expansion_indications
    ]
    antibody = program.model_copy(
        update={
            "modality": Modality.ANTIBODY,
            "initial_indication": initial,
            "expansion_indications": expansions,
            "evidence": {**program.evidence, "loe_retention": loe_evidence},
        }
    )

    result = analyze_program(
        antibody,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert result.critical_evidence_status["cashflow.antibody_comparator_selection"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert any(
        warning.field == "cashflow.antibody_comparator_selection" for warning in result.warnings
    )


def test_antibody_comparator_allowlists_cannot_silently_reference_unknown_ids() -> None:
    program, comparables = _decision_ready_fixture()
    loe_evidence = EvidenceMetadata(
        source_id="antibody-loe-source",
        evidence_type=EvidenceType.REAL_WORLD,
        grade=EvidenceGrade.MODERATE,
    )
    initial = program.initial_indication.model_copy(
        update={
            "comparator_ids": [
                *program.initial_indication.comparator_ids,
                "DOES_NOT_EXIST",
            ]
        }
    )
    antibody = program.model_copy(
        update={
            "modality": Modality.ANTIBODY,
            "initial_indication": initial,
            "evidence": {**program.evidence, "loe_retention": loe_evidence},
        }
    )

    result = analyze_program(
        antibody,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert result.critical_evidence_status["cashflow.antibody_comparator_selection"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE


def test_analysis_boundary_revalidates_unchecked_modality_updates() -> None:
    program, comparables = _decision_ready_fixture()
    unchecked_antibody = program.model_copy(update={"modality": "antibody"})

    result = analyze_program(
        unchecked_antibody,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert result.input_snapshot["program"]["modality"] == "ANTIBODY"
    assert result.critical_evidence_status["cashflow.antibody_loe_retention"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE

    unchecked_biologic = program.model_copy(update={"modality": "BIOLOGIC"})
    with pytest.raises(ValueError, match=r"SMALL_MOLECULE.*PEPTIDE.*ANTIBODY"):
        analyze_program(
            unchecked_biologic,
            comparables,
            simulations=2,
            seed=3,
            simulation_assumptions=FIXED_ASSUMPTIONS,
        )


@pytest.mark.parametrize(
    ("price_retention", "volume_retention"),
    [
        ((), ()),
        ((2.0,), (1.0,)),
    ],
)
def test_analysis_boundary_revalidates_unchecked_low_level_cashflow_inputs(
    price_retention: tuple[float, ...],
    volume_retention: tuple[float, ...],
) -> None:
    unchecked = program_inputs().model_copy(
        update={
            "loe_price_retention": price_retention,
            "loe_volume_retention": volume_retention,
        }
    )

    with pytest.raises(ValueError):
        analyze_program(
            unchecked,
            simulations=2,
            seed=3,
            simulation_assumptions=FIXED_ASSUMPTIONS,
        )


def test_low_level_decision_grade_requires_evidence_references() -> None:
    unsupported = program_inputs().model_copy(update={"evidence_references": ()})

    result = analyze_program(
        unsupported,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert result.critical_evidence_status["cashflow_inputs"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.recommendation == Recommendation.NOT_DECISION_GRADE
    assert any(warning.code == "MISSING_LOW_LEVEL_EVIDENCE" for warning in result.warnings)


def test_empty_antibody_loe_paths_are_rejected() -> None:
    program, comparables = _decision_ready_fixture()
    loe_evidence = EvidenceMetadata(
        source_id="antibody-loe-source",
        evidence_type=EvidenceType.REAL_WORLD,
        grade=EvidenceGrade.MODERATE,
    )
    antibody = program.model_copy(
        update={
            "modality": Modality.ANTIBODY,
            "assumptions": {
                **program.assumptions,
                "loe_price_retention": [],
                "loe_volume_retention": [],
            },
            "evidence": {**program.evidence, "loe_retention": loe_evidence},
        }
    )

    with pytest.raises(ValueError, match="LOE retention paths cannot be empty"):
        analyze_program(
            antibody,
            comparables,
            simulations=2,
            seed=3,
            simulation_assumptions=FIXED_ASSUMPTIONS,
        )


def test_antibody_loe_paths_must_cover_the_modeled_post_loe_horizon() -> None:
    program, comparables = _decision_ready_fixture()
    loe_evidence = EvidenceMetadata(
        source_id="antibody-loe-source",
        evidence_type=EvidenceType.REAL_WORLD,
        grade=EvidenceGrade.MODERATE,
    )
    short_path = program.model_copy(
        update={
            "modality": Modality.ANTIBODY,
            "assumptions": {
                **program.assumptions,
                "loe_price_retention": [0.99],
                "loe_volume_retention": [0.99],
            },
            "evidence": {**program.evidence, "loe_retention": loe_evidence},
        }
    )

    result = analyze_program(
        short_path,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert result.critical_evidence_status["cashflow.antibody_loe_horizon"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert any(warning.field == "cashflow.antibody_loe_horizon" for warning in result.warnings)


def test_aggregate_only_approval_probability_cannot_clear_development_cost_gate() -> None:
    program, comparables = _decision_ready_fixture()
    development = program.development.model_copy(
        update={
            "stage_costs": {},
            "stage_durations_years": {},
            "stage_success_probabilities": {},
            "stage_order": [],
            "program_probability_of_approval": 1.0,
        }
    )
    aggregate_only = program.model_copy(
        update={"development": development, "expansion_indications": []}
    )

    result = analyze_program(
        aggregate_only,
        comparables,
        simulations=2,
        seed=3,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert result.cash_flow.initial_approval_probability == 1.0
    assert result.critical_evidence_status["cashflow.development_cost_path"] is False
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.recommendation == Recommendation.NOT_DECISION_GRADE


def test_current_stage_must_match_first_modeled_remaining_stage() -> None:
    raw_program = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    raw_program["development"]["current_stage"] = "filing"

    with pytest.raises(ValueError, match="current_stage must match the first modeled stage"):
        ProgramInput.model_validate(raw_program)


def test_explicit_known_stage_order_must_follow_lifecycle_order() -> None:
    raw_program = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    raw_program["development"]["stage_order"] = [
        "preclinical",
        "filing",
        "phase_3",
        "phase_2",
        "phase_1",
    ]

    with pytest.raises(ValueError, match="known lifecycle stages must be chronological"):
        ProgramInput.model_validate(raw_program)


def test_mapping_order_cannot_change_results_behind_the_same_run_id() -> None:
    raw = json.loads((FIXTURE_ROOT / "demo_program.json").read_text())
    reordered = deepcopy(raw)
    reordered["development"]["stage_costs"] = dict(
        reversed(list(reordered["development"]["stage_costs"].items()))
    )
    comparables_payload = json.loads((FIXTURE_ROOT / "demo_comparables.json").read_text())
    comparables = ComparableSet.model_validate({"comparables": comparables_payload["comparables"]})

    first = analyze_program(
        ProgramInput.model_validate(raw),
        comparables,
        simulations=25,
        seed=9,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )
    second = analyze_program(
        ProgramInput.model_validate(reordered),
        comparables,
        simulations=25,
        seed=9,
        simulation_assumptions=FIXED_ASSUMPTIONS,
    )

    assert first.input_digest == second.input_digest
    assert first.run_id == second.run_id
    assert first.cash_flow == second.cash_flow
    assert first.uncertainty == second.uncertainty


def test_complete_supported_fixture_can_clear_decision_gate() -> None:
    program, comparables = _decision_ready_fixture()

    result = analyze_program(program, comparables, simulations=5, seed=1)

    assert result.decision_grade == DecisionGrade.DECISION_GRADE
    assert all(result.critical_evidence_status.values())
    assert not any(item.severity.value == "ERROR" for item in result.warnings)
    assert result.input_snapshot["program"]["program_id"] == program.program_id
    assert len(result.input_snapshot["comparables"]["comparables"]) == len(comparables.comparables)


def test_missing_commercial_rates_block_decision_grade_instead_of_issuing_stop() -> None:
    program, comparables = _decision_ready_fixture()
    indication = program.initial_indication
    assumptions = dict(indication.assumptions)
    for key in (
        "peak_adoption_rate",
        "adoption_ramp_years",
        "annual_persistence_rate",
        "dose_intensity",
    ):
        assumptions.pop(key, None)
    indication = indication.model_copy(
        update={
            "access": indication.access.model_copy(update={"adoption_by_year": {}}),
            "assumptions": assumptions,
        }
    )
    program = program.model_copy(
        update={"initial_indication": indication, "expansion_indications": []}
    )

    result = analyze_program(program, comparables, simulations=10, seed=1)

    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.recommendation == Recommendation.NOT_DECISION_GRADE
    error_codes = {item.code for item in result.warnings if item.severity.value == "ERROR"}
    assert {"MISSING_ADOPTION", "MISSING_PERSISTENCE", "MISSING_DOSE_INTENSITY"} <= error_codes


def test_missing_expansion_development_path_blocks_decision_grade() -> None:
    program, comparables = _decision_ready_fixture()
    program = program.model_copy(
        update={"development": program.development.model_copy(update={"assumptions": {}})}
    )

    result = analyze_program(program, comparables, simulations=10, seed=1)

    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.recommendation == Recommendation.NOT_DECISION_GRADE
    assert result.critical_evidence_status["cashflow.expansion.development_path"] is False
    assert any(item.code == "MISSING_EXPANSION_DEVELOPMENT_PATH" for item in result.warnings)


def test_missing_initial_development_probability_blocks_decision_grade() -> None:
    program, comparables = _decision_ready_fixture()
    development = program.development.model_copy(
        update={
            "stage_costs": {},
            "stage_durations_years": {},
            "stage_success_probabilities": {},
            "stage_order": [],
            "program_probability_of_approval": None,
        }
    )
    program = program.model_copy(update={"development": development, "expansion_indications": []})

    result = analyze_program(program, comparables, simulations=5, seed=1)

    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.critical_evidence_status["cashflow.development_path"] is False
    assert any(item.code == "MISSING_INITIAL_DEVELOPMENT_PATH" for item in result.warnings)


def test_diagnosis_and_clinical_eligibility_reduce_stock_and_incident_flow() -> None:
    program, comparables = _decision_ready_fixture()

    def with_fractions(diagnosed: float, clinically_eligible: float) -> ProgramInput:
        population = program.initial_indication.population.model_copy(
            update={
                "diagnosed_fraction": diagnosed,
                "clinically_eligible_fraction": clinically_eligible,
            }
        )
        indication = program.initial_indication.model_copy(update={"population": population})
        return program.model_copy(
            update={"initial_indication": indication, "expansion_indications": []}
        )

    excluded = analyze_program(with_fractions(0, 0), comparables, simulations=5, seed=1)
    eligible = analyze_program(with_fractions(1, 1), comparables, simulations=5, seed=1)

    assert sum(row.initial_new_starts for row in excluded.cash_flow.annual_cash_flows) == 0
    assert sum(row.initial_new_starts for row in eligible.cash_flow.annual_cash_flows) > 0
    assert excluded.value_decomposition.gross_revenue == 0
    assert eligible.value_decomposition.gross_revenue > 0


def test_negative_monte_carlo_cost_multiplier_is_rejected() -> None:
    with pytest.raises(ValueError, match="development_cost_multiplier"):
        SimulationAssumptions(development_cost_multiplier=TriangularRange(low=-1, mode=-1, high=-1))


def test_low_grade_unsupported_evidence_cannot_clear_decision_gate() -> None:
    metadata = EvidenceMetadata(
        source_id="placeholder",
        evidence_type=EvidenceType.UNSUPPORTED,
        grade=EvidenceGrade.LOW,
    )
    assert metadata.supports_decision is False

    program, comparables = _decision_ready_fixture(
        evidence_type="UNSUPPORTED",
        grade="LOW",
    )
    result = analyze_program(program, comparables, simulations=5, seed=1)

    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert result.recommendation == Recommendation.NOT_DECISION_GRADE


def test_inconsistent_aggregate_approval_probability_blocks_decision_grade() -> None:
    program, comparables = _decision_ready_fixture()
    program = program.model_copy(
        update={
            "development": program.development.model_copy(
                update={"program_probability_of_approval": 0.5}
            )
        }
    )

    result = analyze_program(program, comparables, simulations=5, seed=1)

    stage_product = prod(program.development.stage_success_probabilities.values())
    assert result.cash_flow.initial_approval_probability == pytest.approx(stage_product)
    assert (
        result.critical_evidence_status["cashflow.development_probability_reconciliation"] is False
    )
    assert result.decision_grade == DecisionGrade.NOT_DECISION_GRADE
    assert any(
        item.code == "INCONSISTENT_PROGRAM_APPROVAL_PROBABILITY" and item.severity.value == "ERROR"
        for item in result.warnings
    )


def test_standard_input_snapshot_redacts_credentials_but_preserves_access_inputs() -> None:
    program, comparables = _decision_ready_fixture()
    program = program.model_copy(
        update={
            "assumptions": {
                **program.assumptions,
                "authorization_header": "Bearer actual-credential-value",
            }
        }
    )

    result = analyze_program(program, comparables, simulations=2, seed=1)

    snapshot = result.input_snapshot["program"]
    assert snapshot["assumptions"]["authorization_header"] == "[REDACTED]"
    assert (
        snapshot["initial_indication"]["access"]["prior_authorization_pass_fraction"]
        == program.initial_indication.access.prior_authorization_pass_fraction
    )


def test_candidate_list_and_selected_net_reconcile_in_cash_revenue() -> None:
    program, comparables = _decision_ready_fixture()
    program = program.model_copy(update={"expansion_indications": []})

    result = analyze_program(program, comparables, simulations=2, seed=1)

    corridor = result.pricing[0].annual_net_price_corridor
    assert corridor is not None
    assert corridor.candidate_list_price is not None
    assert corridor.selected_annual_net_price <= corridor.candidate_list_price
    first_revenue = next(row for row in result.cash_flow.annual_cash_flows if row.gross_revenue > 0)
    assert first_revenue.net_revenue / first_revenue.gross_revenue == pytest.approx(
        corridor.selected_annual_net_price / corridor.candidate_list_price
    )
