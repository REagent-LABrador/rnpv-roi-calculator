"""Public, auditable program-analysis orchestration."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from labrador_roi.cashflow import (
    CashFlowResult,
    DevelopmentStage,
    ExpansionAssumptions,
    IndicationCommercialAssumptions,
    ProgramCashFlowInputs,
    ValueDecomposition,
    calculate_cashflow,
)
from labrador_roi.cashflow import (
    PatentAssumptions as CashFlowPatentAssumptions,
)
from labrador_roi.comparables import ComparableSet, ComparableTier
from labrador_roi.models import (
    CalculationStep,
    DecisionGrade,
    EvidenceMetadata,
    Modality,
    PriceBasis,
    ProgramInput,
    WarningRecord,
    WarningSeverity,
    development_stage_sort_key,
)
from labrador_roi.pricing import (
    AccessEstimate,
    PriceCorridor,
    PricingInputs,
    PricingResult,
    calculate_pricing_corridor,
)
from labrador_roi.provenance import redact, sha256_digest, utc_now
from labrador_roi.simulation import SimulationAssumptions, SimulationResult, simulate_program

SCHEMA_VERSION = "1.3.0"
ENGINE_VERSION = "0.4.0"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Recommendation(StrEnum):
    ADVANCE = "ADVANCE"
    ADVANCE_WITH_EVIDENCE_GATE = "ADVANCE_WITH_EVIDENCE_GATE"
    OPTION_OR_PARTNER = "OPTION_OR_PARTNER"
    HOLD = "HOLD"
    STOP = "STOP"
    NOT_DECISION_GRADE = "NOT_DECISION_GRADE"


class EvidenceReference(_FrozenModel):
    field_path: str
    source_id: str | None = None
    source_url: str | None = None
    citation: str | None = None
    evidence_type: str
    grade: str
    synthetic: bool


class PricingAnalysisSnapshot(_FrozenModel):
    indication_id: str
    decision_grade: DecisionGrade
    annual_net_price_corridor: PriceCorridor | None
    comparable_ids_by_tier: dict[str, tuple[str, ...]]
    access_estimates: tuple[AccessEstimate, ...]


class AccessSnapshot(_FrozenModel):
    indication_id: str
    coverage_rate: float
    authorization_rate: float
    patient_affordability_rate: float
    initiation_rate: float
    provider_capacity_rate: float
    peak_adoption_rate: float
    annual_persistence_rate: float
    dose_intensity: float


class AnalysisSummary(_FrozenModel):
    program_id: str
    recommendation: Recommendation
    deterministic_rnpv: float
    simulated_mean_rnpv: float
    p10_rnpv: float
    p50_rnpv: float
    p90_rnpv: float
    probability_positive_rnpv: float
    peak_annual_net_revenue: float
    peak_annual_net_revenue_year: int | None
    peak_annual_net_revenue_p50: float
    peak_cash_at_risk_p50: float
    effective_protected_years: float
    value_lost_per_launch_delay_year: float


class AnalysisResult(_FrozenModel):
    schema_version: str
    run_id: str
    generated_at: str
    engine_version: str
    input_digest: str
    seed: int
    simulations: int
    simulation_assumptions: SimulationAssumptions
    input_snapshot: dict[str, Any]
    summary: AnalysisSummary
    decision_grade: DecisionGrade
    recommendation: Recommendation
    pricing: tuple[PricingAnalysisSnapshot, ...]
    access: tuple[AccessSnapshot, ...]
    cash_flow: CashFlowResult
    uncertainty: SimulationResult
    calculation_steps: tuple[CalculationStep, ...]
    value_decomposition: ValueDecomposition
    warnings: tuple[WarningRecord, ...]
    critical_evidence_status: dict[str, bool]
    evidence_references: tuple[EvidenceReference, ...]


def _assumption(scopes: tuple[dict[str, Any], ...], key: str, default: Any = None) -> Any:
    for scope in scopes:
        if key in scope and scope[key] is not None:
            return scope[key]
    return default


def _population_eligibility_multiplier(population: Any) -> float:
    multiplier = 1.0
    if population.diagnosed_fraction is not None:
        multiplier *= population.diagnosed_fraction
    if population.clinically_eligible_fraction is not None:
        multiplier *= population.clinically_eligible_fraction
    return multiplier


def _program_with_adjusted_eligible_population(
    program: ProgramInput, indication_id: str
) -> ProgramInput:
    """Apply epidemiologic gates to the payer population without mutating source inputs."""

    indication = program.indication(indication_id)
    eligible = indication.population.eligible_patients
    multiplier = _population_eligibility_multiplier(indication.population)
    if eligible is None or multiplier == 1.0:
        return program
    population = indication.population.model_copy(
        update={"eligible_patients": eligible * multiplier}
    )
    adjusted = indication.model_copy(update={"population": population})
    if indication_id == program.initial_indication.indication_id:
        return program.model_copy(update={"initial_indication": adjusted})
    expansions = [
        adjusted if item.indication_id == indication_id else item
        for item in program.expansion_indications
    ]
    return program.model_copy(update={"expansion_indications": expansions})


def _pricing_inputs(
    program: ProgramInput,
    comparables: ComparableSet,
    indication_id: str,
) -> PricingInputs:
    indication = program.indication(indication_id)
    pricing_program = _program_with_adjusted_eligible_population(program, indication_id)
    scopes = (indication.assumptions, program.assumptions)
    evidence = {**program.evidence, **indication.evidence}
    selected_basis = _assumption(scopes, "selected_net_anchor_basis", PriceBasis.ESTIMATED_NET)
    return PricingInputs(
        program=pricing_program,
        comparables=comparables,
        indication_id=indication_id,
        incremental_qalys=_assumption(scopes, "incremental_qalys"),
        willingness_to_pay_per_qaly=_assumption(scopes, "willingness_to_pay_per_qaly"),
        comparator_total_cost=_assumption(scopes, "comparator_total_cost"),
        new_non_drug_total_cost=_assumption(scopes, "new_non_drug_total_cost"),
        expected_treatment_years=_assumption(scopes, "expected_treatment_years"),
        annual_comparator_drug_cost=_assumption(scopes, "annual_comparator_drug_cost"),
        annual_non_drug_cost_offsets=_assumption(scopes, "annual_non_drug_cost_offsets"),
        annual_payer_budget_limit=_assumption(scopes, "annual_payer_budget_limit"),
        annual_manufacturer_cost=_assumption(scopes, "annual_manufacturer_cost"),
        required_gross_margin_fraction=_assumption(scopes, "required_gross_margin_fraction"),
        selected_net_anchor_basis=PriceBasis(selected_basis),
        candidate_list_price=_assumption(scopes, "candidate_list_price"),
        access_year_offset=int(_assumption(scopes, "access_year_offset", 0)),
        evidence=evidence,
    )


def _rate(value: float | None, scopes: tuple[dict[str, Any], ...], key: str) -> float:
    resolved = value if value is not None else _assumption(scopes, key)
    return 0.0 if resolved is None else float(resolved)


def _commercial_indication(
    program: ProgramInput,
    indication_id: str,
    pricing: PricingResult,
) -> tuple[IndicationCommercialAssumptions, list[WarningRecord]]:
    indication = program.indication(indication_id)
    scopes = (indication.assumptions, program.assumptions)
    warnings: list[WarningRecord] = []
    population = indication.population
    population_multiplier = _population_eligibility_multiplier(population)
    backlog = population.prevalent_backlog_patients
    if backlog is None:
        backlog = population.eligible_patients
    if backlog is None:
        backlog = 0.0
        warnings.append(
            WarningRecord(
                code="MISSING_BACKLOG_POPULATION",
                field=f"{indication_id}.population",
                severity=WarningSeverity.ERROR,
                message="No prevalent backlog or eligible population was supplied; zero was used.",
            )
        )
    backlog *= population_multiplier
    incidence = population.annual_incident_patients
    if incidence is None:
        incidence = 0.0
        warnings.append(
            WarningRecord(
                code="MISSING_INCIDENCE",
                field=f"{indication_id}.annual_incident_patients",
                severity=WarningSeverity.ERROR,
                message="Annual incidence is unknown; no incident patients were added.",
            )
        )
    incidence *= population_multiplier

    corridor = pricing.annual_net_price_corridor
    selected_net_price = pricing.selected_annual_net_price
    if selected_net_price is None:
        fallback_gross = _assumption(scopes, "annual_gross_price")
        fallback_gtn = float(_assumption(scopes, "gross_to_net_rate", 0.0))
        fallback_net = (
            float(fallback_gross) * (1 - fallback_gtn) if fallback_gross is not None else 0.0
        )
        selected_net_price = _assumption(scopes, "annual_net_price", fallback_net)
        warnings.append(
            WarningRecord(
                code="MISSING_PRICE_CORRIDOR",
                field=f"{indication_id}.price",
                severity=WarningSeverity.ERROR,
                message=(
                    "No supported price corridor was available; explicit fallback or zero was used."
                ),
            )
        )
    gtn = (
        corridor.estimated_gross_to_net_fraction
        if corridor and corridor.estimated_gross_to_net_fraction is not None
        else _assumption(scopes, "gross_to_net_rate", 0.0)
    )
    gtn = float(gtn)
    candidate_list = corridor.candidate_list_price if corridor else None
    if candidate_list is not None:
        annual_gross_price = candidate_list
    elif gtn < 1:
        annual_gross_price = float(selected_net_price) / (1 - gtn)
    else:
        annual_gross_price = 0.0

    selected_access = next(
        (
            estimate
            for estimate in pricing.access_estimates
            if abs(estimate.annual_net_price - float(selected_net_price)) <= 1e-6
        ),
        None,
    )
    affordability = selected_access.income_affordable_share if selected_access else None
    if affordability is None:
        affordability = _assumption(scopes, "patient_affordability_rate")
    if affordability is None:
        affordability = 0.0
        warnings.append(
            WarningRecord(
                code="UNKNOWN_PATIENT_AFFORDABILITY",
                field=f"{indication_id}.patient_affordability_rate",
                severity=WarningSeverity.ERROR,
                message=(
                    "Income-mediated affordability at the selected price is unknown; zero "
                    "affordable starts were used rather than assuming patient wealth."
                ),
            )
        )

    adoption_by_year = indication.access.adoption_by_year
    if adoption_by_year:
        peak_adoption = max(adoption_by_year.values())
        ramp_years = max(adoption_by_year) + 1
    else:
        peak_adoption_input = _assumption(scopes, "peak_adoption_rate")
        peak_adoption = 0.0 if peak_adoption_input is None else float(peak_adoption_input)
        ramp_years = int(_assumption(scopes, "adoption_ramp_years", 1))
        if peak_adoption_input is None:
            warnings.append(
                WarningRecord(
                    code="MISSING_ADOPTION",
                    field=f"{indication_id}.access.adoption_by_year",
                    severity=WarningSeverity.ERROR,
                    message="No adoption curve was supplied; zero adoption was used.",
                )
            )
    persistence = _assumption(scopes, "annual_persistence_rate")
    if persistence is None:
        persistence = 0.0
        warnings.append(
            WarningRecord(
                code="MISSING_PERSISTENCE",
                field=f"{indication_id}.annual_persistence_rate",
                severity=WarningSeverity.ERROR,
                message="Annual persistence is missing; zero carry-forward was used.",
            )
        )
    dose_intensity = _assumption(scopes, "dose_intensity")
    if dose_intensity is None:
        dose_intensity = 0.0
        warnings.append(
            WarningRecord(
                code="MISSING_DOSE_INTENSITY",
                field=f"{indication_id}.dose_intensity",
                severity=WarningSeverity.ERROR,
                message="Dose intensity is missing; zero paid dose was used.",
            )
        )
    annual_manufacturer_cost = _assumption(scopes, "annual_manufacturer_cost", 0.0)
    cogs = _assumption(scopes, "cogs_per_full_dose_patient", annual_manufacturer_cost)
    return (
        IndicationCommercialAssumptions(
            indication_id=indication_id,
            launch_year=indication.launch_year,
            route=(indication.route or program.route).value,
            backlog_patients=float(backlog),
            backlog_release_years=int(_assumption(scopes, "backlog_release_years", 1)),
            annual_incident_patients=float(incidence),
            incidence_growth_rate=float(_assumption(scopes, "incidence_growth_rate", 0.0)),
            coverage_rate=_rate(indication.access.coverage_fraction, scopes, "coverage_rate"),
            authorization_rate=_rate(
                indication.access.prior_authorization_pass_fraction,
                scopes,
                "authorization_rate",
            ),
            patient_affordability_rate=float(affordability),
            initiation_rate=_rate(indication.access.initiation_fraction, scopes, "initiation_rate"),
            provider_capacity_rate=_rate(
                indication.access.provider_capacity_fraction,
                scopes,
                "provider_capacity_rate",
            ),
            adoption_by_year=dict(adoption_by_year),
            peak_adoption_rate=peak_adoption,
            adoption_ramp_years=ramp_years,
            annual_persistence_rate=float(persistence),
            dose_intensity=float(dose_intensity),
            annual_gross_price=annual_gross_price,
            gross_to_net_rate=gtn,
            annual_price_growth_rate=float(_assumption(scopes, "annual_price_growth_rate", 0.0)),
            cogs_per_full_dose_patient=float(cogs),
            variable_commercial_cost_per_patient=float(
                _assumption(scopes, "variable_commercial_cost_per_patient", 0.0)
            ),
            fixed_commercial_cost_per_year=float(
                _assumption(scopes, "fixed_commercial_cost_per_year", 0.0)
            ),
        ),
        warnings,
    )


def _development_stages(
    program: ProgramInput,
    *,
    launch_year: int,
    prefix: str = "",
) -> tuple[DevelopmentStage, ...]:
    development = program.development
    elapsed = 0.0
    stages: list[DevelopmentStage] = []
    for name in development.ordered_stage_names():
        cost = development.stage_costs[name]
        elapsed += development.stage_durations_years[name]
        year = min(launch_year, program.valuation_year + round(elapsed))
        probability = development.stage_success_probabilities[name]
        stages.append(
            DevelopmentStage(
                name=f"{prefix}{name}",
                year=year,
                cost=cost,
                success_probability=probability,
            )
        )
    if not stages and development.program_probability_of_approval is not None:
        stages.append(
            DevelopmentStage(
                name=f"{prefix}remaining_program",
                year=min(launch_year, program.valuation_year),
                cost=0.0,
                success_probability=development.program_probability_of_approval,
            )
        )
    return tuple(stages)


def _expansion_development_stages(
    program: ProgramInput,
    *,
    start_year: int,
    launch_year: int,
) -> tuple[DevelopmentStage, ...]:
    assumptions = program.development.assumptions
    costs = assumptions.get("expansion_stage_costs", {})
    durations = assumptions.get("expansion_stage_durations_years", {})
    probabilities = assumptions.get("expansion_stage_success_probabilities", {})
    explicit_order = assumptions.get("expansion_stage_order", [])
    if set(durations) != set(costs):
        raise ValueError(
            "expansion stage durations must have exactly the same keys as expansion stage costs"
        )
    if set(probabilities) != set(costs):
        raise ValueError(
            "expansion stage success probabilities must have exactly the same keys as "
            "expansion stage costs"
        )
    if explicit_order:
        if not isinstance(explicit_order, list) or any(
            not isinstance(item, str) for item in explicit_order
        ):
            raise ValueError("expansion_stage_order must be a list of stage names")
        if len(explicit_order) != len(set(explicit_order)) or set(explicit_order) != set(costs):
            raise ValueError("expansion_stage_order must contain every costed stage exactly once")
        stage_names = tuple(explicit_order)
    else:
        stage_names = tuple(sorted(costs, key=development_stage_sort_key))
    elapsed = 0.0
    stages: list[DevelopmentStage] = []
    for name in stage_names:
        cost = costs[name]
        elapsed += float(durations[name])
        stages.append(
            DevelopmentStage(
                name=f"expansion:{name}",
                year=min(launch_year, start_year + round(elapsed)),
                cost=float(cost),
                success_probability=float(probabilities[name]),
            )
        )
    return tuple(stages)


def _pricing_snapshot(result: PricingResult) -> PricingAnalysisSnapshot:
    by_tier = {
        tier.value: tuple(
            item.comparable.comparable_id for item in result.comparable_selection.by_tier(tier)
        )
        for tier in ComparableTier
    }
    return PricingAnalysisSnapshot(
        indication_id=result.indication_id,
        decision_grade=result.decision_grade,
        annual_net_price_corridor=result.annual_net_price_corridor,
        comparable_ids_by_tier=by_tier,
        access_estimates=tuple(result.access_estimates),
    )


def _supported(metadata: EvidenceMetadata | None) -> bool:
    return bool(metadata and metadata.supports_decision)


def _cashflow_evidence_status(
    program: ProgramInput,
    indication_ids: list[str],
    available_comparable_ids: set[str] | None = None,
) -> dict[str, bool]:
    patent_fields = ["filing_year"]
    if program.patent.extension_years > 0:
        patent_fields.append("extension_years")
    if program.patent.regulatory_exclusivity_end_year is not None:
        patent_fields.append("regulatory_exclusivity_end_year")
    patent_group = program.patent.evidence.get("patent_inputs")
    status = {
        "cashflow.patent": _supported(patent_group)
        or all(_supported(program.patent.evidence.get(name)) for name in patent_fields),
        "cashflow.development": any(
            item.supports_decision for item in program.development.evidence.values()
        ),
        "cashflow.development_path": bool(program.development.stage_costs)
        or program.development.program_probability_of_approval is not None,
        "cashflow.development_cost_path": bool(program.development.stage_costs)
        or program.development.is_post_approval,
    }
    if program.modality == Modality.ANTIBODY:
        loe_fields = ("loe_price_retention", "loe_volume_retention")
        loe_group = program.evidence.get("loe_retention")
        effective_exclusivity_end = program.patent.effective_exclusivity_end_year
        latest_launch = max(program.indication(item).launch_year for item in indication_ids)
        default_horizon = int(max(effective_exclusivity_end + 5, latest_launch + 10))
        forecast_end_year = int(program.assumptions.get("forecast_end_year", default_horizon))
        required_loe_periods = max(
            0,
            math.ceil(forecast_end_year + 1 - effective_exclusivity_end),
        )
        loe_paths = tuple(program.assumptions.get(field_name) for field_name in loe_fields)
        status["cashflow.antibody_loe_retention"] = all(
            field_name in program.assumptions for field_name in loe_fields
        ) and (
            _supported(loe_group)
            or all(_supported(program.evidence.get(field_name)) for field_name in loe_fields)
        )
        status["cashflow.antibody_loe_horizon"] = all(
            isinstance(path, (list, tuple)) and len(path) >= required_loe_periods
            for path in loe_paths
        )
        status["cashflow.antibody_comparator_selection"] = all(
            bool(program.indication(indication_id).comparator_ids)
            and len(program.indication(indication_id).comparator_ids)
            == len(set(program.indication(indication_id).comparator_ids))
            and available_comparable_ids is not None
            and set(program.indication(indication_id).comparator_ids) <= available_comparable_ids
            for indication_id in indication_ids
        )
    reported_probability = program.development.program_probability_of_approval
    if reported_probability is not None:
        stage_probabilities = program.development.stage_success_probabilities.values()
        stage_product = math.prod(stage_probabilities)
        status["cashflow.development_probability_reconciliation"] = (
            not program.development.stage_success_probabilities
            or math.isclose(
                reported_probability,
                stage_product,
                rel_tol=1e-6,
                abs_tol=1e-9,
            )
        )
    for indication_id in indication_ids:
        indication = program.indication(indication_id)
        population = indication.population
        population_fields = ["prevalent_backlog_patients", "annual_incident_patients"]
        if population.prevalent_backlog_patients is None:
            population_fields[0] = "eligible_patients"
        if population.diagnosed_fraction is not None:
            population_fields.append("diagnosed_fraction")
        if population.clinically_eligible_fraction is not None:
            population_fields.append("clinically_eligible_fraction")
        population_group = population.evidence.get("population_inputs")
        status[f"cashflow.{indication_id}.population"] = _supported(population_group) or all(
            _supported(population.evidence.get(name)) for name in population_fields
        )

        access_group = indication.access.evidence.get("access_curve")
        access_fields = (
            "coverage_fraction",
            "prior_authorization_pass_fraction",
            "initiation_fraction",
            "provider_capacity_fraction",
            "adoption_by_year",
        )
        status[f"cashflow.{indication_id}.access"] = _supported(access_group) or all(
            _supported(indication.access.evidence.get(name)) for name in access_fields
        )

        commercial_group = indication.evidence.get("commercial_inputs")
        commercial_fields = (
            "annual_persistence_rate",
            "dose_intensity",
            "gross_to_net_rate",
            "cogs_per_full_dose_patient",
        )
        status[f"cashflow.{indication_id}.commercial"] = _supported(commercial_group) or all(
            _supported(indication.evidence.get(name)) for name in commercial_fields
        )
    if len(indication_ids) > 1:
        expansion_input = program.indication(indication_ids[1])
        interaction_group = expansion_input.population.evidence.get("population_inputs")
        interaction_fields = ("overlap_with_initial_fraction", "cannibalization_fraction")
        status["cashflow.expansion.population_interaction"] = all(
            getattr(expansion_input.population, name) is not None for name in interaction_fields
        ) and (
            _supported(interaction_group)
            or all(
                _supported(expansion_input.population.evidence.get(name))
                for name in interaction_fields
            )
        )
        expansion_assumptions = program.development.assumptions
        costs = expansion_assumptions.get("expansion_stage_costs", {})
        durations = expansion_assumptions.get("expansion_stage_durations_years", {})
        probabilities = expansion_assumptions.get("expansion_stage_success_probabilities", {})
        status["cashflow.expansion.development_path"] = bool(costs) and (
            set(costs) == set(durations) == set(probabilities)
        )
    if len(program.expansion_indications) > 1:
        status["cashflow.expansion_scope"] = False
    return status


def _cashflow_from_program(
    program: ProgramInput,
    comparables: ComparableSet,
) -> tuple[
    ProgramCashFlowInputs,
    tuple[PricingResult, ...],
    list[WarningRecord],
]:
    if program.patent.base_term_years != 20:
        raise ValueError("LABrador's screening contract requires a 20-year patent term from filing")
    warnings: list[WarningRecord] = []
    indication_ids = [program.initial_indication.indication_id]
    if program.expansion_indications:
        indication_ids.append(program.expansion_indications[0].indication_id)
    if len(program.expansion_indications) > 1:
        warnings.append(
            WarningRecord(
                code="EXPANSION_SCOPE_LIMIT",
                field="expansion_indications",
                severity=WarningSeverity.ERROR,
                message="Only the first expansion indication is modeled in the two-label MVP.",
            )
        )
    pricing_results = tuple(
        calculate_pricing_corridor(_pricing_inputs(program, comparables, indication_id))
        for indication_id in indication_ids
    )
    initial, commercial_warnings = _commercial_indication(
        program, indication_ids[0], pricing_results[0]
    )
    warnings.extend(commercial_warnings)
    expansion: ExpansionAssumptions | None = None
    if len(indication_ids) > 1:
        expansion_indication, expansion_warnings = _commercial_indication(
            program, indication_ids[1], pricing_results[1]
        )
        warnings.extend(expansion_warnings)
        expansion_input = program.expansion_indications[0]
        scopes = (expansion_input.assumptions, program.assumptions)
        expansion_conditional = bool(
            _assumption(scopes, "expansion_conditional_on_initial_success", True)
        )
        default_expansion_start = (
            program.initial_indication.launch_year
            if expansion_conditional
            else program.valuation_year
        )
        expansion_start = int(
            _assumption(scopes, "expansion_development_start_year", default_expansion_start)
        )
        expansion_stages = _expansion_development_stages(
            program,
            start_year=expansion_start,
            launch_year=expansion_indication.launch_year,
        )
        if not expansion_stages:
            warnings.append(
                WarningRecord(
                    code="MISSING_EXPANSION_DEVELOPMENT_PATH",
                    field="development.assumptions.expansion_stage_costs",
                    severity=WarningSeverity.ERROR,
                    message=(
                        "No explicit expansion development path was supplied; expansion "
                        "commercial probability is conditional only on the initial program."
                    ),
                )
            )
        expansion = ExpansionAssumptions(
            indication=expansion_indication,
            development_stages=expansion_stages,
            conditional_on_initial_success=expansion_conditional,
            population_overlap_rate=float(
                expansion_input.population.overlap_with_initial_fraction or 0.0
            ),
            initial_indication_cannibalization_rate=float(
                expansion_input.population.cannibalization_fraction or 0.0
            ),
            franchise_price_spillover_rate=float(
                _assumption(scopes, "expansion_price_spillover_rate", 0.0)
            ),
            shared_commercial_cost_savings_rate=float(
                _assumption(scopes, "shared_commercial_cost_savings_rate", 0.0)
            ),
        )
    effective_exclusivity_end = program.patent.effective_exclusivity_end_year
    latest_launch = max(program.indication(item).launch_year for item in indication_ids)
    default_horizon = int(max(effective_exclusivity_end + 5, latest_launch + 10))
    cashflow_evidence_status = _cashflow_evidence_status(
        program,
        indication_ids,
        {item.comparable_id for item in comparables.comparables},
    )
    reconciliation_key = "cashflow.development_probability_reconciliation"
    if cashflow_evidence_status.get(reconciliation_key) is False:
        stage_product = math.prod(program.development.stage_success_probabilities.values())
        warnings.append(
            WarningRecord(
                code="INCONSISTENT_PROGRAM_APPROVAL_PROBABILITY",
                field="development.program_probability_of_approval",
                severity=WarningSeverity.ERROR,
                message=(
                    "Reported program_probability_of_approval "
                    f"({program.development.program_probability_of_approval:.12g}) does not "
                    f"equal the stage-probability product ({stage_product:.12g}); the stage "
                    "path remains authoritative and the analysis is not decision-grade."
                ),
            )
        )
    for field_name, supported in cashflow_evidence_status.items():
        if not supported:
            if field_name == reconciliation_key:
                continue
            warnings.append(
                WarningRecord(
                    code="UNSUPPORTED_CASHFLOW_INPUT",
                    field=field_name,
                    severity=WarningSeverity.ERROR,
                    message=f"Critical cash-flow evidence group '{field_name}' is unsupported.",
                )
            )
    critical_supported = all(cashflow_evidence_status.values())
    initial_development_stages = _development_stages(program, launch_year=initial.launch_year)
    if not initial_development_stages:
        warnings.append(
            WarningRecord(
                code="MISSING_INITIAL_DEVELOPMENT_PATH",
                field="development",
                severity=WarningSeverity.ERROR,
                message=(
                    "No stage-level development path or explicit program approval probability "
                    "was supplied."
                ),
            )
        )
    cashflow = ProgramCashFlowInputs(
        program_id=program.program_id,
        valuation_year=program.valuation_year,
        currency=program.currency,
        forecast_end_year=int(program.assumptions.get("forecast_end_year", default_horizon)),
        discount_rate=float(program.assumptions.get("discount_rate", 0.1)),
        tax_rate=float(program.assumptions.get("tax_rate", 0.21)),
        patent=CashFlowPatentAssumptions(
            filing_year=program.patent.filing_year,
            extension_years=program.patent.extension_years,
            regulatory_exclusivity_end_year=program.patent.regulatory_exclusivity_end_year,
        ),
        initial_indication=initial,
        initial_development_stages=initial_development_stages,
        expansion=expansion,
        loe_price_retention=tuple(
            program.assumptions.get("loe_price_retention", (0.55, 0.35, 0.25, 0.18, 0.12))
        ),
        loe_volume_retention=tuple(
            program.assumptions.get("loe_volume_retention", (0.75, 0.55, 0.4, 0.3, 0.2))
        ),
        critical_inputs_supported=critical_supported,
        evidence_references=tuple(
            metadata.source_id or metadata.source_url or key
            for key, metadata in {**program.evidence, **program.initial_indication.evidence}.items()
        ),
    )
    return cashflow, pricing_results, warnings


def _walk_evidence(value: Any, path: str = "") -> list[EvidenceReference]:
    if isinstance(value, EvidenceMetadata):
        cleaned = redact(
            {
                "source_id": value.source_id,
                "source_url": value.source_url,
                "citation": value.citation,
            }
        )
        return [
            EvidenceReference(
                field_path=path,
                source_id=cleaned["source_id"],
                source_url=cleaned["source_url"],
                citation=cleaned["citation"],
                evidence_type=value.evidence_type.value,
                grade=value.grade.value,
                synthetic=value.synthetic,
            )
        ]
    if isinstance(value, BaseModel):
        references: list[EvidenceReference] = []
        for name in value.__class__.model_fields:
            nested_path = f"{path}.{name}" if path else name
            references.extend(_walk_evidence(getattr(value, name), nested_path))
        return references
    if isinstance(value, dict):
        references = []
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            references.extend(_walk_evidence(nested, nested_path))
        return references
    if isinstance(value, (list, tuple)):
        references = []
        for index, nested in enumerate(value):
            references.extend(_walk_evidence(nested, f"{path}[{index}]"))
        return references
    return []


def _cashflow_steps(
    cashflow: CashFlowResult,
    inputs: ProgramCashFlowInputs,
) -> list[CalculationStep]:
    total_patient_years = sum(
        row.initial_active_patients + row.expansion_active_patients
        for row in cashflow.annual_cash_flows
    )
    return [
        CalculationStep(
            step_id="patent_clock",
            label="Effective protected commercial window",
            formula=(
                "max(0, max(filing year + 20 + capped extension, regulatory exclusivity end) "
                "- initial launch year)"
            ),
            inputs={
                "filing_year": inputs.patent.filing_year,
                "base_term_years": inputs.patent.base_term_years,
                "extension_years": inputs.patent.extension_years,
                "regulatory_exclusivity_end_year": (inputs.patent.regulatory_exclusivity_end_year),
                "effective_exclusivity_end_year": (inputs.patent.effective_exclusivity_end_year),
                "launch_year": inputs.initial_indication.launch_year,
            },
            result=cashflow.effective_protected_years,
            unit="years",
            notes=["Label expansion uses the same clock and cannot reset it."],
        ),
        CalculationStep(
            step_id="patient_vintages",
            label="Coverage-adjusted persistent patient-years",
            formula=(
                "add each backlog/incident cohort once to the untreated pool; apply coverage, "
                "authorization, patient affordability, conditional initiation, provider "
                "capacity, and adoption; carry non-starters forward; then sum starts by vintage "
                "* persistence^age * dose intensity"
            ),
            inputs={"annual_rows": len(cashflow.annual_cash_flows)},
            result=total_patient_years,
            unit="patient-years",
        ),
        CalculationStep(
            step_id="protected_net_revenue",
            label="Protected manufacturer net revenue",
            formula=(
                "time-weighted gross revenue - gross-to-net deductions before the effective "
                "exclusivity end, prorating fractional transition-year protection"
            ),
            inputs={
                "gross_revenue": cashflow.value_decomposition.gross_revenue,
                "gross_to_net_deductions": (cashflow.value_decomposition.gross_to_net_deductions),
            },
            result=cashflow.value_decomposition.protected_net_revenue,
            unit=inputs.currency,
        ),
        CalculationStep(
            step_id="risk_adjusted_npv",
            label="Risk-adjusted manufacturer NPV",
            formula="sum(expected annual free cash flow / (1 + discount rate)^year offset)",
            inputs={
                "discount_rate": inputs.discount_rate,
                "initial_approval_probability": cashflow.initial_approval_probability,
                "expansion_approval_probability": cashflow.expansion_approval_probability,
            },
            result=cashflow.npv,
            unit=inputs.currency,
        ),
        CalculationStep(
            step_id="peak_annual_net_revenue",
            label="Peak annual manufacturer net revenue",
            formula="max(annual cash-flow ledger net revenue)",
            inputs={
                "peak_year": cashflow.peak_annual_net_revenue_year,
                "annual_rows": len(cashflow.annual_cash_flows),
            },
            result=cashflow.peak_annual_net_revenue,
            unit=f"{inputs.currency}/year",
            notes=["Derived from the same annual ledger used for NPV."],
        ),
        CalculationStep(
            step_id="launch_delay_cost",
            label="Value lost for one year of commercial launch delay",
            formula="base NPV - NPV with all launches delayed one year and patent clock unchanged",
            inputs={"patent_expiry_year": cashflow.patent_expiry_year},
            result=cashflow.value_lost_per_launch_delay_year,
            unit=inputs.currency,
        ),
    ]


def _recommendation(
    grade: DecisionGrade,
    uncertainty: SimulationResult,
) -> Recommendation:
    if grade == DecisionGrade.NOT_DECISION_GRADE:
        return Recommendation.NOT_DECISION_GRADE
    probability_positive = uncertainty.rnpv.probability_positive or 0.0
    if uncertainty.rnpv.p90 <= 0:
        return Recommendation.STOP
    if uncertainty.rnpv.p50 > 0 and probability_positive >= 0.65:
        return Recommendation.ADVANCE
    if uncertainty.rnpv.p50 > 0 and uncertainty.rnpv.mean > 0:
        return Recommendation.ADVANCE_WITH_EVIDENCE_GATE
    if uncertainty.rnpv.mean > 0:
        return Recommendation.OPTION_OR_PARTNER
    return Recommendation.HOLD


def analyze_program(
    program: ProgramInput | ProgramCashFlowInputs,
    comparables: ComparableSet | list[Any] | None = None,
    *,
    simulations: int = 1_000,
    seed: int = 0,
    simulation_assumptions: SimulationAssumptions | None = None,
) -> AnalysisResult:
    """Run comparable selection, pricing, protected cash flow, and seeded uncertainty."""

    raw_comparables: ComparableSet
    pricing_results: tuple[PricingResult, ...] = ()
    adapter_warnings: list[WarningRecord] = []
    if isinstance(program, ProgramCashFlowInputs):
        program = ProgramCashFlowInputs.model_validate(
            program.model_dump(mode="python", warnings=False)
        )
        cashflow_inputs = program
        raw_comparables = ComparableSet(comparables=[])
        source_object: Any = {"cashflow_inputs": program}
        low_level_supported = bool(
            program.critical_inputs_supported and program.evidence_references
        )
        critical_evidence_status = {"cashflow_inputs": low_level_supported}
        if program.critical_inputs_supported and not program.evidence_references:
            adapter_warnings.append(
                WarningRecord(
                    code="MISSING_LOW_LEVEL_EVIDENCE",
                    field="cashflow_inputs.evidence_references",
                    severity=WarningSeverity.ERROR,
                    message=(
                        "Low-level cash-flow inputs cannot qualify for decision grade without "
                        "evidence references."
                    ),
                )
            )
    else:
        program = ProgramInput.model_validate(program.model_dump(mode="python", warnings=False))
        if comparables is None:
            raw_comparables = ComparableSet(comparables=[])
        elif isinstance(comparables, ComparableSet):
            raw_comparables = ComparableSet.model_validate(
                comparables.model_dump(mode="python", warnings=False)
            )
        else:
            raw_comparables = ComparableSet.model_validate({"comparables": comparables})
        cashflow_inputs, pricing_results, adapter_warnings = _cashflow_from_program(
            program, raw_comparables
        )
        source_object = {"program": program, "comparables": raw_comparables}
        modeled_indications = [program.initial_indication.indication_id]
        if program.expansion_indications:
            modeled_indications.append(program.expansion_indications[0].indication_id)
        critical_evidence_status = _cashflow_evidence_status(
            program,
            modeled_indications,
            {item.comparable_id for item in raw_comparables.comparables},
        )
        for result in pricing_results:
            critical_evidence_status.update(
                {
                    f"pricing.{result.indication_id}.{name}": supported
                    for name, supported in result.critical_evidence_status.items()
                }
            )

    resolved_simulation_assumptions = simulation_assumptions or SimulationAssumptions()
    cashflow = calculate_cashflow(cashflow_inputs)
    uncertainty = simulate_program(
        cashflow_inputs,
        simulations=simulations,
        seed=seed,
        assumptions=resolved_simulation_assumptions,
    )
    pricing_grade = all(
        result.decision_grade == DecisionGrade.DECISION_GRADE for result in pricing_results
    )
    warnings = list(adapter_warnings)
    for result in pricing_results:
        warnings.extend(result.warnings)
    warnings.extend(
        WarningRecord(code="CASHFLOW_WARNING", message=message) for message in cashflow.warnings
    )
    has_error = any(item.severity == WarningSeverity.ERROR for item in warnings)
    decision_grade = (
        DecisionGrade.DECISION_GRADE
        if (
            all(critical_evidence_status.values())
            and (pricing_grade or not pricing_results)
            and not has_error
        )
        else DecisionGrade.NOT_DECISION_GRADE
    )
    recommendation = _recommendation(decision_grade, uncertainty)
    evidence = _walk_evidence(source_object)
    calculation_steps: list[CalculationStep] = []
    for pricing_result in pricing_results:
        calculation_steps.extend(
            step.model_copy(
                update={
                    "step_id": f"pricing:{pricing_result.indication_id}:{step.step_id}",
                    "label": f"{pricing_result.indication_id}: {step.label}",
                }
            )
            for step in pricing_result.calculation_steps
        )
    calculation_steps.extend(_cashflow_steps(cashflow, cashflow_inputs))

    digest = sha256_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "inputs": source_object,
            "seed": seed,
            "simulations": simulations,
            "simulation_assumptions": resolved_simulation_assumptions,
        }
    )
    run_id = f"run_{digest.removeprefix('sha256:')[:20]}"
    access = [
        AccessSnapshot(
            indication_id=cashflow_inputs.initial_indication.indication_id,
            coverage_rate=cashflow_inputs.initial_indication.coverage_rate,
            authorization_rate=cashflow_inputs.initial_indication.authorization_rate,
            patient_affordability_rate=(
                cashflow_inputs.initial_indication.patient_affordability_rate
            ),
            initiation_rate=cashflow_inputs.initial_indication.initiation_rate,
            provider_capacity_rate=cashflow_inputs.initial_indication.provider_capacity_rate,
            peak_adoption_rate=cashflow_inputs.initial_indication.peak_adoption_rate,
            annual_persistence_rate=cashflow_inputs.initial_indication.annual_persistence_rate,
            dose_intensity=cashflow_inputs.initial_indication.dose_intensity,
        )
    ]
    if cashflow_inputs.expansion:
        item = cashflow_inputs.expansion.indication
        access.append(
            AccessSnapshot(
                indication_id=item.indication_id,
                coverage_rate=item.coverage_rate,
                authorization_rate=item.authorization_rate,
                patient_affordability_rate=item.patient_affordability_rate,
                initiation_rate=item.initiation_rate,
                provider_capacity_rate=item.provider_capacity_rate,
                peak_adoption_rate=item.peak_adoption_rate,
                annual_persistence_rate=item.annual_persistence_rate,
                dose_intensity=item.dose_intensity,
            )
        )
    probability_positive = uncertainty.rnpv.probability_positive or 0.0
    summary = AnalysisSummary(
        program_id=cashflow_inputs.program_id,
        recommendation=recommendation,
        deterministic_rnpv=cashflow.npv,
        simulated_mean_rnpv=uncertainty.rnpv.mean,
        p10_rnpv=uncertainty.rnpv.p10,
        p50_rnpv=uncertainty.rnpv.p50,
        p90_rnpv=uncertainty.rnpv.p90,
        probability_positive_rnpv=probability_positive,
        peak_annual_net_revenue=cashflow.peak_annual_net_revenue,
        peak_annual_net_revenue_year=cashflow.peak_annual_net_revenue_year,
        peak_annual_net_revenue_p50=uncertainty.peak_annual_net_revenue.p50,
        peak_cash_at_risk_p50=uncertainty.peak_cash_at_risk.p50,
        effective_protected_years=cashflow.effective_protected_years,
        value_lost_per_launch_delay_year=cashflow.value_lost_per_launch_delay_year,
    )
    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        generated_at=utc_now(),
        engine_version=ENGINE_VERSION,
        input_digest=digest,
        seed=seed,
        simulations=simulations,
        simulation_assumptions=resolved_simulation_assumptions,
        input_snapshot=redact(source_object),
        summary=summary,
        decision_grade=decision_grade,
        recommendation=recommendation,
        pricing=tuple(_pricing_snapshot(result) for result in pricing_results),
        access=tuple(access),
        cash_flow=cashflow,
        uncertainty=uncertainty,
        calculation_steps=tuple(calculation_steps),
        value_decomposition=cashflow.value_decomposition,
        warnings=tuple(warnings),
        critical_evidence_status=critical_evidence_status,
        evidence_references=tuple(evidence),
    )
