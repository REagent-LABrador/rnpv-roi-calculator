"""Protected commercial cash-flow model.

The model is deliberately explicit rather than clever: patient starts are built as annual
cohorts, patent protection starts at filing, and label expansion shares the original patent
clock.  It is suitable for screening and comparison, not for legal, reimbursement, or
investment advice.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class DevelopmentStage(_FrozenModel):
    """A stage cost is paid if the program reaches the stage, before its outcome is known."""

    name: str = Field(min_length=1)
    year: int
    cost: float = Field(ge=0)
    success_probability: float = Field(ge=0, le=1)


class PatentAssumptions(_FrozenModel):
    """Simplified US screening contract; the base term is fixed at 20 years from filing."""

    filing_year: int
    base_term_years: Literal[20] = 20
    extension_years: float = Field(default=0, ge=0, le=5)
    regulatory_exclusivity_end_year: float | None = None

    @model_validator(mode="after")
    def validate_regulatory_exclusivity_date(self) -> PatentAssumptions:
        if (
            self.regulatory_exclusivity_end_year is not None
            and self.regulatory_exclusivity_end_year < self.filing_year
        ):
            raise ValueError("regulatory exclusivity cannot end before patent filing")
        return self

    @property
    def expiry_year(self) -> float:
        return self.filing_year + self.base_term_years + self.extension_years

    @property
    def effective_exclusivity_end_year(self) -> float:
        if self.regulatory_exclusivity_end_year is None:
            return self.expiry_year
        return max(self.expiry_year, self.regulatory_exclusivity_end_year)

    def protected_years_at_launch(self, launch_year: int) -> float:
        return max(0.0, self.effective_exclusivity_end_year - launch_year)


class IndicationCommercialAssumptions(_FrozenModel):
    """Patient, access, adherence, price, and cost assumptions for one indication."""

    indication_id: str = Field(min_length=1)
    launch_year: int
    route: str = Field(default="oral", min_length=1)
    backlog_patients: float = Field(default=0, ge=0)
    backlog_release_years: int = Field(default=1, ge=1)
    annual_incident_patients: float = Field(default=0, ge=0)
    incidence_growth_rate: float = Field(default=0, gt=-1)
    coverage_rate: float = Field(default=1, ge=0, le=1)
    authorization_rate: float = Field(default=1, ge=0, le=1)
    patient_affordability_rate: float = Field(ge=0, le=1)
    initiation_rate: float = Field(
        default=1,
        ge=0,
        le=1,
        description="Initiation conditional on access authorization and affordability.",
    )
    provider_capacity_rate: float = Field(default=1, ge=0, le=1)
    adoption_by_year: dict[int, float] = Field(default_factory=dict)
    peak_adoption_rate: float = Field(default=1, ge=0, le=1)
    adoption_ramp_years: int = Field(default=1, ge=1)
    annual_persistence_rate: float = Field(default=1, ge=0, le=1)
    dose_intensity: float = Field(default=1, ge=0, le=1)
    annual_gross_price: float = Field(ge=0)
    gross_to_net_rate: float = Field(default=0, ge=0, le=1)
    annual_price_growth_rate: float = Field(default=0, gt=-1)
    cogs_per_full_dose_patient: float = Field(default=0, ge=0)
    variable_commercial_cost_per_patient: float = Field(default=0, ge=0)
    fixed_commercial_cost_per_year: float = Field(default=0, ge=0)

    @field_validator("adoption_by_year")
    @classmethod
    def validate_adoption_curve(cls, value: dict[int, float]) -> dict[int, float]:
        if any(offset < 0 for offset in value):
            raise ValueError("adoption year offsets must be non-negative")
        if any(fraction < 0 or fraction > 1 for fraction in value.values()):
            raise ValueError("adoption fractions must be between 0 and 1")
        return value

    def adoption_rate(self, year: int) -> float:
        if year < self.launch_year:
            return 0.0
        year_offset = year - self.launch_year
        if self.adoption_by_year:
            if year_offset in self.adoption_by_year:
                return self.adoption_by_year[year_offset]
            offsets = sorted(self.adoption_by_year)
            if year_offset < offsets[0]:
                return 0.0
            if year_offset > offsets[-1]:
                return self.adoption_by_year[offsets[-1]]
            upper_index = next(
                index for index, offset in enumerate(offsets) if offset > year_offset
            )
            lower_offset = offsets[upper_index - 1]
            upper_offset = offsets[upper_index]
            lower_value = self.adoption_by_year[lower_offset]
            upper_value = self.adoption_by_year[upper_offset]
            weight = (year_offset - lower_offset) / (upper_offset - lower_offset)
            return lower_value * (1 - weight) + upper_value * weight
        ramp_fraction = min(1.0, (year - self.launch_year + 1) / self.adoption_ramp_years)
        return self.peak_adoption_rate * ramp_fraction

    def backlog_available(self, year: int) -> float:
        age = year - self.launch_year
        if 0 <= age < self.backlog_release_years:
            return self.backlog_patients / self.backlog_release_years
        return 0.0

    def incident_available(self, year: int) -> float:
        if year < self.launch_year:
            return 0.0
        return self.annual_incident_patients * (1 + self.incidence_growth_rate) ** (
            year - self.launch_year
        )

    def gross_price(self, year: int) -> float:
        if year < self.launch_year:
            return 0.0
        return self.annual_gross_price * (1 + self.annual_price_growth_rate) ** (
            year - self.launch_year
        )


class ExpansionAssumptions(_FrozenModel):
    """A second indication sharing the molecule and therefore the original patent clock."""

    indication: IndicationCommercialAssumptions
    development_stages: tuple[DevelopmentStage, ...] = ()
    conditional_on_initial_success: bool = True
    population_overlap_rate: float = Field(default=0, ge=0, le=1)
    initial_indication_cannibalization_rate: float = Field(default=0, ge=0, le=1)
    franchise_price_spillover_rate: float = Field(default=0, ge=0, le=1)
    shared_commercial_cost_savings_rate: float = Field(default=0, ge=0, le=1)


class ProgramCashFlowInputs(_FrozenModel):
    """Complete manufacturer-perspective inputs for the protected cash-flow calculation."""

    program_id: str = Field(min_length=1)
    valuation_year: int
    currency: str = Field(default="USD", pattern=r"^[A-Za-z]{3}$")
    forecast_end_year: int
    discount_rate: float = Field(default=0.1, gt=-1)
    tax_rate: float = Field(default=0.21, ge=0, le=1)
    patent: PatentAssumptions
    initial_indication: IndicationCommercialAssumptions
    initial_development_stages: tuple[DevelopmentStage, ...] = ()
    expansion: ExpansionAssumptions | None = None
    loe_price_retention: tuple[float, ...] = (0.55, 0.35, 0.25, 0.18, 0.12)
    loe_volume_retention: tuple[float, ...] = (0.75, 0.55, 0.4, 0.3, 0.2)
    critical_inputs_supported: bool = False
    evidence_references: tuple[str, ...] = ()

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _validate_contract(self) -> ProgramCashFlowInputs:
        if self.forecast_end_year < self.valuation_year:
            raise ValueError("forecast_end_year must be on or after valuation_year")
        if not self.loe_price_retention or not self.loe_volume_retention:
            raise ValueError("LOE retention paths cannot be empty")
        if len(self.loe_price_retention) != len(self.loe_volume_retention):
            raise ValueError("LOE price and volume retention paths must have equal length")
        if any(not 0 <= value <= 1 for value in self.loe_price_retention):
            raise ValueError("LOE price retention values must be between zero and one")
        if any(not 0 <= value <= 1 for value in self.loe_volume_retention):
            raise ValueError("LOE volume retention values must be between zero and one")
        if any(
            stage.year > self.initial_indication.launch_year
            for stage in self.initial_development_stages
        ):
            raise ValueError("initial development stages cannot occur after initial launch")
        if (
            self.expansion
            and self.expansion.indication.launch_year < self.initial_indication.launch_year
        ):
            raise ValueError("expansion launch cannot precede initial-indication launch")
        if self.patent.filing_year > self.initial_indication.launch_year:
            raise ValueError("the modeled patent filing cannot occur after initial launch")
        if (
            self.expansion
            and self.expansion.indication.indication_id == self.initial_indication.indication_id
        ):
            raise ValueError("initial and expansion indication IDs must be distinct")
        if self.expansion and any(
            stage.year > self.expansion.indication.launch_year
            for stage in self.expansion.development_stages
        ):
            raise ValueError("expansion development stages cannot occur after expansion launch")
        if (
            self.expansion
            and self.expansion.conditional_on_initial_success
            and any(
                stage.year < self.initial_indication.launch_year
                for stage in self.expansion.development_stages
            )
        ):
            raise ValueError(
                "expansion stages conditional on initial success cannot precede initial launch"
            )
        return self


class DevelopmentRealization(_FrozenModel):
    """Optional sampled development path used by the uncertainty engine."""

    initial_success: bool
    expansion_success: bool = False
    initial_development_costs_by_year: dict[int, float] = Field(default_factory=dict)
    expansion_development_costs_by_year: dict[int, float] = Field(default_factory=dict)

    @field_validator(
        "initial_development_costs_by_year",
        "expansion_development_costs_by_year",
    )
    @classmethod
    def validate_realized_costs(cls, value: dict[int, float]) -> dict[int, float]:
        if any(not math.isfinite(cost) or cost < 0 for cost in value.values()):
            raise ValueError("realized development costs must be finite and non-negative")
        return value


class AnnualCashFlow(_FrozenModel):
    year: int
    protected: bool = Field(description="Whether the full annual bucket is protected.")
    protected_fraction: float = Field(ge=0, le=1)
    initial_new_starts: float
    expansion_new_starts: float
    initial_active_patients: float
    expansion_active_patients: float
    gross_revenue: float
    gross_to_net_deductions: float
    net_revenue: float
    protected_net_revenue: float
    post_loe_net_revenue: float
    cogs: float
    commercial_costs: float
    development_costs: float
    taxes: float
    free_cash_flow: float
    discounted_free_cash_flow: float


class ValueDecomposition(_FrozenModel):
    gross_revenue: float
    gross_to_net_deductions: float
    protected_net_revenue: float
    post_loe_net_revenue: float
    cogs: float
    commercial_costs: float
    development_costs: float
    taxes: float
    initial_indication_discounted_fcf: float
    expansion_increment_discounted_fcf: float


class CashFlowResult(_FrozenModel):
    patent_expiry_year: float
    effective_exclusivity_end_year: float
    effective_protected_years: float
    expansion_effective_protected_years: float | None
    initial_approval_probability: float
    expansion_approval_probability: float
    annual_cash_flows: tuple[AnnualCashFlow, ...]
    npv: float
    peak_annual_net_revenue: float
    peak_annual_net_revenue_year: int | None
    peak_cash_at_risk: float
    value_lost_per_launch_delay_year: float
    value_decomposition: ValueDecomposition
    warnings: tuple[str, ...]


class _IndicationYear:
    """Internal mutable accumulator; never exposed from the public result."""

    def __init__(self, new_starts: float, active_patients: float) -> None:
        self.new_starts = new_starts
        self.active_patients = active_patients


def _probability_and_expected_costs(
    stages: tuple[DevelopmentStage, ...], *, starting_probability: float = 1.0
) -> tuple[float, dict[int, float]]:
    probability_reach = starting_probability
    costs: dict[int, float] = defaultdict(float)
    for stage in sorted(stages, key=lambda item: item.year):
        costs[stage.year] += stage.cost * probability_reach
        probability_reach *= stage.success_probability
    return probability_reach, dict(costs)


def _cohort_path(
    indication: IndicationCommercialAssumptions,
    years: range,
    *,
    population_multiplier: float = 1.0,
) -> dict[int, _IndicationYear]:
    cohorts: dict[int, float] = {}
    result: dict[int, _IndicationYear] = {}
    untreated_pool = 0.0
    for year in years:
        if year < indication.launch_year:
            result[year] = _IndicationYear(0.0, 0.0)
            continue
        newly_available = (
            indication.backlog_available(year) + indication.incident_available(year)
        ) * population_multiplier
        untreated_pool += newly_available
        start_fraction = (
            indication.coverage_rate
            * indication.authorization_rate
            * indication.patient_affordability_rate
            * indication.initiation_rate
            * indication.provider_capacity_rate
            * indication.adoption_rate(year)
        )
        new_starts = untreated_pool * start_fraction
        untreated_pool = max(0.0, untreated_pool - new_starts)
        cohorts[year] = new_starts
        active = sum(
            starts * indication.annual_persistence_rate ** (year - cohort_year)
            for cohort_year, starts in cohorts.items()
        )
        result[year] = _IndicationYear(new_starts, active)
    return result


def _development_expectations(
    inputs: ProgramCashFlowInputs,
    realization: DevelopmentRealization | None,
) -> tuple[float, float, dict[int, float], dict[int, float]]:
    if realization is not None:
        if inputs.expansion is None and (
            realization.expansion_success or realization.expansion_development_costs_by_year
        ):
            raise ValueError("expansion realization supplied for a program without an expansion")
        if (
            inputs.expansion
            and inputs.expansion.conditional_on_initial_success
            and not realization.initial_success
            and (realization.expansion_success or realization.expansion_development_costs_by_year)
        ):
            raise ValueError(
                "a conditional expansion cannot proceed after the initial program fails"
            )
        return (
            float(realization.initial_success),
            float(realization.expansion_success),
            realization.initial_development_costs_by_year,
            realization.expansion_development_costs_by_year,
        )

    initial_probability, initial_costs = _probability_and_expected_costs(
        inputs.initial_development_stages
    )
    expansion_costs: dict[int, float] = {}
    expansion_probability = 0.0
    if inputs.expansion:
        expansion_start = (
            initial_probability if inputs.expansion.conditional_on_initial_success else 1.0
        )
        expansion_probability, expansion_costs = _probability_and_expected_costs(
            inputs.expansion.development_stages,
            starting_probability=expansion_start,
        )
    return initial_probability, expansion_probability, initial_costs, expansion_costs


def _commercial_periods(
    inputs: ProgramCashFlowInputs,
    year: int,
) -> tuple[tuple[float, float, float, bool], ...]:
    """Partition an annual bucket at exclusivity and LOE-anniversary boundaries.

    Each tuple is ``(fraction_of_year, price_retention, volume_retention, protected)``.
    This prevents a fractional patent extension from receiving a full protected calendar year.
    """

    bucket_end = float(year + 1)
    exclusivity_end = inputs.patent.effective_exclusivity_end_year
    cursor = float(year)
    periods: list[tuple[float, float, float, bool]] = []
    epsilon = 1e-12
    while cursor < bucket_end - epsilon:
        if cursor < exclusivity_end - epsilon:
            period_end = min(bucket_end, exclusivity_end)
            periods.append((period_end - cursor, 1.0, 1.0, True))
        else:
            loe_age = max(0, math.floor(cursor - exclusivity_end + epsilon))
            period_end = min(bucket_end, exclusivity_end + loe_age + 1)
            if period_end <= cursor + epsilon:
                loe_age += 1
                period_end = min(bucket_end, exclusivity_end + loe_age + 1)
            if loe_age >= len(inputs.loe_price_retention):
                price_retention = 0.0
                volume_retention = 0.0
            else:
                price_retention = inputs.loe_price_retention[loe_age]
                volume_retention = inputs.loe_volume_retention[loe_age]
            periods.append(
                (
                    period_end - cursor,
                    price_retention,
                    volume_retention,
                    False,
                )
            )
        cursor = period_end
    return tuple(periods)


def _calculate_cashflow(
    inputs: ProgramCashFlowInputs,
    *,
    realization: DevelopmentRealization | None = None,
) -> CashFlowResult:
    years = range(inputs.valuation_year, inputs.forecast_end_year + 1)
    initial_path = _cohort_path(inputs.initial_indication, years)
    expansion_path: dict[int, _IndicationYear] = {year: _IndicationYear(0.0, 0.0) for year in years}
    if inputs.expansion:
        expansion_path = _cohort_path(
            inputs.expansion.indication,
            years,
            population_multiplier=1 - inputs.expansion.population_overlap_rate,
        )

    (
        initial_probability,
        expansion_probability,
        initial_development_costs,
        expansion_development_costs,
    ) = _development_expectations(inputs, realization)
    rows: list[AnnualCashFlow] = []
    discounted_initial_fcf = 0.0
    discounted_expansion_increment = 0.0

    for year in years:
        initial = initial_path[year]
        expansion = expansion_path[year]
        expansion_live = bool(inputs.expansion and year >= inputs.expansion.indication.launch_year)
        expansion_effect_given_initial = 0.0
        if inputs.expansion and expansion_live and initial_probability > 0:
            if realization is not None:
                expansion_effect_given_initial = float(
                    realization.initial_success and realization.expansion_success
                )
            elif inputs.expansion.conditional_on_initial_success:
                expansion_effect_given_initial = min(
                    1.0, expansion_probability / initial_probability
                )
            else:
                expansion_effect_given_initial = expansion_probability
        cannibalization_rate = (
            inputs.expansion.initial_indication_cannibalization_rate if inputs.expansion else 0.0
        )
        spillover_rate = (
            inputs.expansion.franchise_price_spillover_rate if inputs.expansion else 0.0
        )
        expansion_spillover = (
            spillover_rate
            if inputs.expansion and expansion_live and expansion_probability > 0
            else 0.0
        )
        cannibalization = cannibalization_rate * expansion_effect_given_initial
        remaining_initial_volume = 1 - cannibalization
        joint_initial_commercial_factor = 1 - expansion_effect_given_initial * (
            1 - (1 - cannibalization_rate) * (1 - spillover_rate)
        )
        initial_price_factor = (
            joint_initial_commercial_factor / remaining_initial_volume
            if remaining_initial_volume > 0
            else 0.0
        )
        commercial_periods = _commercial_periods(inputs, year)
        protected_fraction = sum(
            fraction for fraction, _, _, is_protected in commercial_periods if is_protected
        )
        volume_retention = sum(
            fraction * period_volume for fraction, _, period_volume, _ in commercial_periods
        )
        revenue_retention = sum(
            fraction * period_price * period_volume
            for fraction, period_price, period_volume, _ in commercial_periods
        )
        protected_revenue_retention = sum(
            fraction * period_price * period_volume
            for fraction, period_price, period_volume, is_protected in commercial_periods
            if is_protected
        )
        post_loe_revenue_retention = revenue_retention - protected_revenue_retention

        initial_active_before_loe = (
            initial.active_patients * initial_probability * (1 - cannibalization)
        )
        expansion_active_before_loe = expansion.active_patients * expansion_probability
        initial_active = initial_active_before_loe * volume_retention
        expansion_active = expansion_active_before_loe * volume_retention
        initial_full_dose = initial_active * inputs.initial_indication.dose_intensity
        expansion_full_dose = (
            expansion_active * inputs.expansion.indication.dose_intensity
            if inputs.expansion
            else 0.0
        )

        initial_gross_before_loe = (
            initial_active_before_loe
            * inputs.initial_indication.dose_intensity
            * inputs.initial_indication.gross_price(year)
            * initial_price_factor
        )
        initial_gross = initial_gross_before_loe * revenue_retention
        expansion_gross = 0.0
        expansion_gross_before_loe = 0.0
        if inputs.expansion:
            expansion_gross_before_loe = (
                expansion_active_before_loe
                * inputs.expansion.indication.dose_intensity
                * inputs.expansion.indication.gross_price(year)
                * (1 - expansion_spillover)
            )
            expansion_gross = expansion_gross_before_loe * revenue_retention
        gross_revenue = initial_gross + expansion_gross
        gtn = initial_gross * inputs.initial_indication.gross_to_net_rate
        if inputs.expansion:
            gtn += expansion_gross * inputs.expansion.indication.gross_to_net_rate
        net_revenue = gross_revenue - gtn
        protected_initial_gross = initial_gross_before_loe * protected_revenue_retention
        post_loe_initial_gross = initial_gross_before_loe * post_loe_revenue_retention
        protected_expansion_gross = expansion_gross_before_loe * protected_revenue_retention
        post_loe_expansion_gross = expansion_gross_before_loe * post_loe_revenue_retention
        protected_net_revenue = protected_initial_gross * (
            1 - inputs.initial_indication.gross_to_net_rate
        )
        post_loe_net_revenue = post_loe_initial_gross * (
            1 - inputs.initial_indication.gross_to_net_rate
        )
        if inputs.expansion:
            protected_net_revenue += protected_expansion_gross * (
                1 - inputs.expansion.indication.gross_to_net_rate
            )
            post_loe_net_revenue += post_loe_expansion_gross * (
                1 - inputs.expansion.indication.gross_to_net_rate
            )

        initial_cogs = initial_full_dose * inputs.initial_indication.cogs_per_full_dose_patient
        initial_variable_cost = (
            initial_active * inputs.initial_indication.variable_commercial_cost_per_patient
        )
        initial_fixed_cost = (
            inputs.initial_indication.fixed_commercial_cost_per_year * initial_probability
            if year >= inputs.initial_indication.launch_year and volume_retention > 0
            else 0.0
        )
        expansion_cogs = 0.0
        expansion_variable_cost = 0.0
        expansion_fixed_cost = 0.0
        if inputs.expansion:
            expansion_cogs = (
                expansion_full_dose * inputs.expansion.indication.cogs_per_full_dose_patient
            )
            expansion_variable_cost = (
                expansion_active * inputs.expansion.indication.variable_commercial_cost_per_patient
            )
            if year >= inputs.expansion.indication.launch_year and volume_retention > 0:
                expansion_fixed_cost = (
                    inputs.expansion.indication.fixed_commercial_cost_per_year
                    * expansion_probability
                    * (1 - inputs.expansion.shared_commercial_cost_savings_rate)
                )
        cogs = initial_cogs + expansion_cogs
        commercial_costs = (
            initial_variable_cost
            + initial_fixed_cost
            + expansion_variable_cost
            + expansion_fixed_cost
        )
        initial_development_cost = initial_development_costs.get(year, 0.0)
        expansion_development_cost = expansion_development_costs.get(year, 0.0)
        development_cost = initial_development_cost + expansion_development_cost
        taxable_operating_income = max(0.0, net_revenue - cogs - commercial_costs)
        taxes = taxable_operating_income * inputs.tax_rate
        fcf = net_revenue - cogs - commercial_costs - development_cost - taxes
        discount_factor = (1 + inputs.discount_rate) ** (year - inputs.valuation_year)
        discounted_fcf = fcf / discount_factor

        initial_net = initial_gross * (1 - inputs.initial_indication.gross_to_net_rate)
        initial_operating = initial_net - initial_cogs - initial_variable_cost - initial_fixed_cost
        initial_tax = max(0.0, initial_operating) * inputs.tax_rate
        initial_fcf = initial_operating - initial_tax - initial_development_cost
        discounted_initial_fcf += initial_fcf / discount_factor
        discounted_expansion_increment += discounted_fcf - initial_fcf / discount_factor

        protected = math.isclose(protected_fraction, 1.0, abs_tol=1e-12)
        rows.append(
            AnnualCashFlow(
                year=year,
                protected=protected,
                protected_fraction=protected_fraction,
                initial_new_starts=initial.new_starts * initial_probability,
                expansion_new_starts=expansion.new_starts * expansion_probability,
                initial_active_patients=initial_active,
                expansion_active_patients=expansion_active,
                gross_revenue=gross_revenue,
                gross_to_net_deductions=gtn,
                net_revenue=net_revenue,
                protected_net_revenue=protected_net_revenue,
                post_loe_net_revenue=post_loe_net_revenue,
                cogs=cogs,
                commercial_costs=commercial_costs,
                development_costs=development_cost,
                taxes=taxes,
                free_cash_flow=fcf,
                discounted_free_cash_flow=discounted_fcf,
            )
        )

    npv = sum(row.discounted_free_cash_flow for row in rows)
    cumulative_cash = 0.0
    minimum_cumulative_cash = 0.0
    for row in rows:
        cumulative_cash += row.free_cash_flow
        minimum_cumulative_cash = min(minimum_cumulative_cash, cumulative_cash)
    warnings: list[str] = []
    if inputs.initial_indication.launch_year >= inputs.patent.expiry_year:
        warnings.append("Initial launch occurs at or after modeled patent expiry.")
    if inputs.expansion and inputs.expansion.indication.launch_year >= inputs.patent.expiry_year:
        warnings.append("Expansion launch occurs at or after modeled patent expiry.")
    if (
        inputs.patent.regulatory_exclusivity_end_year is not None
        and inputs.patent.regulatory_exclusivity_end_year > inputs.patent.expiry_year
    ):
        warnings.append(
            "Regulatory exclusivity extends the modeled protected window beyond patent expiry."
        )
    if not inputs.critical_inputs_supported:
        warnings.append(
            "Critical inputs are not fully evidence-supported; result is screening-only."
        )

    decomposition = ValueDecomposition(
        gross_revenue=sum(row.gross_revenue for row in rows),
        gross_to_net_deductions=sum(row.gross_to_net_deductions for row in rows),
        protected_net_revenue=sum(row.protected_net_revenue for row in rows),
        post_loe_net_revenue=sum(row.post_loe_net_revenue for row in rows),
        cogs=sum(row.cogs for row in rows),
        commercial_costs=sum(row.commercial_costs for row in rows),
        development_costs=sum(row.development_costs for row in rows),
        taxes=sum(row.taxes for row in rows),
        initial_indication_discounted_fcf=discounted_initial_fcf,
        expansion_increment_discounted_fcf=discounted_expansion_increment,
    )
    peak_revenue_row = max(rows, key=lambda row: row.net_revenue)
    return CashFlowResult(
        patent_expiry_year=inputs.patent.expiry_year,
        effective_exclusivity_end_year=inputs.patent.effective_exclusivity_end_year,
        effective_protected_years=inputs.patent.protected_years_at_launch(
            inputs.initial_indication.launch_year
        ),
        expansion_effective_protected_years=(
            inputs.patent.protected_years_at_launch(inputs.expansion.indication.launch_year)
            if inputs.expansion
            else None
        ),
        initial_approval_probability=initial_probability,
        expansion_approval_probability=expansion_probability,
        annual_cash_flows=tuple(rows),
        npv=npv,
        peak_annual_net_revenue=peak_revenue_row.net_revenue,
        peak_annual_net_revenue_year=(
            peak_revenue_row.year if peak_revenue_row.net_revenue > 0 else None
        ),
        peak_cash_at_risk=-minimum_cumulative_cash,
        value_lost_per_launch_delay_year=0.0,
        value_decomposition=decomposition,
        warnings=tuple(warnings),
    )


def delay_launch(inputs: ProgramCashFlowInputs, years: int = 1) -> ProgramCashFlowInputs:
    """Delay all commercial launches without moving the shared patent clock."""

    if years < 0:
        raise ValueError("launch delay cannot be negative")
    initial = inputs.initial_indication.model_copy(
        update={"launch_year": inputs.initial_indication.launch_year + years}
    )
    expansion = inputs.expansion
    if expansion:
        expansion = expansion.model_copy(
            update={
                "indication": expansion.indication.model_copy(
                    update={"launch_year": expansion.indication.launch_year + years}
                )
            }
        )
    return inputs.model_copy(update={"initial_indication": initial, "expansion": expansion})


def calculate_cashflow(
    inputs: ProgramCashFlowInputs,
    *,
    realization: DevelopmentRealization | None = None,
    calculate_delay_cost: bool = True,
) -> CashFlowResult:
    """Calculate manufacturer cash flow for deterministic or sampled development outcomes."""

    result = _calculate_cashflow(inputs, realization=realization)
    if not calculate_delay_cost:
        return result
    delayed = _calculate_cashflow(delay_launch(inputs), realization=realization)
    return result.model_copy(update={"value_lost_per_launch_delay_year": result.npv - delayed.npv})
