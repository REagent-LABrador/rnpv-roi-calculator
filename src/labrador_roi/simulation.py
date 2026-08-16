"""Seeded uncertainty simulation for protected commercial cash flow."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from labrador_roi.cashflow import (
    DevelopmentRealization,
    DevelopmentStage,
    ProgramCashFlowInputs,
    calculate_cashflow,
    delay_launch,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class TriangularRange(_FrozenModel):
    low: float
    mode: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> TriangularRange:
        if not all(math.isfinite(value) for value in (self.low, self.mode, self.high)):
            raise ValueError("triangular range values must be finite")
        if not self.low <= self.mode <= self.high:
            raise ValueError("triangular range must satisfy low <= mode <= high")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return self.low
        return float(rng.triangular(self.low, self.mode, self.high))


class SimulationAssumptions(_FrozenModel):
    """Shared draws preserve correlation between the initial and expansion indications."""

    price_multiplier: TriangularRange = TriangularRange(low=0.75, mode=1.0, high=1.25)
    patient_multiplier: TriangularRange = TriangularRange(low=0.7, mode=1.0, high=1.3)
    gross_to_net_shift: TriangularRange = TriangularRange(low=-0.05, mode=0.0, high=0.1)
    persistence_multiplier: TriangularRange = TriangularRange(low=0.85, mode=1.0, high=1.05)
    development_cost_multiplier: TriangularRange = TriangularRange(low=0.85, mode=1.0, high=1.25)
    launch_delay_years: TriangularRange = TriangularRange(low=0.0, mode=0.0, high=2.0)
    loe_retention_multiplier: TriangularRange = TriangularRange(low=0.75, mode=1.0, high=1.25)

    @model_validator(mode="after")
    def validate_economic_ranges(self) -> SimulationAssumptions:
        nonnegative_multipliers = {
            "price_multiplier": self.price_multiplier,
            "patient_multiplier": self.patient_multiplier,
            "persistence_multiplier": self.persistence_multiplier,
            "development_cost_multiplier": self.development_cost_multiplier,
            "loe_retention_multiplier": self.loe_retention_multiplier,
        }
        for name, value in nonnegative_multipliers.items():
            if value.low < 0:
                raise ValueError(f"{name} cannot include negative values")
        if self.launch_delay_years.low < 0:
            raise ValueError("launch_delay_years cannot include negative values")
        return self


class DistributionSummary(_FrozenModel):
    mean: float
    p10: float
    p50: float
    p90: float
    probability_positive: float | None = None


class SimulationResult(_FrozenModel):
    seed: int
    simulations: int
    rng_bit_generator: str
    numpy_version: str
    draw_order_contract_version: str
    commercial_driver_correlation: str
    rnpv: DistributionSummary
    protected_net_revenue: DistributionSummary
    post_loe_net_revenue: DistributionSummary
    peak_annual_net_revenue: DistributionSummary
    peak_cash_at_risk: DistributionSummary
    effective_protected_years: DistributionSummary


def _summary(values: list[float], *, probability_positive: bool = False) -> DistributionSummary:
    array = np.asarray(values, dtype=float)
    return DistributionSummary(
        mean=float(np.mean(array)),
        p10=float(np.percentile(array, 10)),
        p50=float(np.percentile(array, 50)),
        p90=float(np.percentile(array, 90)),
        probability_positive=float(np.mean(array > 0)) if probability_positive else None,
    )


def _sample_development_path(
    inputs: ProgramCashFlowInputs,
    rng: np.random.Generator,
) -> DevelopmentRealization:
    def run(
        stages: tuple[DevelopmentStage, ...],
        *,
        alive: bool,
    ) -> tuple[bool, dict[int, float]]:
        costs: dict[int, float] = {}
        for stage in sorted(stages, key=lambda item: item.year):
            if not alive:
                break
            costs[stage.year] = costs.get(stage.year, 0.0) + stage.cost
            alive = bool(rng.random() < stage.success_probability)
        return alive, costs

    initial_success, initial_costs = run(inputs.initial_development_stages, alive=True)
    expansion_success = False
    expansion_costs: dict[int, float] = {}
    if inputs.expansion:
        expansion_start = (
            initial_success if inputs.expansion.conditional_on_initial_success else True
        )
        expansion_success, expansion_costs = run(
            inputs.expansion.development_stages, alive=expansion_start
        )
    return DevelopmentRealization(
        initial_success=initial_success,
        expansion_success=expansion_success,
        initial_development_costs_by_year=initial_costs,
        expansion_development_costs_by_year=expansion_costs,
    )


def _scaled_stages(
    stages: tuple[DevelopmentStage, ...], multiplier: float
) -> tuple[DevelopmentStage, ...]:
    return tuple(stage.model_copy(update={"cost": stage.cost * multiplier}) for stage in stages)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _perturb_inputs(
    inputs: ProgramCashFlowInputs,
    rng: np.random.Generator,
    assumptions: SimulationAssumptions,
) -> ProgramCashFlowInputs:
    price_multiplier = assumptions.price_multiplier.sample(rng)
    patient_multiplier = assumptions.patient_multiplier.sample(rng)
    gtn_shift = assumptions.gross_to_net_shift.sample(rng)
    persistence_multiplier = assumptions.persistence_multiplier.sample(rng)
    development_multiplier = assumptions.development_cost_multiplier.sample(rng)
    delay = round(assumptions.launch_delay_years.sample(rng))
    loe_multiplier = assumptions.loe_retention_multiplier.sample(rng)

    def perturb_indication(indication):
        return indication.model_copy(
            update={
                "backlog_patients": indication.backlog_patients * patient_multiplier,
                "annual_incident_patients": indication.annual_incident_patients
                * patient_multiplier,
                "annual_gross_price": indication.annual_gross_price * price_multiplier,
                "gross_to_net_rate": _clip(indication.gross_to_net_rate + gtn_shift),
                "annual_persistence_rate": _clip(
                    indication.annual_persistence_rate * persistence_multiplier
                ),
            }
        )

    initial = perturb_indication(inputs.initial_indication)
    expansion = inputs.expansion
    if expansion:
        expansion = expansion.model_copy(
            update={
                "indication": perturb_indication(expansion.indication),
                "development_stages": _scaled_stages(
                    expansion.development_stages, development_multiplier
                ),
            }
        )
    perturbed = inputs.model_copy(
        update={
            "initial_indication": initial,
            "initial_development_stages": _scaled_stages(
                inputs.initial_development_stages, development_multiplier
            ),
            "expansion": expansion,
            "loe_price_retention": tuple(
                _clip(value * loe_multiplier) for value in inputs.loe_price_retention
            ),
            "loe_volume_retention": tuple(
                _clip(value * loe_multiplier) for value in inputs.loe_volume_retention
            ),
        }
    )
    return delay_launch(perturbed, delay) if delay else perturbed


def simulate_program(
    inputs: ProgramCashFlowInputs,
    *,
    simulations: int = 1_000,
    seed: int = 0,
    assumptions: SimulationAssumptions | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SimulationResult:
    """Return seeded P10/P50/P90 distributions across commercial and development risk."""

    if simulations < 1:
        raise ValueError("simulations must be positive")
    assumptions = assumptions or SimulationAssumptions()
    rng = np.random.default_rng(seed)
    npvs: list[float] = []
    protected_revenue: list[float] = []
    post_loe_revenue: list[float] = []
    peak_annual_net_revenue: list[float] = []
    cash_at_risk: list[float] = []
    protected_years: list[float] = []
    for index in range(simulations):
        perturbed = _perturb_inputs(inputs, rng, assumptions)
        realization = _sample_development_path(perturbed, rng)
        result = calculate_cashflow(
            perturbed,
            realization=realization,
            calculate_delay_cost=False,
        )
        npvs.append(result.npv)
        protected_revenue.append(result.value_decomposition.protected_net_revenue)
        post_loe_revenue.append(result.value_decomposition.post_loe_net_revenue)
        peak_annual_net_revenue.append(result.peak_annual_net_revenue)
        cash_at_risk.append(result.peak_cash_at_risk)
        protected_years.append(result.effective_protected_years)
        if progress:
            progress(index + 1, simulations)
    return SimulationResult(
        seed=seed,
        simulations=simulations,
        rng_bit_generator=rng.bit_generator.__class__.__name__,
        numpy_version=np.__version__,
        draw_order_contract_version="1.0.0",
        commercial_driver_correlation="one shared draw per driver across indications",
        rnpv=_summary(npvs, probability_positive=True),
        protected_net_revenue=_summary(protected_revenue),
        post_loe_net_revenue=_summary(post_loe_revenue),
        peak_annual_net_revenue=_summary(peak_annual_net_revenue),
        peak_cash_at_risk=_summary(cash_at_risk),
        effective_protected_years=_summary(protected_years),
    )
