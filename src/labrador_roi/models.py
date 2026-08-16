"""Typed, provenance-aware contracts for LABrador's screening models.

The contracts deliberately keep observed data, analyst assumptions, and synthetic demo data
distinct.  They are permissive enough for early programs, but they never fill missing evidence
with a population, income, price, or probability default.
"""

from __future__ import annotations

import math
import re
from datetime import date
from enum import StrEnum
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceGrade(StrEnum):
    """Strength of evidence supporting an input."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"
    UNSUPPORTED = "UNSUPPORTED"
    SYNTHETIC = "SYNTHETIC"


class EvidenceType(StrEnum):
    PRIMARY_RESEARCH = "PRIMARY_RESEARCH"
    REGULATORY = "REGULATORY"
    PAYER_OR_HTA = "PAYER_OR_HTA"
    REAL_WORLD = "REAL_WORLD"
    SECONDARY_RESEARCH = "SECONDARY_RESEARCH"
    EXPERT_ELICITATION = "EXPERT_ELICITATION"
    INTERNAL = "INTERNAL"
    ASSUMPTION = "ASSUMPTION"
    SYNTHETIC = "SYNTHETIC"
    UNSUPPORTED = "UNSUPPORTED"


class EvidenceMetadata(BaseModel):
    """Provenance attached to a price, assumption, or observed value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str | None = None
    source_url: str | None = None
    citation: str | None = None
    source_date: date | None = None
    accessed_at: date | None = None
    evidence_type: EvidenceType = EvidenceType.UNSUPPORTED
    grade: EvidenceGrade = EvidenceGrade.UNSUPPORTED
    synthetic: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def validate_source_and_synthetic_status(self) -> EvidenceMetadata:
        if self.synthetic != (self.grade == EvidenceGrade.SYNTHETIC):
            raise ValueError("synthetic must be true exactly when grade is SYNTHETIC")
        if self.synthetic != (self.evidence_type == EvidenceType.SYNTHETIC):
            raise ValueError("synthetic must be true exactly when evidence_type is SYNTHETIC")
        has_source = bool(self.source_id or self.source_url or self.citation)
        grades_without_source_requirement = {
            EvidenceGrade.UNSUPPORTED,
            EvidenceGrade.SYNTHETIC,
        }
        if self.grade not in grades_without_source_requirement and not has_source:
            raise ValueError("non-synthetic supported evidence requires a source identifier")
        return self

    @property
    def supports_decision(self) -> bool:
        """Whether this evidence is strong enough to support a critical screening input."""

        excluded_types = {
            EvidenceType.UNSUPPORTED,
            EvidenceType.ASSUMPTION,
            EvidenceType.SYNTHETIC,
        }
        return (
            not self.synthetic
            and self.grade in {EvidenceGrade.HIGH, EvidenceGrade.MODERATE}
            and self.evidence_type not in excluded_types
        )


class DecisionGrade(StrEnum):
    DECISION_GRADE = "DECISION_GRADE"
    NOT_DECISION_GRADE = "NOT_DECISION_GRADE"


class PriceBasis(StrEnum):
    """Price bases that must never be silently combined."""

    LIST = "LIST"
    PUBLIC_REIMBURSEMENT = "PUBLIC_REIMBURSEMENT"
    ESTIMATED_NET = "ESTIMATED_NET"
    OBSERVED_NET = "OBSERVED_NET"


class PricePeriod(StrEnum):
    ANNUAL = "ANNUAL"
    MONTH = "MONTH"
    COURSE = "COURSE"
    UNIT = "UNIT"


class RouteOfAdministration(StrEnum):
    ORAL = "ORAL"
    SUBCUTANEOUS_SELF = "SUBCUTANEOUS_SELF"
    SUBCUTANEOUS_CLINIC = "SUBCUTANEOUS_CLINIC"
    INTRAMUSCULAR = "INTRAMUSCULAR"
    INTRAVENOUS = "INTRAVENOUS"
    OTHER = "OTHER"


class PayerType(StrEnum):
    PUBLIC_SINGLE_PAYER = "PUBLIC_SINGLE_PAYER"
    PUBLIC_MULTIPAYER = "PUBLIC_MULTIPAYER"
    COMMERCIAL_INSURANCE = "COMMERCIAL_INSURANCE"
    MIXED = "MIXED"
    TENDER = "TENDER"
    CASH_PAY = "CASH_PAY"
    UNKNOWN = "UNKNOWN"


class Modality(StrEnum):
    """Supported therapeutic modalities without implicit economic assumptions."""

    SMALL_MOLECULE = "SMALL_MOLECULE"
    PEPTIDE = "PEPTIDE"
    ANTIBODY = "ANTIBODY"


class WarningSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class WarningRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    severity: WarningSeverity = WarningSeverity.WARNING
    field: str | None = None


class CalculationStep(BaseModel):
    """One reproducible step in a model result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    label: str
    formula: str
    inputs: dict[str, Any]
    result: float | None
    unit: str
    evidence_keys: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PopulationInput(BaseModel):
    """Population stock and flow for one indication and treatment line."""

    model_config = ConfigDict(extra="forbid")

    eligible_patients: float | None = Field(default=None, ge=0)
    prevalent_backlog_patients: float | None = Field(default=None, ge=0)
    annual_incident_patients: float | None = Field(default=None, ge=0)
    diagnosed_fraction: float | None = Field(default=None, ge=0, le=1)
    clinically_eligible_fraction: float | None = Field(default=None, ge=0, le=1)
    overlap_with_initial_fraction: float | None = Field(default=None, ge=0, le=1)
    cannibalization_fraction: float | None = Field(default=None, ge=0, le=1)
    evidence: dict[str, EvidenceMetadata] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)


class AccessAssumptions(BaseModel):
    """Health-system access factors, distinct from patient income affordability."""

    model_config = ConfigDict(extra="forbid")

    payer_type: PayerType = PayerType.UNKNOWN
    universal_or_public_coverage: bool = False
    coverage_fraction: float | None = Field(default=None, ge=0, le=1)
    prior_authorization_pass_fraction: float | None = Field(default=None, ge=0, le=1)
    initiation_fraction: float | None = Field(default=None, ge=0, le=1)
    provider_capacity_fraction: float | None = Field(default=None, ge=0, le=1)
    patient_cost_share_fraction: float | None = Field(default=None, ge=0, le=1)
    annual_patient_oop: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Observed or explicitly modeled annual patient out-of-pocket amount. "
            "This is distinct from manufacturer net revenue."
        ),
    )
    adoption_by_year: dict[int, float] = Field(default_factory=dict)
    restrictions: list[str] = Field(default_factory=list)
    evidence: dict[str, EvidenceMetadata] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("adoption_by_year")
    @classmethod
    def validate_adoption_curve(cls, value: dict[int, float]) -> dict[int, float]:
        if any(year < 0 for year in value):
            raise ValueError("adoption year offsets must be non-negative")
        if any(fraction < 0 or fraction > 1 for fraction in value.values()):
            raise ValueError("adoption fractions must be between 0 and 1")
        return value

    def system_access_fraction(self, year_offset: int = 0) -> float | None:
        """Multiply known health-system gates; return unknown rather than inventing a default."""

        values = [
            self.coverage_fraction,
            self.prior_authorization_pass_fraction,
            self.initiation_fraction,
            self.provider_capacity_fraction,
        ]
        adoption = self.adoption_fraction(year_offset)
        if adoption is not None:
            values.append(adoption)
        if any(value is None for value in values):
            return None
        result = 1.0
        for value in values:
            assert value is not None
            result *= value
        return result

    def adoption_fraction(self, year_offset: int) -> float | None:
        """Return the supplied adoption path, interpolating gaps and holding its last value."""

        if not self.adoption_by_year:
            return None
        if year_offset in self.adoption_by_year:
            return self.adoption_by_year[year_offset]
        offsets = sorted(self.adoption_by_year)
        if year_offset < offsets[0]:
            return 0.0
        if year_offset > offsets[-1]:
            return self.adoption_by_year[offsets[-1]]
        upper_index = next(index for index, offset in enumerate(offsets) if offset > year_offset)
        lower_offset = offsets[upper_index - 1]
        upper_offset = offsets[upper_index]
        lower_value = self.adoption_by_year[lower_offset]
        upper_value = self.adoption_by_year[upper_offset]
        weight = (year_offset - lower_offset) / (upper_offset - lower_offset)
        return lower_value * (1 - weight) + upper_value * weight


class IncomeBandAssumptions(BaseModel):
    """Patient affordability inputs; never an input to clinical value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    population_share: float = Field(ge=0, le=1)
    annual_income: float | None = Field(default=None, ge=0)
    maximum_oop_share: float | None = Field(default=None, ge=0, le=1)
    evidence: EvidenceMetadata


class PatentAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filing_year: int
    base_term_years: int = Field(default=20, ge=1)
    extension_years: float = Field(default=0, ge=0, le=5)
    regulatory_exclusivity_end_year: float | None = None
    evidence: dict[str, EvidenceMetadata] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_regulatory_exclusivity_date(self) -> PatentAssumptions:
        if (
            self.regulatory_exclusivity_end_year is not None
            and self.regulatory_exclusivity_end_year < self.filing_year
        ):
            raise ValueError("regulatory exclusivity cannot end before patent filing")
        return self

    @property
    def patent_expiry_year(self) -> float:
        return self.filing_year + self.base_term_years + self.extension_years

    @property
    def effective_exclusivity_end_year(self) -> float:
        """Later of modeled patent expiry and an explicitly supplied regulatory exclusivity."""

        if self.regulatory_exclusivity_end_year is None:
            return self.patent_expiry_year
        return max(self.patent_expiry_year, self.regulatory_exclusivity_end_year)


_STAGE_RANKS = {
    "discovery": 10,
    "lead_optimization": 20,
    "preclinical": 30,
    "phase_1": 40,
    "phase_1b": 45,
    "phase_2": 50,
    "phase_2b": 55,
    "phase_3": 60,
    "filing": 70,
    "submission": 70,
    "nda": 70,
    "bla": 70,
    "maa": 70,
    "approval": 80,
}


def _normalized_stage_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def development_stage_sort_key(name: str) -> tuple[int, str]:
    """Stable lifecycle ordering for legacy mappings that predate explicit ``stage_order``."""

    normalized = _normalized_stage_name(name)
    return (_STAGE_RANKS.get(normalized, 1_000), normalized)


class DevelopmentAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_stage: str = Field(min_length=1)
    stage_costs: dict[str, float] = Field(default_factory=dict)
    stage_durations_years: dict[str, float] = Field(default_factory=dict)
    stage_success_probabilities: dict[str, float] = Field(default_factory=dict)
    stage_order: list[str] = Field(default_factory=list)
    program_probability_of_approval: float | None = Field(default=None, ge=0, le=1)
    evidence: dict[str, EvidenceMetadata] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "stage_costs",
        "stage_durations_years",
        "stage_success_probabilities",
        mode="before",
    )
    @classmethod
    def reject_boolean_mapping_values(cls, value: Any) -> Any:
        if isinstance(value, dict) and any(isinstance(item, bool) for item in value.values()):
            raise ValueError("development mapping values cannot be boolean")
        return value

    @field_validator("program_probability_of_approval", mode="before")
    @classmethod
    def reject_boolean_aggregate_probability(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("program approval probability cannot be boolean")
        return value

    @field_validator("stage_costs", "stage_durations_years")
    @classmethod
    def validate_nonnegative_mapping(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(item) or item < 0 for item in value.values()):
            raise ValueError("costs and durations must be finite and non-negative")
        return value

    @field_validator("stage_success_probabilities")
    @classmethod
    def validate_probability_mapping(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(item) or item < 0 or item > 1 for item in value.values()):
            raise ValueError("stage success probabilities must be finite and between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_stage_contract(self) -> DevelopmentAssumptions:
        cost_names = set(self.stage_costs)
        if set(self.stage_durations_years) != cost_names:
            raise ValueError("stage durations must have exactly the same keys as stage costs")
        if set(self.stage_success_probabilities) != cost_names:
            raise ValueError(
                "stage success probabilities must have exactly the same keys as stage costs"
            )
        if self.stage_order:
            if len(self.stage_order) != len(set(self.stage_order)):
                raise ValueError("stage_order cannot contain duplicate stage names")
            if set(self.stage_order) != cost_names:
                raise ValueError("stage_order must contain every costed stage exactly once")
        ordered_names = (
            tuple(self.stage_order)
            if self.stage_order
            else tuple(sorted(self.stage_costs, key=development_stage_sort_key))
        )
        if ordered_names:
            if _normalized_stage_name(self.current_stage) != _normalized_stage_name(
                ordered_names[0]
            ):
                raise ValueError("current_stage must match the first modeled stage")
            known_ranks = [
                _STAGE_RANKS[normalized]
                for name in ordered_names
                if (normalized := _normalized_stage_name(name)) in _STAGE_RANKS
            ]
            if any(left > right for left, right in pairwise(known_ranks)):
                raise ValueError("known lifecycle stages must be chronological")
        return self

    def ordered_stage_names(self) -> tuple[str, ...]:
        """Return an explicit order or a stable lifecycle order for legacy inputs."""

        if self.stage_order:
            return tuple(self.stage_order)
        return tuple(sorted(self.stage_costs, key=development_stage_sort_key))

    @property
    def is_post_approval(self) -> bool:
        """Whether no remaining development cost path is expected for the declared stage."""

        return _normalized_stage_name(self.current_stage) in {
            "approval",
            "approved",
            "launched",
            "marketed",
        }


class IndicationInput(BaseModel):
    """One label opportunity; expansions do not alter the asset patent clock."""

    model_config = ConfigDict(extra="forbid")

    indication_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    therapeutic_area: str = Field(min_length=1)
    target_population: str = Field(min_length=1)
    line_of_therapy: str = Field(min_length=1)
    geography: str = Field(min_length=1)
    currency: str = Field(min_length=3, max_length=3)
    launch_year: int
    severity: str | None = None
    biomarker: str | None = None
    route: RouteOfAdministration | None = None
    comparator_ids: list[str] = Field(default_factory=list)
    population: PopulationInput
    access: AccessAssumptions
    income_bands: list[IncomeBandAssumptions] = Field(default_factory=list)
    evidence: dict[str, EvidenceMetadata] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("currency must be a three-letter alphabetic code")
        return value.upper()

    @model_validator(mode="after")
    def validate_income_band_shares(self) -> IndicationInput:
        total_share = sum(band.population_share for band in self.income_bands)
        if total_share > 1.000001:
            raise ValueError("income-band population shares cannot exceed 1")
        return self


class ProgramInput(BaseModel):
    """Flexible asset contract shared by pricing and downstream cash-flow modules."""

    model_config = ConfigDict(extra="forbid")

    program_id: str = Field(min_length=1)
    program_name: str = Field(min_length=1)
    target: str = Field(min_length=1)
    modality: Modality
    molecule_identifier: str | None = None
    route: RouteOfAdministration
    base_year: int
    valuation_year: int
    currency: str = Field(min_length=3, max_length=3)
    initial_indication: IndicationInput
    expansion_indications: list[IndicationInput] = Field(default_factory=list)
    patent: PatentAssumptions
    development: DevelopmentAssumptions
    evidence: dict[str, EvidenceMetadata] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("modality", mode="before")
    @classmethod
    def normalize_modality(cls, value: Any) -> Any:
        """Accept case-insensitive serialized values while retaining one canonical enum."""

        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("currency")
    @classmethod
    def normalize_program_currency(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("currency must be a three-letter alphabetic code")
        return value.upper()

    @model_validator(mode="after")
    def validate_indication_contract(self) -> ProgramInput:
        indications = [self.initial_indication, *self.expansion_indications]
        ids = [item.indication_id for item in indications]
        if len(ids) != len(set(ids)):
            raise ValueError("indication IDs must be unique within a program")
        if any(item.currency != self.currency for item in indications):
            raise ValueError("indication and program currencies must match")
        if self.patent.filing_year > self.initial_indication.launch_year:
            raise ValueError("the modeled patent filing cannot occur after initial launch")
        if any(
            item.launch_year < self.initial_indication.launch_year
            for item in self.expansion_indications
        ):
            raise ValueError("expansion launch cannot precede initial-indication launch")
        return self

    def indication(self, indication_id: str | None = None) -> IndicationInput:
        if indication_id is None or indication_id == self.initial_indication.indication_id:
            return self.initial_indication
        for item in self.expansion_indications:
            if item.indication_id == indication_id:
                return item
        raise KeyError(f"unknown indication_id: {indication_id}")


class PriceObservation(BaseModel):
    """A price with an explicit, non-interchangeable basis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    basis: PriceBasis
    period: PricePeriod = PricePeriod.ANNUAL
    price_year: int
    units_per_year: float | None = Field(default=None, gt=0)
    evidence: EvidenceMetadata

    @field_validator("currency")
    @classmethod
    def normalize_price_currency(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("currency must be a three-letter alphabetic code")
        return value.upper()

    @model_validator(mode="after")
    def require_annualization_units(self) -> PriceObservation:
        if self.period in {PricePeriod.COURSE, PricePeriod.UNIT} and self.units_per_year is None:
            raise ValueError("COURSE and UNIT prices require units_per_year")
        return self

    def annualized_amount(self) -> float:
        if self.period == PricePeriod.ANNUAL:
            return self.amount
        if self.period == PricePeriod.MONTH:
            return self.amount * 12
        assert self.units_per_year is not None
        return self.amount * self.units_per_year


class ComparableTherapy(BaseModel):
    """One market analogue with explicit clinical placement and price provenance."""

    model_config = ConfigDict(extra="forbid")

    comparable_id: str
    name: str
    therapeutic_area: str
    indication: str
    target_population: str
    line_of_therapy: str
    geography: str
    route: RouteOfAdministration
    price: PriceObservation
    target_or_mechanism: str | None = None
    payer_type: PayerType = PayerType.UNKNOWN
    access_restrictions: list[str] = Field(default_factory=list)
    evidence: dict[str, EvidenceMetadata] = Field(default_factory=dict)
    notes: str | None = None
