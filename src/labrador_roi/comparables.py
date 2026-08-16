"""Transparent, non-compensatory matching of therapeutic pricing comparables."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from labrador_roi.models import ComparableTherapy, ProgramInput


class ComparableTier(StrEnum):
    """Permitted roles for a comparable in the pricing analysis."""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    CONTEXT = "CONTEXT"
    EXCLUDED = "EXCLUDED"


class ComparableMatchComponents(BaseModel):
    """Inspectable match components; no single score can hide a failed core gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    therapeutic_area: float = Field(ge=0, le=1)
    indication: float = Field(ge=0, le=1)
    target_population: float = Field(ge=0, le=1)
    line_of_therapy: float = Field(ge=0, le=1)
    geography: float = Field(ge=0, le=1)
    route: float = Field(ge=0, le=1)
    target_or_mechanism: float = Field(ge=0, le=1)

    def weighted_score(self) -> float:
        """Descriptive ranking only; tier assignment uses non-compensatory gates below."""

        weights = {
            "therapeutic_area": 0.10,
            "indication": 0.25,
            "target_population": 0.15,
            "line_of_therapy": 0.20,
            "geography": 0.15,
            "route": 0.10,
            "target_or_mechanism": 0.05,
        }
        return sum(getattr(self, name) * weight for name, weight in weights.items())


class ComparableAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    comparable: ComparableTherapy
    components: ComparableMatchComponents
    tier: ComparableTier
    score: float = Field(ge=0, le=1)
    usable_for_same_basis_anchor: bool
    reasons: list[str] = Field(default_factory=list)


class ComparableSet(BaseModel):
    """Raw comparable catalog accepted from JSON/YAML fixtures and the CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparables: list[ComparableTherapy]

    @model_validator(mode="after")
    def validate_unique_records(self) -> ComparableSet:
        ids = [item.comparable_id for item in self.comparables]
        if len(ids) != len(set(ids)):
            raise ValueError("comparable_id values must be unique")

        fingerprints: dict[tuple[object, ...], str] = {}
        for item in self.comparables:
            price = item.price
            fingerprint = (
                item.name.casefold().strip(),
                item.therapeutic_area.casefold().strip(),
                item.indication.casefold().strip(),
                item.target_population.casefold().strip(),
                item.line_of_therapy.casefold().strip(),
                item.geography.casefold().strip(),
                item.route,
                (item.target_or_mechanism or "").casefold().strip(),
                price.amount,
                price.currency,
                price.basis,
                price.period,
                price.price_year,
                price.units_per_year,
            )
            prior_id = fingerprints.get(fingerprint)
            if prior_id is not None:
                raise ValueError(
                    "exact duplicate comparable product/price fingerprint for "
                    f"'{prior_id}' and '{item.comparable_id}'"
                )
            fingerprints[fingerprint] = item.comparable_id
        return self


class ComparableSelection(BaseModel):
    """Assessed analogues for one program indication, including excluded records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indication_id: str
    assessments: list[ComparableAssessment]

    def by_tier(self, tier: ComparableTier) -> list[ComparableAssessment]:
        return [item for item in self.assessments if item.tier == tier]

    def anchor_candidates(self) -> list[ComparableAssessment]:
        return [item for item in self.assessments if item.usable_for_same_basis_anchor]


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _match(left: str | None, right: str | None) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens == right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def assess_comparable(
    program: ProgramInput,
    comparable: ComparableTherapy,
    indication_id: str | None = None,
) -> ComparableAssessment:
    """Assess an analogue without allowing a target match to erase a PICO mismatch."""

    indication = program.indication(indication_id)
    explicitly_selected = (
        not indication.comparator_ids or comparable.comparable_id in indication.comparator_ids
    )
    route = indication.route or program.route
    components = ComparableMatchComponents(
        therapeutic_area=_match(indication.therapeutic_area, comparable.therapeutic_area),
        indication=_match(indication.name, comparable.indication),
        target_population=_match(indication.target_population, comparable.target_population),
        line_of_therapy=_match(indication.line_of_therapy, comparable.line_of_therapy),
        geography=_match(indication.geography, comparable.geography),
        route=1.0 if route == comparable.route else 0.0,
        target_or_mechanism=_match(program.target, comparable.target_or_mechanism),
    )

    core_exact = (
        components.indication == 1
        and components.target_population == 1
        and components.line_of_therapy == 1
        and components.geography == 1
        and components.route == 1
    )
    secondary_gate = (
        components.indication == 1
        and components.geography == 1
        and (
            components.line_of_therapy == 1
            or components.target_population >= 0.5
            or components.route == 1
        )
    )
    context_gate = (
        components.therapeutic_area >= 0.5
        or components.indication >= 0.5
        or components.target_or_mechanism == 1
    )

    if not explicitly_selected:
        tier = ComparableTier.EXCLUDED
    elif core_exact:
        tier = ComparableTier.PRIMARY
    elif secondary_gate:
        tier = ComparableTier.SECONDARY
    elif context_gate:
        tier = ComparableTier.CONTEXT
    else:
        tier = ComparableTier.EXCLUDED

    reasons: list[str] = []
    labels = {
        "therapeutic_area": "therapeutic area",
        "indication": "indication",
        "target_population": "target population",
        "line_of_therapy": "line of therapy",
        "geography": "geography",
        "route": "route",
    }
    for field_name, label in labels.items():
        if getattr(components, field_name) < 1:
            reasons.append(f"{label} is not an exact match")
    if not explicitly_selected:
        reasons.append("comparable is not in the indication's explicit comparator_ids allowlist")
    if comparable.price.currency != indication.currency:
        reasons.append(
            "price currency differs from indication currency; no FX conversion was supplied"
        )

    usable = explicitly_selected and tier in {ComparableTier.PRIMARY, ComparableTier.SECONDARY}
    usable = usable and comparable.price.currency == indication.currency
    return ComparableAssessment(
        comparable=comparable,
        components=components,
        tier=tier,
        score=components.weighted_score(),
        usable_for_same_basis_anchor=usable,
        reasons=reasons,
    )


def select_comparables(
    program: ProgramInput,
    comparables: ComparableSet | list[ComparableTherapy],
    indication_id: str | None = None,
) -> ComparableSelection:
    """Return all analogues sorted by explicit role and descriptive match score."""

    indication = program.indication(indication_id)
    tier_order = {
        ComparableTier.PRIMARY: 0,
        ComparableTier.SECONDARY: 1,
        ComparableTier.CONTEXT: 2,
        ComparableTier.EXCLUDED: 3,
    }
    catalog = (
        comparables
        if isinstance(comparables, ComparableSet)
        else ComparableSet(comparables=comparables)
    )
    raw_comparables = catalog.comparables
    assessments = [
        assess_comparable(program, item, indication.indication_id) for item in raw_comparables
    ]
    assessments.sort(key=lambda item: (tier_order[item.tier], -item.score, item.comparable.name))
    return ComparableSelection(indication_id=indication.indication_id, assessments=assessments)
