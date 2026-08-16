"""Shared LABrador interpretability contract and rNPV/ROI adapter.

The adapter is intentionally downstream of :class:`AnalysisResult`: it maps the
authoritative engine result into a UI-oriented contract without re-running any
scientific or economic calculation.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from labrador_roi.engine import AnalysisResult

INTERPRETABILITY_SCHEMA_VERSION = "1.0.0"

HeadlineStatus = Literal[
    "SUPPORTED",
    "QUALIFIED",
    "INCONCLUSIVE",
    "FAILED",
    "NOT_APPLICABLE",
]
HeadlineBasis = Literal["OBSERVED", "INFERRED", "MODELED", "SYNTHETIC"]
MetricDirection = Literal["positive", "negative", "neutral", "mixed", "unknown"]
EvidenceGrade = Literal["HIGH", "MODERATE", "LOW", "UNSUPPORTED"]
LimitationSeverity = Literal["INFO", "WARNING", "ERROR"]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InterpretabilityHeadline(_ContractModel):
    title: str = Field(min_length=1)
    result: str = Field(min_length=1)
    plain_language: str = Field(min_length=1)
    status: HeadlineStatus
    basis: tuple[HeadlineBasis, ...] = Field(min_length=1)


class InterpretabilityMetric(_ContractModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    label: str = Field(min_length=1)
    value: JsonValue
    unit: str = Field(min_length=1)
    display: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    direction: MetricDirection
    evidence_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]


class InterpretabilityStepInput(_ContractModel):
    path: str = Field(min_length=1)
    value: JsonValue
    unit: str | None


class InterpretabilityStepResult(_ContractModel):
    value: JsonValue
    unit: str = Field(min_length=1)


class InterpretabilityStep(_ContractModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    label: str = Field(min_length=1)
    method: str = Field(min_length=1)
    formula: str | None
    inputs: tuple[InterpretabilityStepInput, ...]
    result: InterpretabilityStepResult
    evidence_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]


class InterpretabilityEvidence(_ContractModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    claim: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_id: str | None
    source_url: str | None
    locator: str | None
    quote: str | None
    grade: EvidenceGrade
    synthetic: bool


class InterpretabilityAssumption(_ContractModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    path: str = Field(min_length=1)
    value: JsonValue
    unit: str | None
    basis: str = Field(min_length=1)
    synthetic: bool | None


class InterpretabilityInterval(_ContractModel):
    metric_id: str = Field(min_length=1)
    low: float | int | None
    central: float | int | None
    high: float | int | None
    unit: str = Field(min_length=1)
    confidence_level: float | None


class InterpretabilityUncertainty(_ContractModel):
    method: str = Field(min_length=1)
    intervals: tuple[InterpretabilityInterval, ...]
    seed: int | None
    draws: int | None
    limitations: tuple[str, ...]


class InterpretabilityLimitation(_ContractModel):
    code: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    severity: LimitationSeverity
    message: str = Field(min_length=1)
    field_path: str | None


class InterpretabilityCounterfactual(_ContractModel):
    change: str = Field(min_length=1)
    result: str = Field(min_length=1)
    meaning: str = Field(min_length=1)


class InterpretabilityLineage(_ContractModel):
    output_path: str = Field(min_length=1)
    input_paths: tuple[str, ...]
    transformation: str = Field(min_length=1)


class Interpretability(_ContractModel):
    """Common UI contract with runtime reference-integrity validation."""

    schema_version: Literal["1.0.0"]
    headline: InterpretabilityHeadline
    metrics: tuple[InterpretabilityMetric, ...]
    steps: tuple[InterpretabilityStep, ...]
    evidence: tuple[InterpretabilityEvidence, ...]
    assumptions: tuple[InterpretabilityAssumption, ...]
    uncertainty: InterpretabilityUncertainty
    limitations: tuple[InterpretabilityLimitation, ...]
    counterfactuals: tuple[InterpretabilityCounterfactual, ...]
    lineage: tuple[InterpretabilityLineage, ...]
    extensions: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_integrity(self) -> Interpretability:
        collections = {
            "metrics": [item.id for item in self.metrics],
            "steps": [item.id for item in self.steps],
            "evidence": [item.id for item in self.evidence],
            "assumptions": [item.id for item in self.assumptions],
        }
        for name, ids in collections.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} IDs must be unique")

        evidence_ids = set(collections["evidence"])
        assumption_ids = set(collections["assumptions"])
        metric_ids = set(collections["metrics"])
        for item in (*self.metrics, *self.steps):
            missing_evidence = set(item.evidence_ids) - evidence_ids
            missing_assumptions = set(item.assumption_ids) - assumption_ids
            if missing_evidence:
                raise ValueError(f"unresolved evidence IDs: {sorted(missing_evidence)}")
            if missing_assumptions:
                raise ValueError(f"unresolved assumption IDs: {sorted(missing_assumptions)}")
        for interval in self.uncertainty.intervals:
            if interval.metric_id not in metric_ids:
                raise ValueError(f"unresolved metric ID: {interval.metric_id}")

        untagged_paths = {
            item.field_path for item in self.limitations if item.code == "UNTAGGED_VALUE"
        }
        for metric in self.metrics:
            if not metric.evidence_ids and not metric.assumption_ids:
                expected_path = f"interpretability.metrics.{metric.id}"
                if expected_path not in untagged_paths:
                    raise ValueError(f"important metric {metric.id!r} is untagged")

        limitation_codes = {item.code for item in self.limitations}
        empty_collection_codes = {
            "metrics": "NO_METRICS",
            "steps": "NO_CALCULATION_STEPS",
            "evidence": "NO_EVIDENCE_REFERENCES",
            "assumptions": "NO_ASSUMPTIONS",
            "counterfactuals": "NO_COUNTERFACTUALS",
            "lineage": "NO_LINEAGE",
        }
        for name, code in empty_collection_codes.items():
            if not getattr(self, name) and code not in limitation_codes:
                raise ValueError(f"empty {name} requires a {code} limitation")
        if not self.uncertainty.intervals and "NO_UNCERTAINTY_INTERVALS" not in limitation_codes:
            raise ValueError(
                "empty uncertainty intervals require a NO_UNCERTAINTY_INTERVALS limitation"
            )

        serialized = self.model_dump(mode="python")
        _reject_nonfinite(serialized)
        return self


def _reject_nonfinite(value: Any, path: str = "interpretability") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _escape_html(value: Any) -> tuple[Any, int]:
    """Escape HTML-looking input text while the native payload retains the raw value."""

    if isinstance(value, str):
        if re.search(r"<[^>]*>", value):
            return html.escape(value), 1
        return value, 0
    if isinstance(value, dict):
        escaped: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            safe_key, key_count = _escape_html(str(key))
            if key_count:
                safe_key = f"{safe_key}.{_digest(str(key))}"
            if str(safe_key) in escaped:
                safe_key = f"{safe_key}.{_digest(str(key))}"
            safe_item, item_count = _escape_html(item)
            escaped[str(safe_key)] = safe_item
            count += key_count + item_count
        return escaped, count
    if isinstance(value, (list, tuple)):
        escaped_items = []
        count = 0
        for item in value:
            safe_item, item_count = _escape_html(item)
            escaped_items.append(safe_item)
            count += item_count
        return escaped_items, count
    return value, 0


def _data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_data(item) for item in value]
    return value


def _contains_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_unknown(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unknown(item) for item in value)
    return False


def _digest(value: Any, length: int = 10) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _slug(value: Any, *, fallback: str = "item", limit: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9]+", ".", str(value).casefold()).strip(".")
    return (normalized or fallback)[:limit].rstrip(".")


def _stable_id(prefix: str, identity: Any, label: Any) -> str:
    return f"{prefix}.{_slug(label)}.{_digest(identity)}"


def _common_step_id(native_step_id: str) -> str:
    return _stable_id("step", native_step_id, native_step_id)


def _indication_metric_id(prefix: str, indication_id: str) -> str:
    return f"{prefix}.{_slug(indication_id)}.{_digest(indication_id)}"


def _income_band_segments(bands: list[Any]) -> list[str]:
    """Return order-independent, collision-safe identifiers for affordability bands."""

    occurrences: dict[str, int] = defaultdict(int)
    segments: list[str] = []
    for band in bands:
        if isinstance(band, dict):
            label = str(band.get("name") or "unnamed")
            base = f"name={label},key={_digest(band)}"
        else:
            base = f"item={_digest(band)}"
        occurrences[base] += 1
        segments.append(f"{base},occurrence={occurrences[base]}")
    return segments


def _canonical_evidence_path(path: str, snapshot: dict[str, Any]) -> str:
    """Replace known array positions with durable domain identifiers."""

    program = snapshot.get("program", {}) if isinstance(snapshot, dict) else {}
    if isinstance(program, dict):
        expansions = program.get("expansion_indications", [])
        if isinstance(expansions, list):
            for index, item in enumerate(expansions):
                if isinstance(item, dict) and item.get("indication_id"):
                    path = path.replace(
                        f"program.expansion_indications[{index}]",
                        f"program.expansion_indications[id={item['indication_id']}]",
                    )
        initial = program.get("initial_indication", {})
        indications = [initial, *(expansions if isinstance(expansions, list) else [])]
        prefixes = ["program.initial_indication"] + [
            f"program.expansion_indications[id={item.get('indication_id')}]"
            for item in expansions
            if isinstance(item, dict) and item.get("indication_id")
        ]
        for prefix, indication in zip(prefixes, indications, strict=False):
            if not isinstance(indication, dict):
                continue
            bands = indication.get("income_bands", [])
            if not isinstance(bands, list):
                continue
            for index, segment in enumerate(_income_band_segments(bands)):
                path = path.replace(
                    f"{prefix}.income_bands[{index}]",
                    f"{prefix}.income_bands[{segment}]",
                )
    comparables = snapshot.get("comparables", {}) if isinstance(snapshot, dict) else {}
    comparable_rows = comparables.get("comparables", []) if isinstance(comparables, dict) else []
    if isinstance(comparable_rows, list):
        for index, item in enumerate(comparable_rows):
            if isinstance(item, dict) and item.get("comparable_id"):
                path = path.replace(
                    f"comparables.comparables[{index}]",
                    f"comparables.comparables[id={item['comparable_id']}]",
                )
    return path


def _evidence_grade(native_grade: str) -> EvidenceGrade:
    return {
        "HIGH": "HIGH",
        "MODERATE": "MODERATE",
        "LOW": "LOW",
        "VERY_LOW": "UNSUPPORTED",
        "UNSUPPORTED": "UNSUPPORTED",
        "SYNTHETIC": "UNSUPPORTED",
    }.get(native_grade, "UNSUPPORTED")  # type: ignore[return-value]


def _build_evidence(
    result: AnalysisResult,
) -> tuple[
    tuple[InterpretabilityEvidence, ...],
    dict[str, str],
    list[dict[str, JsonValue]],
    list[InterpretabilityLimitation],
]:
    snapshot = _data(result.input_snapshot)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for reference in result.evidence_references:
        record = _data(reference)
        canonical_path = _canonical_evidence_path(str(record["field_path"]), snapshot)
        source_identity = (
            record.get("source_id") or record.get("source_url") or record.get("citation")
        )
        native_synthetic = bool(record["synthetic"])
        resolved_synthetic = native_synthetic or str(record["evidence_type"]) == "SYNTHETIC"
        resolved_synthetic = resolved_synthetic or str(record["grade"]) == "SYNTHETIC"
        identity = (
            source_identity or canonical_path,
            record["evidence_type"],
            record["grade"],
            resolved_synthetic,
        )
        record["canonical_field_path"] = canonical_path
        record["native_synthetic"] = native_synthetic
        record["resolved_synthetic"] = resolved_synthetic
        groups[identity].append(record)

    evidence: list[InterpretabilityEvidence] = []
    path_to_id: dict[str, str] = {}
    native_records: list[dict[str, JsonValue]] = []
    limitations: list[InterpretabilityLimitation] = []
    missing_source_id = False
    missing_source_url = False
    normalized_grades: set[str] = set()
    for identity in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        records = groups[identity]
        first = records[0]
        paths = sorted({str(item["canonical_field_path"]) for item in records})
        source_label = first.get("source_id") or first.get("source_url") or paths[0]
        evidence_id = _stable_id("evidence", identity, source_label)
        claim_paths = ", ".join(paths[:3])
        if len(paths) > 3:
            claim_paths += f", and {len(paths) - 3} more input fields"
        citation_suffix = f" Native citation: {first['citation']}." if first.get("citation") else ""
        native_grade = str(first["grade"])
        resolved_synthetic = bool(first["resolved_synthetic"])
        mapped_grade = "UNSUPPORTED" if resolved_synthetic else _evidence_grade(native_grade)
        if native_grade != mapped_grade:
            normalized_grades.add(native_grade)
        evidence.append(
            InterpretabilityEvidence(
                id=evidence_id,
                claim=(
                    f"Source metadata supplied for model input(s): {claim_paths}.{citation_suffix}"
                ),
                source_type=str(first["evidence_type"]).casefold(),
                source_id=first.get("source_id"),
                source_url=first.get("source_url"),
                locator=None,
                quote=None,
                grade=mapped_grade,
                synthetic=resolved_synthetic,
            )
        )
        missing_source_id = missing_source_id or first.get("source_id") is None
        missing_source_url = missing_source_url or first.get("source_url") is None
        for path in paths:
            path_to_id[path] = evidence_id
        native_records.append(
            {
                "evidence_id": evidence_id,
                "field_paths": paths,
                "native_grade": native_grade,
                "native_evidence_type": str(first["evidence_type"]),
                "native_synthetic": bool(first["native_synthetic"]),
                "shared_synthetic": resolved_synthetic,
                "citation": first.get("citation"),
            }
        )
        if any(bool(item["native_synthetic"]) != resolved_synthetic for item in records):
            limitations.append(
                InterpretabilityLimitation(
                    code="EVIDENCE_SYNTHETIC_STATUS_CONFLICT",
                    severity="ERROR",
                    message=(
                        f"Evidence {evidence_id} has a synthetic type or grade that conflicts "
                        "with synthetic=false; the shared evidence is conservatively synthetic."
                    ),
                    field_path=f"interpretability.evidence.{evidence_id}.synthetic",
                )
            )
        if resolved_synthetic:
            limitations.append(
                InterpretabilityLimitation(
                    code="SYNTHETIC_EVIDENCE",
                    severity="WARNING",
                    message=(
                        f"Evidence {evidence_id} is explicitly synthetic and cannot support a "
                        "decision-grade conclusion."
                    ),
                    field_path=f"interpretability.evidence.{evidence_id}",
                )
            )
        elif mapped_grade == "UNSUPPORTED":
            limitations.append(
                InterpretabilityLimitation(
                    code="UNSUPPORTED_EVIDENCE",
                    severity="WARNING",
                    message=f"Evidence {evidence_id} is explicitly unsupported.",
                    field_path=f"interpretability.evidence.{evidence_id}",
                )
            )

    if not evidence:
        limitations.append(
            InterpretabilityLimitation(
                code="NO_EVIDENCE_REFERENCES",
                severity="ERROR",
                message="The native result contains no evidence references.",
                field_path="payload.evidence_references",
            )
        )
    else:
        limitations.append(
            InterpretabilityLimitation(
                code="INTERPRETABILITY_PARTIAL",
                severity="WARNING",
                message=(
                    "The native evidence contract does not carry substantive claim text, verified "
                    "quotes, or page/section locators; claims identify mapped input scope and the "
                    "unavailable fields remain null."
                ),
                field_path="interpretability.evidence",
            )
        )
        if missing_source_id:
            limitations.append(
                InterpretabilityLimitation(
                    code="EVIDENCE_SOURCE_ID_UNAVAILABLE",
                    severity="INFO",
                    message="One or more evidence records do not provide a source identifier.",
                    field_path="interpretability.evidence[].source_id",
                )
            )
        if missing_source_url:
            limitations.append(
                InterpretabilityLimitation(
                    code="EVIDENCE_SOURCE_URL_UNAVAILABLE",
                    severity="INFO",
                    message="One or more evidence records do not provide a source URL.",
                    field_path="interpretability.evidence[].source_url",
                )
            )
        if normalized_grades:
            limitations.append(
                InterpretabilityLimitation(
                    code="EVIDENCE_GRADE_NORMALIZED",
                    severity="INFO",
                    message=(
                        "Native evidence grades outside the shared enum were conservatively "
                        f"normalized: {', '.join(sorted(normalized_grades))}."
                    ),
                    field_path="interpretability.evidence[].grade",
                )
            )
    return tuple(evidence), path_to_id, native_records, limitations


def _stable_list_segment(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("indication_id", "comparable_id", "id"):
            if value.get(key) not in (None, ""):
                return f"{key}={value[key]}"
        if value.get("name") not in (None, ""):
            return f"name={value['name']},key={_digest(value)}"
    return f"item={_digest(value)}"


def _iter_explicit_assumptions(
    value: Any, path: str = "input"
) -> Iterator[tuple[str, Any, bool | None]]:
    if isinstance(value, dict):
        assumptions = value.get("assumptions")
        if isinstance(assumptions, dict):
            explicit_synthetic = assumptions.get("synthetic")
            synthetic = explicit_synthetic if isinstance(explicit_synthetic, bool) else None
            for key in sorted(assumptions):
                yield f"{path}.assumptions.{key}", assumptions[key], synthetic
        for key in sorted(value):
            if key != "assumptions":
                yield from _iter_explicit_assumptions(value[key], f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            yield from _iter_explicit_assumptions(item, f"{path}[{_stable_list_segment(item)}]")


def _iter_material_input_assumptions(snapshot: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield typed economic inputs that are assumptions even when not in an assumptions map."""

    program = snapshot.get("program")
    if not isinstance(program, dict):
        return
    patent = program.get("patent")
    if isinstance(patent, dict):
        for key in (
            "filing_year",
            "base_term_years",
            "extension_years",
            "regulatory_exclusivity_end_year",
        ):
            if key in patent:
                yield f"input.program.patent.{key}", patent[key]
    development = program.get("development")
    if isinstance(development, dict):
        for key in (
            "current_stage",
            "stage_costs",
            "stage_durations_years",
            "stage_success_probabilities",
            "stage_order",
            "program_probability_of_approval",
        ):
            if key in development:
                yield f"input.program.development.{key}", development[key]

    indications: list[tuple[str, dict[str, Any]]] = []
    initial = program.get("initial_indication")
    if isinstance(initial, dict):
        indications.append(("input.program.initial_indication", initial))
    expansions = program.get("expansion_indications", [])
    if isinstance(expansions, list):
        for item in expansions:
            if isinstance(item, dict) and item.get("indication_id"):
                indications.append(
                    (
                        "input.program.expansion_indications"
                        f"[indication_id={item['indication_id']}]",
                        item,
                    )
                )
    for base, indication in indications:
        if "launch_year" in indication:
            yield f"{base}.launch_year", indication["launch_year"]
        population = indication.get("population")
        if isinstance(population, dict):
            for key in (
                "eligible_patients",
                "prevalent_backlog_patients",
                "annual_incident_patients",
                "diagnosed_fraction",
                "clinically_eligible_fraction",
                "overlap_with_initial_fraction",
                "cannibalization_fraction",
            ):
                if key in population:
                    yield f"{base}.population.{key}", population[key]
        access = indication.get("access")
        if isinstance(access, dict):
            for key in (
                "payer_type",
                "universal_or_public_coverage",
                "coverage_fraction",
                "prior_authorization_pass_fraction",
                "initiation_fraction",
                "provider_capacity_fraction",
                "patient_cost_share_fraction",
                "annual_patient_oop",
                "adoption_by_year",
                "restrictions",
            ):
                if key in access:
                    yield f"{base}.access.{key}", access[key]
        income_bands = indication.get("income_bands", [])
        if isinstance(income_bands, list):
            for band, segment in zip(
                income_bands, _income_band_segments(income_bands), strict=True
            ):
                if not isinstance(band, dict):
                    continue
                band_base = f"{base}.income_bands[{segment}]"
                for key in ("population_share", "annual_income", "maximum_oop_share"):
                    if key in band:
                        yield f"{band_base}.{key}", band[key]


def _unit_for_key(key: str, currency: str | None) -> str | None:
    normalized = key.casefold()
    fixed_units = {
        "adoption_ramp_years": "years",
        "annual_incident_patients": "patients/year",
        "annual_persistence_rate": "fraction",
        "annual_rows": "rows",
        "adoption_by_year": "fraction",
        "authorization_rate": "fraction",
        "backlog_release_years": "years",
        "base_term_years": "years",
        "base_year": "year",
        "coverage_rate": "fraction",
        "coverage_fraction": "fraction",
        "cannibalization_fraction": "fraction",
        "clinically_eligible_fraction": "fraction",
        "current_stage": None,
        "decision_grade": None,
        "diagnosed_fraction": "fraction",
        "discount_rate": "fraction",
        "dose_intensity": "fraction",
        "effective_exclusivity_end_year": "year",
        "eligible_patients": "patients",
        "expansion_approval_probability": "fraction",
        "expansion_price_spillover_rate": "fraction",
        "expansion_stage_durations_years": "years",
        "expansion_stage_success_probabilities": "fraction",
        "expected_treatment_years": "years",
        "extension_years": "years",
        "filing_year": "year",
        "forecast_end_year": "year",
        "gross_to_net_rate": "fraction",
        "gross_to_net_shift": "fraction",
        "incidence_growth_rate": "fraction",
        "incremental_qalys": "QALYs",
        "initial_approval_probability": "fraction",
        "initiation_rate": "fraction",
        "initiation_fraction": "fraction",
        "launch_delay_years": "years",
        "launch_year": "year",
        "loe_price_retention": "fraction",
        "loe_retention_multiplier": "multiplier",
        "loe_volume_retention": "fraction",
        "note": None,
        "maximum_oop_share": "fraction",
        "overlap_with_initial_fraction": "fraction",
        "paid_units_per_patient": None,
        "patent_expiry_year": "year",
        "patient_affordability_rate": "fraction",
        "patient_cost_share_fraction": "fraction",
        "patient_multiplier": "multiplier",
        "peak_adoption_rate": "fraction",
        "peak_year": "year",
        "persistence_multiplier": "multiplier",
        "prevalent_backlog_patients": "patients",
        "price_multiplier": "multiplier",
        "prior_authorization_pass_fraction": "fraction",
        "program_probability_of_approval": "fraction",
        "provider_capacity_rate": "fraction",
        "provider_capacity_fraction": "fraction",
        "regulatory_exclusivity_end_year": "year",
        "required_gross_margin_fraction": "fraction",
        "population_share": "fraction",
        "shared_commercial_cost_savings_rate": "fraction",
        "synthetic": None,
        "system_access_fraction": "fraction",
        "stage_durations_years": "years",
        "stage_order": None,
        "stage_success_probabilities": "fraction",
        "tax_rate": "fraction",
        "total_access_fraction": "fraction",
        "valuation_year": "year",
    }
    if normalized in fixed_units:
        return fixed_units[normalized]
    if not currency:
        return None
    financial_units = {
        "annual_comparator_drug_cost": f"{currency}/patient-year",
        "annual_gross_price": f"{currency}/patient-year",
        "annual_manufacturer_cost": f"{currency}/patient-year",
        "annual_net_price": f"{currency}/patient-year",
        "annual_non_drug_cost_offsets": f"{currency}/patient-year",
        "annual_patient_oop": f"{currency}/patient-year",
        "annual_payer_budget_limit": f"{currency}/year",
        "candidate_list_price": f"{currency}/patient-year",
        "cogs_per_full_dose_patient": f"{currency}/patient-year",
        "comparator_total_cost": f"{currency}/patient",
        "development_cost_multiplier": "multiplier",
        "expansion_stage_costs": currency,
        "fixed_commercial_cost_per_year": f"{currency}/year",
        "gross_revenue": currency,
        "gross_to_net_deductions": currency,
        "annual_income": f"{currency}/year",
        "new_non_drug_total_cost": f"{currency}/patient",
        "stage_costs": currency,
        "variable_commercial_cost_per_patient": f"{currency}/patient-year",
        "willingness_to_pay_per_qaly": f"{currency}/QALY",
        "wtp_per_qaly": f"{currency}/QALY",
    }
    if normalized in financial_units:
        return financial_units[normalized]
    return None


def _assumption_evidence_scope(path: str) -> str | None:
    normalized_path = path.removeprefix("input.")
    if ".assumptions." in normalized_path:
        scope = normalized_path.split(".assumptions.", 1)[0]
    else:
        scope = normalized_path.rsplit(".", 1)[0]
    scope = re.sub(r"\[(?:indication_id|comparable_id)=([^\]]+)\]", r"[id=\1]", scope)
    return scope or None


def _matching_evidence_ids(
    keys: Iterable[str],
    path_to_id: dict[str, str],
    *,
    scope_path: str | None = None,
    include_descendants: bool = True,
) -> tuple[str, ...]:
    matches: set[str] = set()
    for key in keys:
        normalized_key = str(key).casefold()
        candidates = [
            (path, evidence_id)
            for path, evidence_id in path_to_id.items()
            if path.casefold().endswith(f".{normalized_key}")
            or f".evidence.{normalized_key}" in path.casefold()
        ]
        if scope_path is not None:
            if include_descendants:
                candidates = [item for item in candidates if item[0].startswith(scope_path)]
            else:
                evidence_prefix = f"{scope_path}.evidence"
                candidates = [
                    item
                    for item in candidates
                    if item[0] == evidence_prefix or item[0].startswith(f"{evidence_prefix}.")
                ]
        for _, evidence_id in candidates:
            matches.add(evidence_id)
    return tuple(sorted(matches))


def _assumption_synthetic_status(evidence_ids: Iterable[str], evidence_by_id: dict[str, Any]):
    records = [evidence_by_id[item] for item in evidence_ids if item in evidence_by_id]
    if not records:
        return None
    return any(bool(item.synthetic) for item in records)


def _build_assumptions(
    result: AnalysisResult,
    evidence: tuple[InterpretabilityEvidence, ...],
    path_to_evidence_id: dict[str, str],
) -> tuple[
    tuple[InterpretabilityAssumption, ...],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    list[InterpretabilityLimitation],
]:
    snapshot = _data(result.input_snapshot)
    program = snapshot.get("program", {}) if isinstance(snapshot, dict) else {}
    currency = program.get("currency") if isinstance(program, dict) else None
    evidence_by_id = {item.id: item for item in evidence}
    indication_scope_paths: dict[str, str] = {}
    if isinstance(program, dict):
        initial = program.get("initial_indication")
        if isinstance(initial, dict) and initial.get("indication_id"):
            indication_scope_paths[str(initial["indication_id"])] = "program.initial_indication"
        expansions = program.get("expansion_indications", [])
        if isinstance(expansions, list):
            for item in expansions:
                if isinstance(item, dict) and item.get("indication_id"):
                    indication_id = str(item["indication_id"])
                    indication_scope_paths[indication_id] = (
                        f"program.expansion_indications[id={indication_id}]"
                    )
    assumptions: list[InterpretabilityAssumption] = []
    limitations: list[InterpretabilityLimitation] = []
    paths_seen: set[str] = set()
    assumption_ids_by_key: dict[str, list[str]] = defaultdict(list)

    def add_assumption(
        *,
        path: str,
        value: Any,
        unit: str | None,
        basis: str,
        evidence_ids: tuple[str, ...],
        synthetic_override: bool | None = None,
    ) -> str:
        assumption_id = _stable_id("assumption", path, path.rsplit(".", 1)[-1])
        if path in paths_seen:
            return assumption_id
        paths_seen.add(path)
        linked_synthetic = _assumption_synthetic_status(evidence_ids, evidence_by_id)
        synthetic_conflict = synthetic_override is False and linked_synthetic is True
        if synthetic_override is True or linked_synthetic is True:
            synthetic = True
        elif synthetic_override is False or linked_synthetic is False:
            synthetic = False
        else:
            synthetic = None
        assumptions.append(
            InterpretabilityAssumption(
                id=assumption_id,
                path=path,
                value=_data(value),
                unit=unit,
                basis=basis,
                synthetic=synthetic,
            )
        )
        leaf = path.rsplit(".", 1)[-1]
        assumption_ids_by_key[leaf].append(assumption_id)
        if value is None:
            limitations.append(
                InterpretabilityLimitation(
                    code="UNKNOWN_ASSUMPTION_VALUE",
                    severity="WARNING",
                    message=f"The authoritative value at {path} is unknown and remains null.",
                    field_path=path,
                )
            )
        elif _contains_unknown(_data(value)):
            limitations.append(
                InterpretabilityLimitation(
                    code="UNKNOWN_ASSUMPTION_VALUE",
                    severity="WARNING",
                    message=(
                        f"The structured assumption at {path} contains one or more unknown null "
                        "values; they remain null."
                    ),
                    field_path=path,
                )
            )
        if unit is None:
            limitations.append(
                InterpretabilityLimitation(
                    code="ASSUMPTION_UNIT_UNSPECIFIED",
                    severity="INFO",
                    message=f"The native contract does not specify a unit for {path}.",
                    field_path=f"interpretability.assumptions.{assumption_id}.unit",
                )
            )
        if synthetic is None:
            limitations.append(
                InterpretabilityLimitation(
                    code="ASSUMPTION_SYNTHETIC_STATUS_UNKNOWN",
                    severity="INFO",
                    message=(
                        f"No evidence link establishes whether the assumption at {path} is "
                        "synthetic; synthetic remains null."
                    ),
                    field_path=f"interpretability.assumptions.{assumption_id}.synthetic",
                )
            )
        if synthetic_conflict:
            limitations.append(
                InterpretabilityLimitation(
                    code="SYNTHETIC_STATUS_CONFLICT",
                    severity="ERROR",
                    message=(
                        f"The explicit synthetic=false marker at {path} conflicts with linked "
                        "synthetic evidence; the shared assumption remains synthetic=true."
                    ),
                    field_path=f"interpretability.assumptions.{assumption_id}.synthetic",
                )
            )
        return assumption_id

    for path, value, explicit_synthetic in _iter_explicit_assumptions(snapshot):
        key = path.rsplit(".", 1)[-1]
        evidence_ids = _matching_evidence_ids(
            (key,),
            path_to_evidence_id,
            scope_path=_assumption_evidence_scope(path),
            include_descendants=False,
        )
        if not evidence_ids and path.startswith("input.program.development.assumptions."):
            evidence_ids = _matching_evidence_ids(
                ("development_inputs",),
                path_to_evidence_id,
                scope_path="program.development",
                include_descendants=False,
            )
        add_assumption(
            path=path,
            value=value,
            unit=_unit_for_key(key, currency),
            basis=(
                "Supplied in the module request; the native contract does not provide a per-field "
                "selection rationale."
            ),
            evidence_ids=evidence_ids,
            synthetic_override=explicit_synthetic,
        )

    for path, value in _iter_material_input_assumptions(snapshot):
        key = path.rsplit(".", 1)[-1]
        scope_path = _assumption_evidence_scope(path)
        evidence_ids = _matching_evidence_ids(
            (key,),
            path_to_evidence_id,
            scope_path=scope_path,
            include_descendants=False,
        )
        if not evidence_ids and key == "adoption_by_year":
            evidence_ids = _matching_evidence_ids(
                ("access_curve",),
                path_to_evidence_id,
                scope_path=scope_path,
                include_descendants=False,
            )
        if not evidence_ids and path.startswith("input.program.development."):
            evidence_ids = _matching_evidence_ids(
                ("development_inputs",),
                path_to_evidence_id,
                scope_path="program.development",
                include_descendants=False,
            )
        if not evidence_ids and ".income_bands[" in path:
            evidence_ids = _matching_evidence_ids(
                ("evidence",),
                path_to_evidence_id,
                scope_path=scope_path,
                include_descendants=False,
            )
        add_assumption(
            path=path,
            value=value,
            unit=_unit_for_key(key, currency),
            basis=(
                "Typed economic input supplied in the module request; the native contract does "
                "not provide a per-field selection rationale."
            ),
            evidence_ids=evidence_ids,
        )

    step_assumption_ids: dict[str, tuple[str, ...]] = {}
    step_evidence_ids: dict[str, tuple[str, ...]] = {}
    for step in result.calculation_steps:
        indication_id = next(
            (
                item
                for item in sorted(indication_scope_paths, key=len, reverse=True)
                if step.step_id.startswith(f"pricing:{item}:")
            ),
            None,
        )
        evidence_ids = _matching_evidence_ids(
            step.evidence_keys,
            path_to_evidence_id,
            scope_path=indication_scope_paths.get(indication_id) if indication_id else None,
        )
        ids: list[str] = []
        for key, value in sorted(step.inputs.items()):
            path = f"payload.calculation_steps.{step.step_id}.inputs.{key}"
            input_evidence_ids = _matching_evidence_ids(
                (key,),
                path_to_evidence_id,
                scope_path=indication_scope_paths.get(indication_id) if indication_id else None,
            )
            assumption_id = add_assumption(
                path=path,
                value=value,
                unit=_unit_for_key(key, currency),
                basis=(
                    "Resolved input exposed by the authoritative native calculation step; the "
                    "interpretability adapter did not recalculate it, and the native contract does "
                    "not provide a per-field selection rationale."
                ),
                evidence_ids=input_evidence_ids,
            )
            ids.append(assumption_id)
        step_assumption_ids[step.step_id] = tuple(ids)
        step_evidence_ids[step.step_id] = evidence_ids

    simulation_assumption_ids: list[str] = []
    for key, value in sorted(_data(result.simulation_assumptions).items()):
        assumption_id = add_assumption(
            path=f"input.execution.simulation_assumptions.{key}",
            value=value,
            unit=_unit_for_key(key, currency),
            basis=(
                "Explicit seeded Monte Carlo scenario range supplied in execution settings; the "
                "native contract does not provide a per-field selection rationale."
            ),
            evidence_ids=(),
        )
        simulation_assumption_ids.append(assumption_id)
    step_assumption_ids["__simulation__"] = tuple(simulation_assumption_ids)
    step_evidence_ids["__simulation__"] = ()
    return tuple(assumptions), step_assumption_ids, step_evidence_ids, limitations


def _display(value: JsonValue, unit: str) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        if unit == "fraction":
            return f"{float(value) * 100:.3g}%"
        if unit == "year":
            return f"{int(value)}"
        magnitude = abs(float(value))
        if magnitude >= 1_000_000_000:
            return f"{float(value) / 1_000_000_000:.3g} billion {unit}"
        if magnitude >= 1_000_000:
            return f"{float(value) / 1_000_000:.3g} million {unit}"
        return f"{float(value):,.6g} {unit}"
    return f"{value} {unit}"


def _build_metrics(
    result: AnalysisResult,
    step_assumption_ids: dict[str, tuple[str, ...]],
    step_evidence_ids: dict[str, tuple[str, ...]],
    path_to_evidence_id: dict[str, str],
) -> tuple[
    tuple[InterpretabilityMetric, ...],
    dict[str, dict[str, JsonValue]],
    dict[str, dict[str, str]],
    list[InterpretabilityLimitation],
]:
    snapshot = _data(result.input_snapshot)
    program = snapshot.get("program", {}) if isinstance(snapshot, dict) else {}
    currency = (
        str(program.get("currency") or "currency") if isinstance(program, dict) else "currency"
    )
    valuation_year = program.get("valuation_year") if isinstance(program, dict) else None
    price_bases = sorted(
        {
            item.annual_net_price_corridor.basis.value
            for item in result.pricing
            if item.annual_net_price_corridor is not None
        }
    )
    price_basis: JsonValue = (
        price_bases[0] if len(price_bases) == 1 else "MIXED" if price_bases else None
    )
    metrics: list[InterpretabilityMetric] = []
    limitations: list[InterpretabilityLimitation] = []
    financial_context: dict[str, dict[str, JsonValue]] = {}
    indication_metric_ids: dict[str, dict[str, str]] = {}
    indication_scope_paths: dict[str, str] = {}
    if isinstance(program, dict):
        initial = program.get("initial_indication")
        if isinstance(initial, dict) and initial.get("indication_id"):
            indication_scope_paths[str(initial["indication_id"])] = "program.initial_indication"
        expansions = program.get("expansion_indications", [])
        if isinstance(expansions, list):
            for item in expansions:
                if isinstance(item, dict) and item.get("indication_id"):
                    indication_id = str(item["indication_id"])
                    indication_scope_paths[indication_id] = (
                        f"program.expansion_indications[id={indication_id}]"
                    )

    def links(
        step_ids: tuple[str, ...],
        *,
        simulation: bool = False,
        extra_evidence_ids: Iterable[str] = (),
    ):
        assumptions = {
            item for step_id in step_ids for item in step_assumption_ids.get(step_id, ())
        }
        evidence = {item for step_id in step_ids for item in step_evidence_ids.get(step_id, ())}
        evidence.update(extra_evidence_ids)
        if simulation:
            assumptions.update(step_assumption_ids.get("__simulation__", ()))
        return tuple(sorted(evidence)), tuple(sorted(assumptions))

    def add(
        metric_id: str,
        label: str,
        value: JsonValue,
        unit: str,
        meaning: str,
        direction: MetricDirection,
        *,
        step_ids: tuple[str, ...] = (),
        simulation: bool = False,
        metric_price_basis: JsonValue = price_basis,
        extra_evidence_ids: Iterable[str] = (),
    ) -> None:
        evidence_ids, assumption_ids = links(
            step_ids,
            simulation=simulation,
            extra_evidence_ids=extra_evidence_ids,
        )
        metrics.append(
            InterpretabilityMetric(
                id=metric_id,
                label=label,
                value=value,
                unit=unit,
                display=_display(value, unit),
                meaning=meaning,
                direction=direction,
                evidence_ids=evidence_ids,
                assumption_ids=assumption_ids,
            )
        )
        if value is None:
            limitations.append(
                InterpretabilityLimitation(
                    code="UNKNOWN_METRIC_VALUE",
                    severity="WARNING",
                    message=f"Metric {metric_id} is unknown and remains null.",
                    field_path=f"interpretability.metrics.{metric_id}.value",
                )
            )
        if not evidence_ids and not assumption_ids:
            limitations.append(
                InterpretabilityLimitation(
                    code="UNTAGGED_VALUE",
                    severity="WARNING",
                    message=f"Metric {metric_id} has no native evidence or assumption link.",
                    field_path=f"interpretability.metrics.{metric_id}",
                )
            )
        if unit.startswith(currency):
            financial_context[metric_id] = {
                "currency": None if currency == "currency" else currency,
                "valuation_year": valuation_year,
                "price_basis": metric_price_basis,
                "price_bases": price_bases,
            }
            if currency == "currency" or valuation_year is None or metric_price_basis is None:
                limitations.append(
                    InterpretabilityLimitation(
                        code="FINANCIAL_CONTEXT_PARTIAL",
                        severity="WARNING",
                        message=(
                            f"Currency, valuation year, or price basis is unknown for {metric_id}; "
                            "the missing field remains null."
                        ),
                        field_path=f"interpretability.extensions.financial_metric_context.{metric_id}",
                    )
                )

    summary = result.summary
    cashflow_steps = {
        "rnpv": ("risk_adjusted_npv",),
        "peak": ("peak_annual_net_revenue",),
        "patent": ("patent_clock",),
        "delay": ("launch_delay_cost",),
    }
    add(
        "metric.rnpv.deterministic",
        "Deterministic risk-adjusted NPV",
        summary.deterministic_rnpv,
        currency,
        "Risk-adjusted manufacturer value from the deterministic annual cash-flow ledger.",
        "positive",
        step_ids=cashflow_steps["rnpv"],
    )
    add(
        "metric.rnpv.mean",
        "Simulated mean rNPV",
        summary.simulated_mean_rnpv,
        currency,
        "Mean rNPV across the declared seeded scenario draws.",
        "positive",
        step_ids=cashflow_steps["rnpv"],
        simulation=True,
    )
    for percentile, value in (
        ("p10", summary.p10_rnpv),
        ("p50", summary.p50_rnpv),
        ("p90", summary.p90_rnpv),
    ):
        add(
            f"metric.rnpv.{percentile}",
            f"rNPV {percentile.upper()}",
            value,
            currency,
            f"{percentile.upper()} percentile of the declared Monte Carlo scenario distribution.",
            "positive",
            step_ids=cashflow_steps["rnpv"],
            simulation=True,
        )
    add(
        "metric.rnpv.probability_positive",
        "Share of simulated rNPV draws above zero",
        summary.probability_positive_rnpv,
        "fraction",
        "Scenario-draw share above zero; it is not a probability of approval or forecast accuracy.",
        "positive",
        step_ids=cashflow_steps["rnpv"],
        simulation=True,
    )
    add(
        "metric.revenue.peak_annual",
        "Peak annual manufacturer net revenue",
        summary.peak_annual_net_revenue,
        f"{currency}/year",
        "Largest annual net-revenue value in the deterministic ledger.",
        "positive",
        step_ids=cashflow_steps["peak"],
    )
    add(
        "metric.revenue.peak_annual_p50",
        "Peak annual manufacturer net revenue P50",
        summary.peak_annual_net_revenue_p50,
        f"{currency}/year",
        "Median peak annual net revenue across declared scenario draws.",
        "positive",
        step_ids=cashflow_steps["peak"],
        simulation=True,
    )
    add(
        "metric.revenue.peak_year",
        "Peak annual net revenue year",
        summary.peak_annual_net_revenue_year,
        "year",
        (
            "Calendar year of deterministic peak annual net revenue, or unknown if no positive "
            "peak exists."
        ),
        "neutral",
        step_ids=cashflow_steps["peak"],
    )
    add(
        "metric.cash_at_risk.p50",
        "Peak cash at risk P50",
        summary.peak_cash_at_risk_p50,
        currency,
        "Median maximum cumulative cash deficit across declared scenario draws.",
        "negative",
        step_ids=cashflow_steps["rnpv"],
        simulation=True,
    )
    add(
        "metric.protected_years.effective",
        "Effective protected years",
        summary.effective_protected_years,
        "years",
        "Modeled commercial protection after initial launch under the shared asset clock.",
        "positive",
        step_ids=cashflow_steps["patent"],
    )
    add(
        "metric.launch_delay.value_lost",
        "Value lost per one-year launch delay",
        summary.value_lost_per_launch_delay_year,
        currency,
        "Difference between base NPV and the native one-year delayed-launch counterfactual.",
        "negative",
        step_ids=cashflow_steps["delay"],
    )

    for pricing in result.pricing:
        corridor = pricing.annual_net_price_corridor
        selected_price = corridor.selected_annual_net_price if corridor is not None else None
        selected_basis: JsonValue = corridor.basis.value if corridor is not None else None
        pricing_step_ids = tuple(
            step_id
            for step_id in step_assumption_ids
            if next(
                (
                    indication_id
                    for indication_id in sorted(indication_scope_paths, key=len, reverse=True)
                    if step_id.startswith(f"pricing:{indication_id}:")
                ),
                None,
            )
            == pricing.indication_id
        )
        comparable_gate_prefix = f"pricing.{pricing.indication_id}.comparable:"
        comparator_ids = {
            key.removeprefix(comparable_gate_prefix).removesuffix(":price")
            for key in result.critical_evidence_status
            if key.startswith(comparable_gate_prefix) and key.endswith(":price")
        }
        comparator_evidence_ids = {
            evidence_id
            for path, evidence_id in path_to_evidence_id.items()
            if any(
                path.startswith(f"comparables.comparables[id={comparable_id}].price.evidence")
                for comparable_id in comparator_ids
            )
        }
        price_access_evidence_ids: set[str] = set()
        scoped_access_evidence_ids: set[str] = set()
        indication_scope = indication_scope_paths.get(pricing.indication_id)
        if indication_scope:
            price_access_markers = (
                ".population.evidence.",
                ".access.evidence.",
            )
            price_access_evidence_ids = {
                evidence_id
                for path, evidence_id in path_to_evidence_id.items()
                if path.startswith(indication_scope)
                and any(marker in path for marker in price_access_markers)
            }
            accessible_patient_markers = (
                *price_access_markers,
                ".income_bands[",
            )
            scoped_access_evidence_ids = {
                evidence_id
                for path, evidence_id in path_to_evidence_id.items()
                if path.startswith(indication_scope)
                and any(marker in path for marker in accessible_patient_markers)
            }
        selected_price_metric_id = _indication_metric_id(
            "metric.price.selected", pricing.indication_id
        )
        accessible_patients_metric_id = _indication_metric_id(
            "metric.accessible_patients", pricing.indication_id
        )
        indication_metric_ids[pricing.indication_id] = {
            "selected_price": selected_price_metric_id,
            "accessible_patients": accessible_patients_metric_id,
        }
        add(
            selected_price_metric_id,
            f"Selected annual net-price scenario: {pricing.indication_id}",
            selected_price,
            f"{currency}/patient-year",
            (
                "Authoritative selected price from the native price corridor; not a confidential "
                "observed net price."
            ),
            "mixed",
            step_ids=pricing_step_ids,
            metric_price_basis=selected_basis,
            extra_evidence_ids=(*comparator_evidence_ids, *price_access_evidence_ids),
        )
        selected_access = None
        if selected_price is not None:
            selected_access = next(
                (
                    item
                    for item in pricing.access_estimates
                    if item.annual_net_price == selected_price
                ),
                None,
            )
        accessible_patients = selected_access.accessible_patients if selected_access else None
        add(
            accessible_patients_metric_id,
            f"Accessible patients at selected price: {pricing.indication_id}",
            accessible_patients,
            "patients",
            (
                "Modeled accessible population at the native selected price after access and "
                "affordability gates."
            ),
            "positive",
            step_ids=pricing_step_ids,
            extra_evidence_ids=(*comparator_evidence_ids, *scoped_access_evidence_ids),
        )
    return tuple(metrics), financial_context, indication_metric_ids, limitations


def _build_steps(
    result: AnalysisResult,
    assumptions: tuple[InterpretabilityAssumption, ...],
    step_assumption_ids: dict[str, tuple[str, ...]],
    step_evidence_ids: dict[str, tuple[str, ...]],
) -> tuple[tuple[InterpretabilityStep, ...], list[InterpretabilityLimitation]]:
    assumptions_by_id = {item.id: item for item in assumptions}
    steps: list[InterpretabilityStep] = []
    limitations: list[InterpretabilityLimitation] = []
    for native in result.calculation_steps:
        common_step_id = _common_step_id(native.step_id)
        assumption_ids = step_assumption_ids[native.step_id]
        inputs = tuple(
            InterpretabilityStepInput(
                path=assumptions_by_id[assumption_id].path,
                value=assumptions_by_id[assumption_id].value,
                unit=assumptions_by_id[assumption_id].unit,
            )
            for assumption_id in assumption_ids
        )
        steps.append(
            InterpretabilityStep(
                id=common_step_id,
                label=native.label,
                method=native.step_id,
                formula=native.formula,
                inputs=inputs,
                result=InterpretabilityStepResult(value=native.result, unit=native.unit),
                evidence_ids=step_evidence_ids[native.step_id],
                assumption_ids=assumption_ids,
            )
        )
        if native.result is None:
            limitations.append(
                InterpretabilityLimitation(
                    code="UNKNOWN_STEP_RESULT",
                    severity="WARNING",
                    message=f"Calculation step {native.step_id} returned an unknown result.",
                    field_path=f"interpretability.steps.{common_step_id}.result.value",
                )
            )
    return tuple(steps), limitations


def _build_uncertainty(result: AnalysisResult) -> InterpretabilityUncertainty:
    uncertainty = result.uncertainty
    snapshot = _data(result.input_snapshot)
    program = snapshot.get("program", {}) if isinstance(snapshot, dict) else {}
    currency = (
        str(program.get("currency") or "currency") if isinstance(program, dict) else "currency"
    )
    intervals = (
        InterpretabilityInterval(
            metric_id="metric.rnpv.p50",
            low=uncertainty.rnpv.p10,
            central=uncertainty.rnpv.p50,
            high=uncertainty.rnpv.p90,
            unit=currency,
            confidence_level=None,
        ),
        InterpretabilityInterval(
            metric_id="metric.revenue.peak_annual_p50",
            low=uncertainty.peak_annual_net_revenue.p10,
            central=uncertainty.peak_annual_net_revenue.p50,
            high=uncertainty.peak_annual_net_revenue.p90,
            unit=f"{currency}/year",
            confidence_level=None,
        ),
        InterpretabilityInterval(
            metric_id="metric.cash_at_risk.p50",
            low=uncertainty.peak_cash_at_risk.p10,
            central=uncertainty.peak_cash_at_risk.p50,
            high=uncertainty.peak_cash_at_risk.p90,
            unit=currency,
            confidence_level=None,
        ),
        InterpretabilityInterval(
            metric_id="metric.protected_years.effective",
            low=uncertainty.effective_protected_years.p10,
            central=uncertainty.effective_protected_years.p50,
            high=uncertainty.effective_protected_years.p90,
            unit="years",
            confidence_level=None,
        ),
    )
    return InterpretabilityUncertainty(
        method=(
            "Seeded Monte Carlo scenario percentiles (P10/P50/P90); these are not confidence "
            "intervals or observed frequencies."
        ),
        intervals=intervals,
        seed=result.seed,
        draws=result.simulations,
        limitations=(
            (
                "Low, central, and high are scenario P10, P50, and P90 percentiles, not confidence "
                "intervals."
            ),
            (
                "The declared ranges and model structure are assumptions; the distribution is not "
                "externally calibrated."
            ),
        ),
    )


def _build_lineage(result: AnalysisResult) -> tuple[InterpretabilityLineage, ...]:
    snapshot = _data(result.input_snapshot)
    program = snapshot.get("program", {}) if isinstance(snapshot, dict) else {}
    indication_paths: list[str] = []
    launch_paths: list[str] = []
    if isinstance(program, dict):
        initial = program.get("initial_indication")
        if isinstance(initial, dict):
            indication_paths.append("input.program.initial_indication")
            launch_paths.append("input.program.initial_indication.launch_year")
        expansions = program.get("expansion_indications", [])
        if isinstance(expansions, list):
            for item in expansions:
                if isinstance(item, dict) and item.get("indication_id"):
                    base = (
                        "input.program.expansion_indications"
                        f"[indication_id={item['indication_id']}]"
                    )
                    indication_paths.append(base)
                    launch_paths.append(f"{base}.launch_year")
    if not indication_paths:
        indication_paths.append("input.cashflow_inputs.initial_indication")
        launch_paths.append("input.cashflow_inputs.initial_indication.launch_year")

    price_inputs = tuple(
        [
            path
            for indication_path in indication_paths
            for path in (
                f"{indication_path}.assumptions",
                f"{indication_path}.comparator_ids",
                f"{indication_path}.population",
                f"{indication_path}.access",
            )
        ]
        + ["input.program.assumptions", "input.comparables"]
    )
    access_inputs = (
        *(
            path
            for indication_path in indication_paths
            for path in (
                f"{indication_path}.population",
                f"{indication_path}.access",
                f"{indication_path}.income_bands",
                f"{indication_path}.assumptions",
                f"{indication_path}.comparator_ids",
            )
        ),
        "input.comparables",
        "payload.pricing[].annual_net_price_corridor.selected_annual_net_price",
    )
    persistence_inputs = (
        *(
            path
            for indication_path in indication_paths
            for path in (
                f"{indication_path}.assumptions.annual_persistence_rate",
                f"{indication_path}.assumptions.dose_intensity",
            )
        ),
        "input.program.assumptions",
    )
    development = program.get("development", {}) if isinstance(program, dict) else {}
    uses_stage_path = isinstance(development, dict) and bool(
        development.get("stage_success_probabilities")
    )
    explicit_stage_order = development.get("stage_order") if isinstance(development, dict) else None
    if uses_stage_path:
        if explicit_stage_order:
            development_probability_inputs = (
                "input.program.development.stage_success_probabilities",
                "input.program.development.stage_order",
            )
            ordering = "the explicitly declared stage order"
        else:
            development_probability_inputs = (
                "input.program.development.stage_success_probabilities",
            )
            ordering = "the deterministic lifecycle ordering of the supplied stage names"
        development_probability_transformation = (
            "The native engine multiplies sequential stage success probabilities using "
            f"{ordering}; it does not apply a hidden modality prior."
        )
        development_cost_inputs = (
            "input.program.development.stage_costs",
            "input.program.development.stage_durations_years",
            "input.program.development.stage_success_probabilities",
            *(("input.program.development.stage_order",) if explicit_stage_order else ()),
        )
        development_cost_transformation = (
            "The native ledger schedules stage costs and weights expected spending by the "
            f"probability of reaching each stage using {ordering}."
        )
    else:
        development_probability_inputs = (
            "input.program.development.program_probability_of_approval",
        )
        development_probability_transformation = (
            "With no stage path supplied, the native engine uses the explicit aggregate program "
            "probability of approval; it does not apply a hidden modality prior."
        )
        development_cost_inputs = ("input.program.development.stage_costs",)
        development_cost_transformation = (
            "No remaining stage-cost path was supplied, so the native ledger represents remaining "
            "development cost as zero; aggregate approval probability does not create a cost path."
        )
    development_assumptions = (
        development.get("assumptions", {}) if isinstance(development, dict) else {}
    )
    has_expansion_development_path = isinstance(development_assumptions, dict) and bool(
        development_assumptions.get("expansion_stage_costs")
    )
    if has_expansion_development_path:
        explicit_expansion_order = development_assumptions.get("expansion_stage_order")
        development_cost_inputs = (
            *development_cost_inputs,
            "input.program.development.assumptions.expansion_stage_costs",
            "input.program.development.assumptions.expansion_stage_durations_years",
            "input.program.development.assumptions.expansion_stage_success_probabilities",
            *(
                ("input.program.development.assumptions.expansion_stage_order",)
                if explicit_expansion_order
                else ()
            ),
        )
        development_cost_transformation += (
            " Total development costs also include the expansion path, weighted by its "
            "conditional reach probabilities."
        )

    lineage = (
        InterpretabilityLineage(
            output_path="payload.pricing[].annual_net_price_corridor",
            input_paths=price_inputs,
            transformation=(
                "The native pricing engine combines value, payer-affordability, commercial-floor, "
                "and allowlisted comparable anchors, then clips and selects the corridor value."
            ),
        ),
        InterpretabilityLineage(
            output_path="payload.pricing[].access_estimates[].accessible_patients",
            input_paths=access_inputs,
            transformation=(
                "The native access engine applies system-access and income-affordability gates to "
                "eligible patients at each modeled price; unknown affordability stays null."
            ),
        ),
        InterpretabilityLineage(
            output_path="payload.access[].annual_persistence_rate",
            input_paths=persistence_inputs,
            transformation=(
                "The authoritative cohort ledger carries each patient vintage by the supplied "
                "annual persistence and dose-intensity assumptions."
            ),
        ),
        InterpretabilityLineage(
            output_path="payload.cash_flow.initial_approval_probability",
            input_paths=development_probability_inputs,
            transformation=development_probability_transformation,
        ),
        InterpretabilityLineage(
            output_path="payload.value_decomposition.development_costs",
            input_paths=development_cost_inputs,
            transformation=development_cost_transformation,
        ),
        InterpretabilityLineage(
            output_path="payload.cash_flow.effective_exclusivity_end_year",
            input_paths=(
                "input.program.patent.filing_year",
                "input.program.patent.base_term_years",
                "input.program.patent.extension_years",
                "input.program.patent.regulatory_exclusivity_end_year",
            ),
            transformation=(
                "The native shared asset clock takes the later modeled patent or regulatory "
                "exclusivity end; label expansion does not restart the patent term."
            ),
        ),
        InterpretabilityLineage(
            output_path="payload.cash_flow.effective_protected_years",
            input_paths=(
                "input.program.patent.filing_year",
                "input.program.patent.base_term_years",
                "input.program.patent.extension_years",
                "input.program.patent.regulatory_exclusivity_end_year",
                "input.program.initial_indication.launch_year",
            ),
            transformation=(
                "The native engine measures the initial indication's commercial protection from "
                "launch through the shared effective exclusivity end year."
            ),
        ),
        InterpretabilityLineage(
            output_path="payload.summary.value_lost_per_launch_delay_year",
            input_paths=(*launch_paths, "input.program.patent"),
            transformation=(
                "The native counterfactual subtracts NPV after shifting all launches by one year "
                "while leaving the shared patent clock unchanged."
            ),
        ),
    )
    expansion_paths = tuple(path for path in indication_paths if "expansion_indications" in path)
    if not expansion_paths:
        return lineage
    if has_expansion_development_path:
        explicit_expansion_order = development_assumptions.get("expansion_stage_order")
        expansion_development_inputs = (
            "input.program.development.assumptions.expansion_stage_costs",
            "input.program.development.assumptions.expansion_stage_durations_years",
            "input.program.development.assumptions.expansion_stage_success_probabilities",
            *(
                ("input.program.development.assumptions.expansion_stage_order",)
                if explicit_expansion_order
                else ()
            ),
        )
        expansion_ordering = (
            "explicit stage order"
            if explicit_expansion_order
            else "deterministic lifecycle ordering of the supplied stage names"
        )
        expansion_transformation = (
            "The native ledger models the first label expansion with its supplied development "
            f"path, {expansion_ordering}, overlap, cannibalization, price "
            "spillover, shared costs, and the original asset protection clock."
        )
    else:
        expansion_development_inputs = ()
        expansion_transformation = (
            "No expansion development path was supplied. The native degraded result makes the "
            "expansion commercial probability conditional only on initial-program success and "
            "emits MISSING_EXPANSION_DEVELOPMENT_PATH."
        )
    return (
        *lineage,
        InterpretabilityLineage(
            output_path="payload.value_decomposition.expansion_increment_discounted_fcf",
            input_paths=(
                "input.program.initial_indication",
                "input.program.assumptions",
                *expansion_paths,
                *expansion_development_inputs,
                "input.program.patent",
            ),
            transformation=expansion_transformation,
        ),
    )


def build_interpretability(result: AnalysisResult) -> Interpretability:
    """Map one completed native analysis into the shared contract deterministically."""

    result = AnalysisResult.model_validate(result.model_dump(mode="python"))
    evidence, path_to_evidence_id, native_evidence, evidence_limitations = _build_evidence(result)
    (
        assumptions,
        step_assumption_ids,
        step_evidence_ids,
        assumption_limitations,
    ) = _build_assumptions(result, evidence, path_to_evidence_id)
    (
        metrics,
        financial_context,
        indication_metric_ids,
        metric_limitations,
    ) = _build_metrics(
        result,
        step_assumption_ids,
        step_evidence_ids,
        path_to_evidence_id,
    )
    steps, step_limitations = _build_steps(
        result, assumptions, step_assumption_ids, step_evidence_ids
    )

    limitations = [
        InterpretabilityLimitation(
            code=warning.code,
            severity=warning.severity.value,
            message=warning.message,
            field_path="payload.warnings",
        )
        for warning in result.warnings
    ]
    limitations.extend(
        InterpretabilityLimitation(
            code="CRITICAL_EVIDENCE_GATE_FAILED",
            severity="ERROR",
            message=f"Critical evidence gate {key!r} did not pass.",
            field_path=f"payload.critical_evidence_status.{key}",
        )
        for key, passed in sorted(result.critical_evidence_status.items())
        if not passed
    )
    if result.decision_grade.value == "NOT_DECISION_GRADE":
        limitations.append(
            InterpretabilityLimitation(
                code="NOT_DECISION_GRADE",
                severity="ERROR",
                message=(
                    "The calculation completed, but unsupported, synthetic, missing, or failed "
                    "critical evidence prevents decision-grade use."
                ),
                field_path="payload.decision_grade",
            )
        )
    limitations.extend(evidence_limitations)
    limitations.extend(assumption_limitations)
    limitations.extend(metric_limitations)
    limitations.extend(step_limitations)
    synthetic_assumption_ids = tuple(item.id for item in assumptions if item.synthetic is True)
    if synthetic_assumption_ids:
        limitations.append(
            InterpretabilityLimitation(
                code="SYNTHETIC_ASSUMPTIONS_PRESENT",
                severity=(
                    "ERROR" if result.decision_grade.value != "NOT_DECISION_GRADE" else "WARNING"
                ),
                message=(
                    f"{len(synthetic_assumption_ids)} shared assumptions are explicitly synthetic "
                    "or linked to synthetic evidence; they cannot support an unqualified result."
                ),
                field_path="interpretability.assumptions",
            )
        )
    if not steps:
        limitations.append(
            InterpretabilityLimitation(
                code="NO_CALCULATION_STEPS",
                severity="ERROR",
                message="The native result contains no calculation steps.",
                field_path="payload.calculation_steps",
            )
        )
    limitations.extend(
        (
            InterpretabilityLimitation(
                code="INTERPRETABILITY_PARTIAL",
                severity="WARNING",
                message=(
                    "Native calculation-step inputs do not carry original source paths or typed "
                    "per-input units; unresolved units remain null and resolved values are linked "
                    "to their stable native step paths."
                ),
                field_path="interpretability.steps[].inputs",
            ),
            InterpretabilityLimitation(
                code="INTERPRETABILITY_PARTIAL",
                severity="WARNING",
                message=(
                    "Assumption basis records request or calculation-step origin because the "
                    "native contract does not provide a per-field selection rationale."
                ),
                field_path="interpretability.assumptions[].basis",
            ),
            InterpretabilityLimitation(
                code="COUNTERFACTUAL_COVERAGE_PARTIAL",
                severity="INFO",
                message=(
                    "The native engine calculates only the one-year launch-delay counterfactual; "
                    "other evidence changes are described without invented numeric outcomes."
                ),
                field_path="interpretability.counterfactuals",
            ),
        )
    )

    synthetic = any(item.synthetic for item in evidence) or bool(synthetic_assumption_ids)
    basis: tuple[HeadlineBasis, ...] = ("MODELED", "SYNTHETIC") if synthetic else ("MODELED",)
    decision_grade = result.decision_grade.value
    recommendation = result.recommendation.value
    headline_result = decision_grade if decision_grade == "NOT_DECISION_GRADE" else recommendation
    failed_evidence_gate = any(not passed for passed in result.critical_evidence_status.values())
    if decision_grade == "NOT_DECISION_GRADE" or not evidence or failed_evidence_gate or synthetic:
        headline_status: HeadlineStatus = "INCONCLUSIVE"
    elif result.warnings:
        headline_status = "QUALIFIED"
    else:
        headline_status = "SUPPORTED"
    delay_value = result.summary.value_lost_per_launch_delay_year
    snapshot = _data(result.input_snapshot)
    program = snapshot.get("program", {}) if isinstance(snapshot, dict) else {}
    currency = (
        str(program.get("currency") or "currency") if isinstance(program, dict) else "currency"
    )
    if isinstance(program, dict) and not program.get("expansion_indications"):
        limitations.append(
            InterpretabilityLimitation(
                code="LABEL_EXPANSION_NOT_APPLICABLE",
                severity="INFO",
                message="No label expansion was supplied, so expansion lineage is not applicable.",
                field_path="payload.value_decomposition.expansion_increment_discounted_fcf",
            )
        )

    calculation_step_notes = {
        step.step_id: list(step.notes) for step in result.calculation_steps if step.notes
    }
    extensions: dict[str, JsonValue] = {
        "decision_grade": decision_grade,
        "recommendation": recommendation,
        "financial_metric_context": financial_context,
        "indication_metric_ids": indication_metric_ids,
        "financial_output_context": {
            "currency": None if currency == "currency" else currency,
            "valuation_year": program.get("valuation_year") if isinstance(program, dict) else None,
            "price_bases": sorted(
                {
                    item.annual_net_price_corridor.basis.value
                    for item in result.pricing
                    if item.annual_net_price_corridor is not None
                }
            ),
        },
        "value_decomposition": _data(result.value_decomposition),
        "annual_ledger": _data(result.cash_flow.annual_cash_flows),
        "uncertainty_design": {
            "rng_bit_generator": result.uncertainty.rng_bit_generator,
            "numpy_version": result.uncertainty.numpy_version,
            "draw_order_contract_version": result.uncertainty.draw_order_contract_version,
            "commercial_driver_correlation": result.uncertainty.commercial_driver_correlation,
            "assumptions": _data(result.simulation_assumptions),
        },
        "critical_evidence_status": _data(result.critical_evidence_status),
        "native_evidence": native_evidence,
        "calculation_step_notes": calculation_step_notes,
        "native_step_ids": {
            _common_step_id(step.step_id): step.step_id for step in result.calculation_steps
        },
        "interpretability_partial_fields": [
            "evidence.claim",
            "evidence.locator",
            "evidence.quote",
            "steps.inputs[].path",
            "steps.inputs[].unit",
            "assumptions[].basis",
            "assumptions[].synthetic",
        ],
    }
    interpretation = Interpretability(
        schema_version=INTERPRETABILITY_SCHEMA_VERSION,
        headline=InterpretabilityHeadline(
            title="rNPV / ROI screening result",
            result=headline_result,
            plain_language=(
                f"The module completed with {recommendation}; decision grade is {decision_grade}, "
                "and the result remains screening decision support rather than a validated "
                "forecast."
            ),
            status=headline_status,
            basis=basis,
        ),
        metrics=metrics,
        steps=steps,
        evidence=evidence,
        assumptions=assumptions,
        uncertainty=_build_uncertainty(result),
        limitations=tuple(limitations),
        counterfactuals=(
            InterpretabilityCounterfactual(
                change=(
                    "Delay all modeled commercial launches by one year without moving the patent "
                    "clock."
                ),
                result=f"Native modeled value loss: {_display(delay_value, currency)}.",
                meaning=(
                    "Launch delay consumes protected commercial time; this is a modeled economic "
                    "counterfactual, not an observed launch forecast."
                ),
            ),
            InterpretabilityCounterfactual(
                change="Replace unsupported or synthetic critical inputs with verified evidence.",
                result=(
                    "Decision-grade status would be re-evaluated; the resulting economics are "
                    "unknown."
                ),
                meaning=(
                    "Evidence replacement could change both status and model inputs, so the "
                    "adapter does not predict an upgraded conclusion."
                ),
            ),
        ),
        lineage=_build_lineage(result),
        extensions=extensions,
    )
    escaped, escaped_count = _escape_html(interpretation.model_dump(mode="python"))
    if not escaped_count:
        return interpretation
    escaped["limitations"].append(
        {
            "code": "HTML_ESCAPED",
            "severity": "WARNING",
            "message": (
                "HTML-looking text from native inputs was escaped in the shared contract; the "
                "unchanged native value remains available under payload."
            ),
            "field_path": "interpretability",
        }
    )
    escaped["extensions"]["html_escaped_field_count"] = escaped_count
    return Interpretability.model_validate(escaped)
