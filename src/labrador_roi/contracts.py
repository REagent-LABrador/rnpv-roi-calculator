"""Versioned JSON transport contracts for the standalone ROI module."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from labrador_roi import __version__
from labrador_roi.comparables import ComparableSet
from labrador_roi.engine import (
    ENGINE_VERSION,
    SCHEMA_VERSION,
    AnalysisResult,
    EvidenceReference,
    analyze_program,
)
from labrador_roi.interpretability import Interpretability, build_interpretability
from labrador_roi.models import (
    ComparableTherapy,
    ProgramInput,
    WarningRecord,
)
from labrador_roi.simulation import SimulationAssumptions

CONTRACT_VERSION = "1.0.0"
MODULE_NAME = "rnpv_roi_calculator"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
INPUT_SCHEMA_ID = f"urn:reagent-labrador:{MODULE_NAME}:input:{CONTRACT_VERSION}"
OUTPUT_SCHEMA_ID = f"urn:reagent-labrador:{MODULE_NAME}:output:{CONTRACT_VERSION}"


def _reject_nonfinite_numbers(value: Any, path: str = "$") -> None:
    """Reject non-standard JSON numbers before Pydantic or the engine sees them."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, f"{path}[{index}]")


class ModuleExecution(BaseModel):
    """Deterministic execution settings supplied explicitly by the orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulations: int = Field(ge=1, le=100_000)
    seed: int = Field(ge=0, le=4_294_967_295)
    simulation_assumptions: SimulationAssumptions = Field(default_factory=SimulationAssumptions)


class ModuleRunRequest(BaseModel):
    """Self-contained input for one ROI/rNPV calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"]
    module: Literal["rnpv_roi_calculator"]
    request_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")
    program: ProgramInput
    comparables: list[ComparableTherapy]
    execution: ModuleExecution

    @model_validator(mode="before")
    @classmethod
    def require_standard_json_numbers(cls, value: Any) -> Any:
        _reject_nonfinite_numbers(value)
        return value

    @model_validator(mode="after")
    def validate_module_scope(self) -> ModuleRunRequest:
        if len(self.program.expansion_indications) > 1:
            raise ValueError("contract v1 supports at most one label expansion")
        if self.program.patent.base_term_years != 20:
            raise ValueError("contract v1 requires the modeled 20-year patent term")
        ComparableSet(comparables=self.comparables)
        return self


class ModuleArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    media_type: str | None = None


class ModuleErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(min_length=1)
    path: tuple[str | int, ...] = ()
    message: str = Field(min_length=1)


class ModuleRunSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    module: Literal["rnpv_roi_calculator"] = MODULE_NAME
    module_version: str = __version__
    engine_version: str = ENGINE_VERSION
    engine_schema_version: str = SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    status: Literal["ok"] = "ok"
    payload: AnalysisResult
    interpretability: Interpretability
    artifacts: tuple[ModuleArtifact, ...] = ()
    warnings: tuple[WarningRecord, ...]
    provenance: tuple[EvidenceReference, ...]


class ModuleRunFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    module: Literal["rnpv_roi_calculator"] = MODULE_NAME
    module_version: str = __version__
    engine_version: str = ENGINE_VERSION
    engine_schema_version: str = SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    status: Literal["error"] = "error"
    payload: None = None
    artifacts: tuple[ModuleArtifact, ...] = ()
    warnings: tuple[WarningRecord, ...] = ()
    provenance: tuple[EvidenceReference, ...] = ()
    errors: tuple[ModuleErrorDetail, ...] = Field(min_length=1)


ModuleRunResponse = Annotated[
    ModuleRunSuccess | ModuleRunFailure,
    Field(discriminator="status"),
]
MODULE_RESPONSE_ADAPTER = TypeAdapter(ModuleRunResponse)


def execute_request(request: ModuleRunRequest) -> ModuleRunSuccess:
    """Execute the existing validated engine behind the portable request contract."""

    result = analyze_program(
        request.program,
        ComparableSet(comparables=request.comparables),
        simulations=request.execution.simulations,
        seed=request.execution.seed,
        simulation_assumptions=request.execution.simulation_assumptions,
    )
    return ModuleRunSuccess(
        request_id=request.request_id,
        payload=result,
        interpretability=build_interpretability(result),
        warnings=result.warnings,
        provenance=result.evidence_references,
    )


def input_json_schema() -> dict[str, Any]:
    schema = ModuleRunRequest.model_json_schema(mode="validation")
    schema.update(
        {
            "$schema": JSON_SCHEMA_DIALECT,
            "$id": INPUT_SCHEMA_ID,
            "title": "rNPV ROI Calculator Input v1",
        }
    )
    return schema


def output_json_schema() -> dict[str, Any]:
    schema = MODULE_RESPONSE_ADAPTER.json_schema(mode="serialization")
    schema.update(
        {
            "$schema": JSON_SCHEMA_DIALECT,
            "$id": OUTPUT_SCHEMA_ID,
            "title": "rNPV ROI Calculator Output v1",
        }
    )
    return schema


def write_json_schemas(directory: Path) -> tuple[Path, Path]:
    """Write deterministic checked-in schema artifacts from the typed contracts."""

    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "input.schema.json"
    output_path = directory / "output.schema.json"
    input_path.write_text(
        json.dumps(input_json_schema(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps(output_json_schema(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return input_path, output_path
