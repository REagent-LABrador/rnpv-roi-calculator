"""Replay and verify a serialized LABrador analysis artifact.

The analysis result already carries the complete redacted input snapshot, seed,
simulation count, resolved uncertainty assumptions, schema version, and engine
version.  This module turns that audit material into an executable contract:
the artifact must reproduce the same deterministic outputs or verification
fails loudly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from labrador_roi.cashflow import ProgramCashFlowInputs
from labrador_roi.comparables import ComparableSet
from labrador_roi.engine import ENGINE_VERSION, SCHEMA_VERSION, AnalysisResult, analyze_program
from labrador_roi.models import ProgramInput
from labrador_roi.provenance import canonical_json
from labrador_roi.simulation import SimulationAssumptions


class ReplayVerificationError(ValueError):
    """Raised when an artifact is incomplete, incompatible, or non-reproducible."""


def _payload(value: AnalysisResult | BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _deterministic_payload(value: AnalysisResult | Mapping[str, Any]) -> dict[str, Any]:
    """Return replay-critical engine fields, excluding presentation envelopes and wall time."""

    payload = _payload(value)
    return {
        field_name: payload[field_name]
        for field_name in AnalysisResult.model_fields
        if field_name != "generated_at" and field_name in payload
    }


def replay_analysis(
    artifact: AnalysisResult | Mapping[str, Any],
    *,
    verify_outputs: bool = True,
) -> AnalysisResult:
    """Re-run an analysis artifact and verify its digest and deterministic outputs.

    ``verify_outputs=False`` is useful only when deliberately regenerating an
    artifact after reviewing an engine-version change.  The default is strict.
    """

    payload = _payload(artifact)
    artifact_version = payload.get("engine_version")
    if artifact_version != ENGINE_VERSION:
        raise ReplayVerificationError(
            f"engine version mismatch: artifact={artifact_version!r}, current={ENGINE_VERSION!r}"
        )
    artifact_schema_version = payload.get("schema_version")
    if artifact_schema_version != SCHEMA_VERSION:
        raise ReplayVerificationError(
            "schema version mismatch: "
            f"artifact={artifact_schema_version!r}, current={SCHEMA_VERSION!r}"
        )

    snapshot = payload.get("input_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ReplayVerificationError("artifact is missing a mapping input_snapshot")
    try:
        simulations = int(payload["simulations"])
        seed = int(payload["seed"])
        assumptions = SimulationAssumptions.model_validate(payload["simulation_assumptions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayVerificationError(
            "artifact is missing valid seed, simulations, or simulation assumptions"
        ) from exc

    if "program" in snapshot:
        try:
            program = ProgramInput.model_validate(snapshot["program"])
            comparables = ComparableSet.model_validate(snapshot.get("comparables", {}))
        except (TypeError, ValueError) as exc:
            raise ReplayVerificationError("program input snapshot is not replayable") from exc
        replayed = analyze_program(
            program,
            comparables,
            simulations=simulations,
            seed=seed,
            simulation_assumptions=assumptions,
        )
    elif "cashflow_inputs" in snapshot:
        try:
            cashflow_inputs = ProgramCashFlowInputs.model_validate(snapshot["cashflow_inputs"])
        except (TypeError, ValueError) as exc:
            raise ReplayVerificationError("cash-flow input snapshot is not replayable") from exc
        replayed = analyze_program(
            cashflow_inputs,
            simulations=simulations,
            seed=seed,
            simulation_assumptions=assumptions,
        )
    else:
        raise ReplayVerificationError(
            "input_snapshot must contain either program or cashflow_inputs"
        )

    expected_digest = payload.get("input_digest")
    if replayed.input_digest != expected_digest:
        raise ReplayVerificationError(
            f"input digest mismatch: artifact={expected_digest!r}, replay={replayed.input_digest!r}"
        )

    if verify_outputs and canonical_json(_deterministic_payload(payload)) != canonical_json(
        _deterministic_payload(replayed)
    ):
        raise ReplayVerificationError(
            "artifact outputs do not match a replay with the recorded inputs and RNG contract"
        )
    return replayed
