import json
from copy import deepcopy
from pathlib import Path

import pytest

import labrador_roi.replay as replay_module
from labrador_roi.comparables import ComparableSet
from labrador_roi.engine import analyze_program
from labrador_roi.models import ProgramInput
from labrador_roi.replay import ReplayVerificationError, replay_analysis

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _analysis_payload() -> dict[str, object]:
    program = ProgramInput.model_validate_json((FIXTURES / "demo_program.json").read_text())
    raw_comparables = json.loads((FIXTURES / "demo_comparables.json").read_text())
    comparables = ComparableSet.model_validate({"comparables": raw_comparables["comparables"]})
    return analyze_program(program, comparables, simulations=25, seed=314).model_dump(mode="json")


def test_serialized_analysis_replays_from_its_recorded_contract() -> None:
    payload = _analysis_payload()

    replayed = replay_analysis(payload)

    assert replayed.input_digest == payload["input_digest"]
    assert replayed.run_id == payload["run_id"]
    assert replayed.uncertainty.model_dump(mode="json") == payload["uncertainty"]


def test_replay_rejects_tampered_output_even_when_inputs_are_unchanged() -> None:
    payload = _analysis_payload()
    tampered = deepcopy(payload)
    tampered["summary"]["p50_rnpv"] += 1  # type: ignore[index]

    with pytest.raises(ReplayVerificationError, match="outputs do not match"):
        replay_analysis(tampered)


def test_replay_accepts_a_non_engine_interpretability_envelope() -> None:
    payload = _analysis_payload()
    payload["interpretability"] = {
        "status_legend": ["presentation metadata is not an engine output"]
    }

    replayed = replay_analysis(payload)

    assert replayed.input_digest == payload["input_digest"]


def test_replay_rejects_an_engine_version_mismatch() -> None:
    payload = _analysis_payload()
    payload["engine_version"] = "obsolete-engine"

    with pytest.raises(ReplayVerificationError, match="engine version mismatch"):
        replay_analysis(payload)


def test_replay_rejects_a_schema_version_mismatch_before_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _analysis_payload()
    payload["schema_version"] = "obsolete-schema"

    def fail_if_recomputed(*args: object, **kwargs: object) -> None:
        pytest.fail("schema compatibility must be checked before recomputation")

    monkeypatch.setattr(replay_module, "analyze_program", fail_if_recomputed)

    with pytest.raises(ReplayVerificationError, match="schema version mismatch"):
        replay_analysis(payload)
