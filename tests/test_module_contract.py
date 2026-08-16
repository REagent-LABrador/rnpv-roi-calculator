from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from labrador_roi.contracts import input_json_schema, output_json_schema
from labrador_roi.module_runner import app
from labrador_roi.replay import replay_analysis

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _strict_load(path: Path) -> dict:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON number {value}")

    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)


def _request(tmp_path: Path, *, simulations: int = 2) -> Path:
    payload = _strict_load(ROOT / "examples" / "input.json")
    payload["execution"]["simulations"] = simulations
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_checked_in_schemas_are_valid_and_match_typed_contracts() -> None:
    stored_input = _strict_load(ROOT / "schemas" / "input.schema.json")
    stored_output = _strict_load(ROOT / "schemas" / "output.schema.json")

    Draft202012Validator.check_schema(stored_input)
    Draft202012Validator.check_schema(stored_output)
    assert stored_input == input_json_schema()
    assert stored_output == output_json_schema()


def test_golden_input_and_output_validate_against_published_schemas() -> None:
    Draft202012Validator(input_json_schema()).validate(
        _strict_load(ROOT / "examples" / "input.json")
    )
    Draft202012Validator(output_json_schema()).validate(
        _strict_load(ROOT / "examples" / "output.json")
    )


def test_run_writes_schema_valid_success_and_replays(tmp_path: Path) -> None:
    request_path = _request(tmp_path)
    output_path = tmp_path / "result.json"

    result = runner.invoke(
        app,
        ["run", "--input", str(request_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert result.stderr == ""
    payload = _strict_load(output_path)
    Draft202012Validator(output_json_schema()).validate(payload)
    assert payload["status"] == "ok"
    assert payload["request_id"] == "synthetic-roi-demo-001"
    assert payload["payload"]["seed"] == 42
    assert payload["warnings"] == payload["payload"]["warnings"]
    assert payload["provenance"] == payload["payload"]["evidence_references"]
    replayed = replay_analysis(payload["payload"])
    assert replayed.run_id == payload["payload"]["run_id"]


def test_invalid_input_writes_schema_valid_error_and_exits_nonzero(tmp_path: Path) -> None:
    request_path = _request(tmp_path)
    request = _strict_load(request_path)
    request["execution"]["simulations"] = 0
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "error.json"

    result = runner.invoke(
        app,
        ["run", "--input", str(request_path), "--output", str(output_path)],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "structured error written" in result.stderr
    payload = _strict_load(output_path)
    Draft202012Validator(output_json_schema()).validate(payload)
    assert payload["status"] == "error"
    assert payload["payload"] is None
    assert payload["request_id"] == "synthetic-roi-demo-001"
    assert payload["errors"]


def test_nonstandard_json_numbers_fail_closed_without_leaking_invalid_json(
    tmp_path: Path,
) -> None:
    request_path = _request(tmp_path)
    valid = request_path.read_text(encoding="utf-8")
    request_path.write_text(
        valid.replace('"cogs_per_full_dose_patient": 7200', '"cogs_per_full_dose_patient": NaN'),
        encoding="utf-8",
    )
    output_path = tmp_path / "nonfinite-error.json"

    result = runner.invoke(
        app,
        ["run", "--input", str(request_path), "--output", str(output_path)],
    )

    assert result.exit_code == 2
    payload = _strict_load(output_path)
    Draft202012Validator(output_json_schema()).validate(payload)
    assert payload["status"] == "error"
    assert "non-standard JSON numeric constant" in payload["errors"][0]["message"]


def test_nonfinite_string_assumption_cannot_produce_a_success_artifact(tmp_path: Path) -> None:
    request_path = _request(tmp_path)
    request = _strict_load(request_path)
    request["program"]["initial_indication"]["assumptions"]["cogs_per_full_dose_patient"] = (
        "Infinity"
    )
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "string-nonfinite-error.json"

    result = runner.invoke(
        app,
        ["run", "--input", str(request_path), "--output", str(output_path)],
    )

    assert result.exit_code == 2
    payload = _strict_load(output_path)
    Draft202012Validator(output_json_schema()).validate(payload)
    assert payload["status"] == "error"


def test_fixed_request_and_seed_reproduce_engine_owned_fields(tmp_path: Path) -> None:
    request_path = _request(tmp_path, simulations=8)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = runner.invoke(
        app,
        ["run", "--input", str(request_path), "--output", str(first_path)],
    )
    second = runner.invoke(
        app,
        ["run", "--input", str(request_path), "--output", str(second_path)],
    )

    assert first.exit_code == second.exit_code == 0
    first_payload = _strict_load(first_path)["payload"]
    second_payload = _strict_load(second_path)["payload"]
    first_deterministic = deepcopy(first_payload)
    second_deterministic = deepcopy(second_payload)
    first_deterministic.pop("generated_at")
    second_deterministic.pop("generated_at")
    assert first_deterministic == second_deterministic
