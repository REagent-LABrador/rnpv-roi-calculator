from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from labrador_roi import __version__
from labrador_roi.cli import (
    DEMO_COMPARABLES_CSV,
    DEMO_COMPARABLES_JSON,
    DEMO_PROGRAM,
    DEMO_PROGRAM_B,
    app,
    comparable_summary,
    load_comparables,
    validate_comparables_payload,
)
from labrador_roi.engine import ENGINE_VERSION, SCHEMA_VERSION

runner = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload(output: str) -> dict:
    value = json.loads(output)
    assert isinstance(value, dict)
    return value


def test_example_manifest_is_explicitly_synthetic() -> None:
    result = runner.invoke(app, ["example", "--kind", "manifest", "--compact"])

    assert result.exit_code == 0, result.output
    payload = _payload(result.stdout)
    assert payload["synthetic"] is True
    assert payload["decision_grade"] == "NOT_DECISION_GRADE"
    assert "synthetic" in payload["warning"].lower()


def test_package_metadata_matches_the_engine_release() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == __version__ == ENGINE_VERSION


def test_cli_version_reports_package_engine_and_schema_versions() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0, result.output
    assert f"labrador {__version__}" in result.stdout
    assert f"engine {ENGINE_VERSION}" in result.stdout
    assert f"schema {SCHEMA_VERSION}" in result.stdout


def test_example_can_copy_agent_starter_inputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "starter"
    result = runner.invoke(
        app,
        ["example", "--output-dir", str(output_dir), "--compact"],
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result.stdout)
    assert len(payload["copied"]) == 4
    assert (output_dir / "demo_program.json").exists()
    assert (output_dir / "demo_program_b.json").exists()
    assert (output_dir / "demo_comparables.json").exists()
    assert (output_dir / "demo_comparables.csv").exists()


def test_validate_accepts_bundled_program_and_json_comparables() -> None:
    result = runner.invoke(
        app,
        [
            "validate",
            str(DEMO_PROGRAM),
            "--comparables",
            str(DEMO_COMPARABLES_JSON),
            "--compact",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result.stdout)
    assert payload["status"] == "ok"
    assert payload["program"]["program_id"] == "SYNTHETIC-LAB-001"
    assert len(payload["comparables"]["comparables"]) == 6


def test_compare_normalizes_flat_csv_without_losing_synthetic_warning() -> None:
    result = runner.invoke(
        app,
        ["compare", str(DEMO_COMPARABLES_CSV), "--compact"],
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result.stdout)
    assert payload["comparable_count"] == 6
    assert payload["synthetic_count"] == 6
    assert payload["decision_grade"] == "NOT_DECISION_GRADE"
    assert "confidential manufacturer net price" in payload["warning"]


def test_raw_csv_records_use_the_same_normalizer_as_file_loading() -> None:
    with DEMO_COMPARABLES_CSV.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))

    validated = validate_comparables_payload(raw_rows)

    assert len(validated.comparables) == 6
    assert validated.comparables[0].price.currency == "USD"


def test_comparable_statistics_never_pool_currency_or_price_basis() -> None:
    summary = comparable_summary(load_comparables(DEMO_COMPARABLES_JSON))
    groups = summary["price_statistics_by_currency_and_basis"]

    assert "price_statistics_unadjusted" not in summary
    assert {(item["currency"], item["price_basis"]) for item in groups} == {
        ("GBP", "LIST"),
        ("USD", "ESTIMATED_NET"),
        ("USD", "LIST"),
        ("USD", "PUBLIC_REIMBURSEMENT"),
    }


def test_validate_error_is_machine_readable(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"program_id": "missing-everything-else"}\n', encoding="utf-8")

    result = runner.invoke(app, ["validate", str(invalid), "--compact"])

    assert result.exit_code == 2
    payload = _payload(result.stdout)
    assert payload["status"] == "error"
    assert payload["operation"] == "validate"
    assert payload["error_type"] == "ValidationError"
    assert payload["errors"]


def test_cli_redacts_secrets_from_success_and_error_payloads(tmp_path: Path) -> None:
    secret = "gxl_" + "1234567890abcdefghijklmnopqrstuvwxyz"
    valid_payload = json.loads(DEMO_PROGRAM.read_text(encoding="utf-8"))
    valid_payload["assumptions"]["api_key"] = secret
    valid_path = tmp_path / "valid-with-secret.json"
    valid_path.write_text(json.dumps(valid_payload), encoding="utf-8")

    success = runner.invoke(app, ["validate", str(valid_path), "--compact"])
    assert success.exit_code == 0, success.output
    assert secret not in success.stdout
    assert "[REDACTED]" in success.stdout

    invalid_path = tmp_path / "invalid-with-secret.json"
    invalid_path.write_text(
        json.dumps({"program_id": "incomplete", "api_key": secret}),
        encoding="utf-8",
    )
    failure = runner.invoke(app, ["validate", str(invalid_path), "--compact"])
    assert failure.exit_code == 2
    assert secret not in failure.stdout


def test_analyze_demo_is_seed_reproducible() -> None:
    arguments = [
        "analyze",
        str(DEMO_PROGRAM),
        "--comparables",
        str(DEMO_COMPARABLES_JSON),
        "--simulations",
        "25",
        "--seed",
        "7",
        "--compact",
    ]

    first = runner.invoke(app, arguments)
    second = runner.invoke(app, arguments)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    first_payload.pop("generated_at")
    second_payload.pop("generated_at")
    assert first_payload == second_payload
    assert first_payload["decision_grade"] == "NOT_DECISION_GRADE"
    assert first_payload["pricing"][0]["annual_net_price_corridor"] is not None
    assert first_payload["pricing"][1]["annual_net_price_corridor"] is not None
    assert first_payload["access"][1]["patient_affordability_rate"] > 0
    assert (
        max(
            row["expansion_active_patients"]
            for row in first_payload["cash_flow"]["annual_cash_flows"]
        )
        > 0
    )
    assert first_payload["value_decomposition"]["protected_net_revenue"] > 0


def test_portfolio_returns_standardized_transparently_sorted_rows() -> None:
    result = runner.invoke(
        app,
        [
            "portfolio",
            str(DEMO_PROGRAM),
            str(DEMO_PROGRAM_B),
            "--comparables",
            str(DEMO_COMPARABLES_JSON),
            "--simulations",
            "20",
            "--seed",
            "11",
            "--sort-by",
            "p50_rnpv",
            "--descending",
            "--compact",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result.stdout)
    assert payload["program_count"] == 2
    assert payload["sort"] == {"field": "p50_rnpv", "direction": "descending"}
    assert [row["screening_rank"] for row in payload["rows"]] == [1, 2]
    assert payload["rows"][0]["p50_rnpv"] >= payload["rows"][1]["p50_rnpv"]
    assert all(row["decision_grade"] == "NOT_DECISION_GRADE" for row in payload["rows"])
    assert payload["currency"] == "USD"
    assert payload["valuation_year"] == 2026
    assert all(row["currency"] == "USD" for row in payload["rows"])
    assert all(row["valuation_year"] == 2026 for row in payload["rows"])
    assert all("warning_count" in row for row in payload["rows"])
    assert all("critical_evidence_gaps" in row for row in payload["rows"])
    assert all("/" not in row["source_file"] for row in payload["rows"])
    assert "screening" in payload["warning"].lower()


def test_portfolio_refuses_mixed_currency_or_valuation_year(tmp_path: Path) -> None:
    program_b = json.loads(DEMO_PROGRAM_B.read_text(encoding="utf-8"))
    program_b["currency"] = "GBP"
    program_b["initial_indication"]["currency"] = "GBP"
    mixed_currency = tmp_path / "mixed-currency.json"
    mixed_currency.write_text(json.dumps(program_b), encoding="utf-8")

    currency_result = runner.invoke(
        app,
        [
            "portfolio",
            str(DEMO_PROGRAM),
            str(mixed_currency),
            "--comparables",
            str(DEMO_COMPARABLES_JSON),
            "--simulations",
            "2",
            "--compact",
        ],
    )
    assert currency_result.exit_code == 2
    assert "share one currency" in currency_result.stdout

    program_b = json.loads(DEMO_PROGRAM_B.read_text(encoding="utf-8"))
    program_b["valuation_year"] = 2027
    mixed_year = tmp_path / "mixed-year.json"
    mixed_year.write_text(json.dumps(program_b), encoding="utf-8")
    year_result = runner.invoke(
        app,
        [
            "portfolio",
            str(DEMO_PROGRAM),
            str(mixed_year),
            "--comparables",
            str(DEMO_COMPARABLES_JSON),
            "--simulations",
            "2",
            "--compact",
        ],
    )
    assert year_result.exit_code == 2
    assert "share one valuation_year" in year_result.stdout
