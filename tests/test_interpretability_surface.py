from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from labrador_roi.cli import (
    DEMO_COMPARABLES_JSON,
    DEMO_PROGRAM,
    app,
    build_interpretability_manifest,
    load_comparables,
    load_program,
    reality_anchor_surface,
    run_analysis,
)

runner = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def analysis_result():
    return run_analysis(
        load_program(DEMO_PROGRAM),
        load_comparables(DEMO_COMPARABLES_JSON),
        simulations=20,
        seed=7,
    )


def test_manifest_exposes_decision_context_and_reconciles_outputs(analysis_result) -> None:
    manifest = build_interpretability_manifest(analysis_result)

    legend = {item["kind"]: item["meaning"] for item in manifest["status_legend"]}
    assert set(legend) == {
        "MODEL_OUTPUT",
        "CITED_REALITY_ANCHOR",
        "CONFIGURATION_CHECK",
        "FALSIFICATION_CONTROL",
    }
    assert "not model validation" in legend["CITED_REALITY_ANCHOR"]

    decision = manifest["decision_status"]
    assert decision["decision_grade"] == "NOT_DECISION_GRADE"
    assert decision["critical_evidence_gaps"]
    assert "independently validated" in decision["interpretation"]

    input_record = manifest["input_record"]
    assert input_record["full_snapshot_included"] is True
    assert input_record["currency"] == "USD"
    assert input_record["base_year"] == 2026
    assert input_record["valuation_year"] == 2026
    assert input_record["input_digest"].startswith("sha256:")

    assert manifest["price_context"]
    for context in manifest["price_context"]:
        assert context["price_basis"] == "ESTIMATED_NET"
        assert context["currency"] == "USD"
        assert context["valuation_year"] == 2026
        assert context["anchor_price_years"]
        assert context["year_basis_aligned"] is True

    oop_context = manifest["patient_oop_context"]
    assert oop_context
    assert oop_context[0]["basis"] == "MANUFACTURER_NET_PROXY"
    assert oop_context[0]["decision_grade_eligible"] is False

    patent = manifest["patent_clock"]
    assert patent["shared_across_indications"] is True
    assert patent["filing_year"] == 2024
    assert patent["initial_launch_year"] == 2031
    assert patent["expansion_launch_years"] == [2034]
    assert patent["patent_expiry_year"] == 2045.5

    simulation = manifest["simulation_design"]
    assert simulation["seed"] == 7
    assert simulation["draws"] == 20
    assert "persistence_multiplier" in simulation["sampled_drivers"]
    assert "not confidence intervals" in simulation["interpretation"]
    assert simulation["rng_contract"]["generator"] == "numpy.random.default_rng"
    assert (
        simulation["rng_contract"]["bit_generator"] == analysis_result.uncertainty.rng_bit_generator
    )
    assert simulation["rng_contract"]["numpy_version"] == analysis_result.uncertainty.numpy_version
    assert (
        simulation["rng_contract"]["draw_order_contract_version"]
        == analysis_result.uncertainty.draw_order_contract_version
    )
    assert "shared draw" in simulation["rng_contract"]["commercial_correlation"]
    assert "sequentially" in simulation["rng_contract"]["development_path"]
    assert "seed plus JSON alone" in simulation["rng_contract"]["replay_requirement"]

    reconciliation = manifest["output_reconciliation"]
    assert reconciliation["status"] == "PASS"
    assert len(reconciliation["checks"]) >= 4
    assert all(check["status"] == "PASS" for check in reconciliation["checks"])
    assert "not external validation" in reconciliation["interpretation"]


def test_reality_anchor_adapter_preserves_typed_bucket_results(monkeypatch) -> None:
    @dataclass(frozen=True)
    class FakeAnchorResult:
        id: str
        bucket: str
        status: str
        actual: float
        expected_band: tuple[float, float]
        reason: str

    @dataclass(frozen=True)
    class FakeRealityReport:
        results: tuple[FakeAnchorResult, ...]
        bucket_counts: dict[str, dict[str, int]]

    fake_report = FakeRealityReport(
        results=(
            FakeAnchorResult(
                id="ra-pos-chain",
                bucket="COMPUTATION",
                status="PASS",
                actual=0.12,
                expected_band=(0.08, 0.18),
                reason="inside cited band",
            ),
        ),
        bucket_counts={"COMPUTATION": {"PASS": 1, "FAIL": 0, "SKIP": 0}},
    )
    monkeypatch.setitem(
        sys.modules,
        "labrador_roi.evaluation",
        SimpleNamespace(evaluate_reality_anchors=lambda: fake_report),
    )

    surface = reality_anchor_surface(enabled=True)

    assert surface["status"] == "REPORTED"
    assert surface["report"]["results"][0]["id"] == "ra-pos-chain"
    assert surface["report"]["results"][0]["bucket"] == "COMPUTATION"
    assert surface["report"]["bucket_counts"]["COMPUTATION"]["PASS"] == 1
    assert "not model validation" in surface["interpretation"]


def test_cli_analysis_adds_interpretation_without_calling_range_checks() -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            str(DEMO_PROGRAM),
            "--comparables",
            str(DEMO_COMPARABLES_JSON),
            "--simulations",
            "5",
            "--seed",
            "17",
            "--no-reality-checks",
            "--compact",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    interpretation = payload["interpretability"]
    assert interpretation["simulation_design"]["seed"] == 17
    assert interpretation["simulation_design"]["draws"] == 5
    assert interpretation["reality_anchors"]["status"] == "NOT_RUN"
    assert interpretation["output_reconciliation"]["status"] == "PASS"


def test_replay_command_uses_public_replay_adapter(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "analysis.json"
    artifact.write_text(json.dumps({"input_snapshot": {"program": {}, "comparables": {}}}))
    monkeypatch.setitem(
        sys.modules,
        "labrador_roi.replay",
        SimpleNamespace(
            replay_analysis=lambda payload: {
                "status": "MATCH",
                "snapshot_seen": "input_snapshot" in payload,
                "meaning": "deterministic artifact consistency; not validation",
            }
        ),
    )

    result = runner.invoke(app, ["replay", str(artifact), "--compact"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "MATCH"
    assert payload["operation"] == "replay"
    assert payload["analysis"]["snapshot_seen"] is True
    assert "seed, draw count, uncertainty assumptions" in payload["verified_scope"]
    assert "engine-owned analysis fields" in payload["verified_scope"]
    assert "presentation envelope" in payload["excluded_scope"]
    assert "not external validation" in payload["interpretation"]


def test_dashboard_renders_interpretation_surfaces_and_runs_demo() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        dashboard = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        dashboard.run()
        assert not dashboard.exception
        dashboard.slider[0].set_value(100)
        dashboard.button[0].click().run(timeout=30)
    finally:
        sys.path.remove(str(PROJECT_ROOT / "src"))

    assert not dashboard.exception
    visible_text = "\n".join(
        str(element.value)
        for collection in (
            dashboard.markdown,
            dashboard.caption,
            dashboard.info,
            dashboard.warning,
            dashboard.error,
            dashboard.success,
        )
        for element in collection
    )
    assert "MODEL OUTPUT" in visible_text
    assert "SHARED ASSET CLOCK" in visible_text
    assert "Patient OOP basis is MANUFACTURER_NET_PROXY" in visible_text
    assert "not external validation" in visible_text
    assert "Cited reality anchors" in visible_text
