from __future__ import annotations

import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from labrador_roi.contracts import (
    MODULE_RESPONSE_ADAPTER,
    ModuleRunRequest,
    execute_request,
    output_json_schema,
)
from labrador_roi.engine import AnalysisResult
from labrador_roi.interpretability import Interpretability, build_interpretability

ROOT = Path(__file__).resolve().parents[1]


def _strict_load(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON number {value}")

    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)


@pytest.fixture(scope="module")
def success_output() -> dict[str, Any]:
    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 4
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    return MODULE_RESPONSE_ADAPTER.dump_python(response, mode="json")


def test_success_response_contains_required_interpretability(success_output) -> None:
    assert success_output["status"] == "ok"
    interpretation = success_output["interpretability"]
    assert interpretation["schema_version"] == "1.0.0"
    assert set(interpretation) == {
        "schema_version",
        "headline",
        "metrics",
        "steps",
        "evidence",
        "assumptions",
        "uncertainty",
        "limitations",
        "counterfactuals",
        "lineage",
        "extensions",
    }


def test_checked_in_output_example_contains_and_validates_interpretability() -> None:
    example = _strict_load(ROOT / "examples" / "output.json")
    assert "interpretability" in example
    Draft202012Validator(output_json_schema()).validate(example)
    native = AnalysisResult.model_validate(example["payload"])
    assert example["interpretability"] == build_interpretability(native).model_dump(mode="json")


def test_success_schema_rejects_missing_interpretability() -> None:
    example = _strict_load(ROOT / "examples" / "output.json")
    without_interpretability = deepcopy(example)
    without_interpretability.pop("interpretability", None)

    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(output_json_schema()).validate(without_interpretability)


def test_all_interpretability_ids_are_unique_and_references_resolve(success_output) -> None:
    interpretation = success_output["interpretability"]
    collections = {
        name: {item["id"] for item in interpretation[name]}
        for name in ("metrics", "steps", "evidence", "assumptions")
    }
    for name, ids in collections.items():
        assert len(ids) == len(interpretation[name]), f"duplicate {name} ID"

    for item in [*interpretation["metrics"], *interpretation["steps"]]:
        assert set(item["evidence_ids"]) <= collections["evidence"]
        assert set(item["assumption_ids"]) <= collections["assumptions"]
    for interval in interpretation["uncertainty"]["intervals"]:
        assert interval["metric_id"] in collections["metrics"]


def test_typed_contract_rejects_duplicate_and_dangling_ids(success_output) -> None:
    duplicate = deepcopy(success_output["interpretability"])
    duplicate["metrics"][1]["id"] = duplicate["metrics"][0]["id"]
    with pytest.raises(ValueError, match="metrics IDs must be unique"):
        Interpretability.model_validate(duplicate)

    dangling = deepcopy(success_output["interpretability"])
    dangling["steps"][0]["evidence_ids"] = ["evidence.does.not.exist"]
    with pytest.raises(ValueError, match="unresolved evidence IDs"):
        Interpretability.model_validate(dangling)


def test_numeric_metrics_have_units_and_are_tagged(success_output) -> None:
    interpretation = success_output["interpretability"]
    untagged_paths = {
        item["field_path"]
        for item in interpretation["limitations"]
        if item["code"] == "UNTAGGED_VALUE"
    }
    for metric in interpretation["metrics"]:
        if isinstance(metric["value"], (int, float)) and not isinstance(metric["value"], bool):
            assert metric["unit"]
        assert (
            metric["evidence_ids"]
            or metric["assumption_ids"]
            or f"interpretability.metrics.{metric['id']}" in untagged_paths
        )


def test_unknown_values_remain_null_and_emit_limitation(success_output) -> None:
    interpretation = success_output["interpretability"]
    payer_steps = [
        item for item in interpretation["steps"] if "payer_affordability" in item["method"]
    ]
    assert payer_steps
    annual_oop_inputs = [
        input_value
        for step in payer_steps
        for input_value in step["inputs"]
        if input_value["path"].endswith("annual_patient_oop")
    ]
    assert annual_oop_inputs
    assert all(item["value"] is None for item in annual_oop_inputs)
    assert any(
        item["code"] == "PATIENT_OOP_NET_PRICE_PROXY" and item["severity"] == "ERROR"
        for item in interpretation["limitations"]
    )

    patent_step = next(item for item in interpretation["steps"] if item["method"] == "patent_clock")
    regulatory_exclusivity = next(
        item
        for item in patent_step["inputs"]
        if item["path"].endswith("regulatory_exclusivity_end_year")
    )
    assert regulatory_exclusivity["value"] is None
    assert any(
        item["code"] == "UNKNOWN_ASSUMPTION_VALUE"
        and item["field_path"] == regulatory_exclusivity["path"]
        for item in interpretation["limitations"]
    )

    unknown_access_id = interpretation["extensions"]["indication_metric_ids"]["syn-indication-b"][
        "accessible_patients"
    ]
    unknown_access = next(
        item for item in interpretation["metrics"] if item["id"] == unknown_access_id
    )
    assert unknown_access["value"] is None
    assert unknown_access["unit"] == "patients"
    assert any(
        item["code"] == "UNKNOWN_METRIC_VALUE"
        and item["field_path"] == f"interpretability.metrics.{unknown_access['id']}.value"
        for item in interpretation["limitations"]
    )


def test_success_artifact_is_strict_finite_json(success_output) -> None:
    rendered = json.dumps(success_output, allow_nan=False)
    strict = json.loads(
        rendered,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON number {value}")
        ),
    )

    def assert_finite(value: Any) -> None:
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for item in value.values():
                assert_finite(item)
        elif isinstance(value, list):
            for item in value:
                assert_finite(item)

    assert_finite(strict)


def test_adapter_maps_native_values_without_recalculation(success_output) -> None:
    payload = success_output["payload"]
    interpretation = success_output["interpretability"]
    native_steps = {item["step_id"]: item for item in payload["calculation_steps"]}
    native_step_ids = interpretation["extensions"]["native_step_ids"]
    shared_steps = {native_step_ids[item["id"]]: item for item in interpretation["steps"]}
    assert shared_steps.keys() == native_steps.keys()
    for step_id, native in native_steps.items():
        shared = shared_steps[step_id]
        assert shared["formula"] == native["formula"]
        assert shared["result"] == {"value": native["result"], "unit": native["unit"]}

    metrics = {item["id"]: item["value"] for item in interpretation["metrics"]}
    summary_mapping = {
        "metric.rnpv.deterministic": "deterministic_rnpv",
        "metric.rnpv.mean": "simulated_mean_rnpv",
        "metric.rnpv.p10": "p10_rnpv",
        "metric.rnpv.p50": "p50_rnpv",
        "metric.rnpv.p90": "p90_rnpv",
        "metric.rnpv.probability_positive": "probability_positive_rnpv",
        "metric.revenue.peak_annual": "peak_annual_net_revenue",
        "metric.revenue.peak_annual_p50": "peak_annual_net_revenue_p50",
        "metric.revenue.peak_year": "peak_annual_net_revenue_year",
        "metric.cash_at_risk.p50": "peak_cash_at_risk_p50",
        "metric.protected_years.effective": "effective_protected_years",
        "metric.launch_delay.value_lost": "value_lost_per_launch_delay_year",
    }
    for metric_id, summary_key in summary_mapping.items():
        assert metrics[metric_id] == payload["summary"][summary_key]
    assert interpretation["extensions"]["value_decomposition"] == payload["value_decomposition"]
    assert (
        interpretation["extensions"]["annual_ledger"] == payload["cash_flow"]["annual_cash_flows"]
    )


def test_adapter_is_deterministic_and_ignores_runtime_timestamp(success_output) -> None:
    native = AnalysisResult.model_validate(success_output["payload"])
    changed_timestamp = native.model_copy(update={"generated_at": "2099-01-01T00:00:00Z"})
    first = build_interpretability(native).model_dump(mode="json")
    second = build_interpretability(changed_timestamp).model_dump(mode="json")
    assert first == second


def test_missing_evidence_detail_is_null_and_explicitly_partial(success_output) -> None:
    interpretation = success_output["interpretability"]
    assert interpretation["evidence"]
    assert all(
        item["locator"] is None and item["quote"] is None for item in interpretation["evidence"]
    )
    assert any(
        item["code"] == "INTERPRETABILITY_PARTIAL"
        and item["field_path"] == "interpretability.evidence"
        for item in interpretation["limitations"]
    )


def test_pricing_step_evidence_does_not_cross_indications(success_output) -> None:
    interpretation = success_output["interpretability"]
    evidence_paths = {
        item["evidence_id"]: item["field_paths"]
        for item in interpretation["extensions"]["native_evidence"]
    }
    initial = next(
        item
        for item in interpretation["steps"]
        if item["method"] == "pricing:syn-indication-a:value_ceiling"
    )
    expansion = next(
        item
        for item in interpretation["steps"]
        if item["method"] == "pricing:syn-indication-b:value_ceiling"
    )
    initial_paths = {
        path for evidence_id in initial["evidence_ids"] for path in evidence_paths[evidence_id]
    }
    expansion_paths = {
        path for evidence_id in expansion["evidence_ids"] for path in evidence_paths[evidence_id]
    }
    assert initial_paths
    assert all(path.startswith("program.initial_indication") for path in initial_paths)
    assert expansion_paths
    assert all("expansion_indications[id=syn-indication-b]" in path for path in expansion_paths)


def test_price_and_access_metrics_link_their_actual_native_evidence(success_output) -> None:
    interpretation = success_output["interpretability"]
    metrics = {item["id"]: item for item in interpretation["metrics"]}
    evidence_paths = {
        item["evidence_id"]: item["field_paths"]
        for item in interpretation["extensions"]["native_evidence"]
    }

    for indication_id, expected_comparable_id in (
        ("syn-indication-a", "syn-comp-net-anchor"),
        ("syn-indication-b", "syn-comp-expansion-net"),
    ):
        metric_ids = interpretation["extensions"]["indication_metric_ids"][indication_id]
        price_metric = metrics[metric_ids["selected_price"]]
        price_paths = {
            path
            for evidence_id in price_metric["evidence_ids"]
            for path in evidence_paths[evidence_id]
        }
        assert any(
            path.startswith(f"comparables.comparables[id={expected_comparable_id}].price.evidence")
            for path in price_paths
        )
        assert any(".access.evidence." in path for path in price_paths)
        assert all(".income_bands[" not in path for path in price_paths)

        access_metric = metrics[metric_ids["accessible_patients"]]
        access_paths = {
            path
            for evidence_id in access_metric["evidence_ids"]
            for path in evidence_paths[evidence_id]
        }
        assert any(".access.evidence." in path for path in access_paths)


def test_valid_odd_indication_ids_produce_safe_collision_free_shared_ids() -> None:
    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 2
    request_data["program"]["initial_indication"]["indication_id"] = "A-B"
    request_data["program"]["expansion_indications"][0]["indication_id"] = "A_B"
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    output = MODULE_RESPONSE_ADAPTER.dump_python(response, mode="json")

    ids = [
        item["id"]
        for name in ("metrics", "steps", "evidence", "assumptions")
        for item in output["interpretability"][name]
    ]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", item) for item in ids)


def test_colon_prefix_indication_ids_keep_metric_links_scoped() -> None:
    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 2
    request_data["program"]["initial_indication"]["indication_id"] = "A"
    request_data["program"]["expansion_indications"][0]["indication_id"] = "A:B"
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    interpretation = response.interpretability.model_dump(mode="json")
    evidence_paths = {
        item["evidence_id"]: item["field_paths"]
        for item in interpretation["extensions"]["native_evidence"]
    }
    assumptions = {item["id"]: item for item in interpretation["assumptions"]}
    metrics = {item["id"]: item for item in interpretation["metrics"]}

    initial_ids = interpretation["extensions"]["indication_metric_ids"]["A"]
    for metric_id in initial_ids.values():
        metric = metrics[metric_id]
        paths = {
            path for evidence_id in metric["evidence_ids"] for path in evidence_paths[evidence_id]
        }
        assert all("expansion_indications[id=A:B]" not in path for path in paths)
        assumption_paths = {
            assumptions[assumption_id]["path"] for assumption_id in metric["assumption_ids"]
        }
        assert all("pricing:A:B:" not in path for path in assumption_paths)


def test_material_typed_inputs_are_exposed_without_fabricated_provenance(
    success_output,
) -> None:
    assumptions = {item["path"]: item for item in success_output["interpretability"]["assumptions"]}
    expected_paths = {
        "input.program.patent.filing_year",
        "input.program.patent.base_term_years",
        "input.program.development.stage_costs",
        "input.program.development.stage_success_probabilities",
        "input.program.initial_indication.population.annual_incident_patients",
        "input.program.initial_indication.access.coverage_fraction",
    }
    assert expected_paths <= assumptions.keys()
    assert assumptions["input.program.patent.filing_year"]["unit"] == "year"
    assert assumptions["input.program.patent.filing_year"]["synthetic"] is True
    assert assumptions["input.program.patent.base_term_years"]["synthetic"] is None
    assert assumptions["input.program.development.stage_costs"]["unit"] == "USD"
    assert assumptions["input.program.development.stage_costs"]["synthetic"] is True
    expansion_costs = assumptions["input.program.development.assumptions.expansion_stage_costs"]
    assert expansion_costs["synthetic"] is True
    assert "selection rationale" in expansion_costs["basis"]
    assert (
        "assumptions[].basis"
        in success_output["interpretability"]["extensions"]["interpretability_partial_fields"]
    )


def test_duplicate_income_band_names_do_not_collapse_paths_or_evidence() -> None:
    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 2
    bands = request_data["program"]["initial_indication"]["income_bands"]
    bands[1]["name"] = bands[0]["name"]
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    interpretation = response.interpretability.model_dump(mode="json")

    population_share_assumptions = [
        item
        for item in interpretation["assumptions"]
        if item["path"].startswith("input.program.initial_indication.income_bands[")
        and item["path"].endswith(".population_share")
    ]
    assert len(population_share_assumptions) == len(bands)
    assert len({item["path"] for item in population_share_assumptions}) == len(bands)
    income_evidence_paths = {
        path
        for item in interpretation["extensions"]["native_evidence"]
        for path in item["field_paths"]
        if path.startswith("program.initial_indication.income_bands[")
    }
    assert len(income_evidence_paths) == len(bands)


def test_common_assumption_units_use_explicit_semantics(success_output) -> None:
    assumptions = success_output["interpretability"]["assumptions"]

    def units_for_suffix(suffix: str) -> set[str | None]:
        return {item["unit"] for item in assumptions if item["path"].endswith(suffix)}

    assert units_for_suffix(".cogs_per_full_dose_patient") == {"USD/patient-year"}
    assert units_for_suffix(".paid_units_per_patient") == {None}
    assert units_for_suffix(".wtp_per_qaly") == {"USD/QALY"}
    assert units_for_suffix(".loe_price_retention") == {"fraction"}
    assert units_for_suffix(".effective_exclusivity_end_year") == {"year"}


def test_explicit_synthetic_marker_and_linked_provenance_are_preserved(success_output) -> None:
    program_assumptions = [
        item
        for item in success_output["interpretability"]["assumptions"]
        if item["path"].startswith("input.program.assumptions.")
    ]
    assert program_assumptions
    assert all(item["synthetic"] is True for item in program_assumptions)

    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 2
    request_data["program"]["initial_indication"]["assumptions"]["synthetic"] = False
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    interpretation = response.interpretability.model_dump(mode="json")
    wtp = next(
        item
        for item in interpretation["assumptions"]
        if item["path"].endswith("initial_indication.assumptions.willingness_to_pay_per_qaly")
    )
    assert wtp["synthetic"] is True
    assert any(
        item["code"] == "SYNTHETIC_STATUS_CONFLICT"
        and item["field_path"] == f"interpretability.assumptions.{wtp['id']}.synthetic"
        for item in interpretation["limitations"]
    )


def test_parent_assumption_does_not_inherit_child_evidence() -> None:
    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 2
    request_data["program"]["assumptions"]["willingness_to_pay_per_qaly"] = 123456
    request_data["program"]["assumptions"]["synthetic"] = False
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    interpretation = response.interpretability.model_dump(mode="json")
    assumption = next(
        item
        for item in interpretation["assumptions"]
        if item["path"] == "input.program.assumptions.willingness_to_pay_per_qaly"
    )
    assert assumption["synthetic"] is False
    assert not any(
        item["code"] == "SYNTHETIC_STATUS_CONFLICT"
        and item["field_path"] == f"interpretability.assumptions.{assumption['id']}.synthetic"
        for item in interpretation["limitations"]
    )


def test_nested_unknown_assumption_value_gets_a_limitation() -> None:
    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 2
    request_data["program"]["assumptions"]["custom_threshold"] = {"threshold": None}
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    interpretation = response.interpretability.model_dump(mode="json")
    path = "input.program.assumptions.custom_threshold"
    assumption = next(item for item in interpretation["assumptions"] if item["path"] == path)
    assert assumption["value"] == {"threshold": None}
    assert any(
        item["code"] == "UNKNOWN_ASSUMPTION_VALUE" and item["field_path"] == path
        for item in interpretation["limitations"]
    )


def test_html_looking_native_text_is_escaped_without_failing_the_run() -> None:
    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 2
    request_data["program"]["initial_indication"]["indication_id"] = "<b>x</b>"
    request_data["program"]["expansion_indications"][0]["indication_id"] = "&lt;b&gt;x&lt;/b&gt;"
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    output = MODULE_RESPONSE_ADAPTER.dump_python(response, mode="json")

    assert output["status"] == "ok"
    assert (
        output["payload"]["input_snapshot"]["program"]["initial_indication"]["indication_id"]
        == "<b>x</b>"
    )
    rendered = json.dumps(output["interpretability"])
    assert not re.search(r"</?[A-Za-z][^>]*>", rendered)
    assert any(item["code"] == "HTML_ESCAPED" for item in output["interpretability"]["limitations"])
    assert len(output["interpretability"]["extensions"]["indication_metric_ids"]) == len(
        output["payload"]["pricing"]
    )
    assert len(output["interpretability"]["extensions"]["critical_evidence_status"]) == len(
        output["payload"]["critical_evidence_status"]
    )

    request_data["program"]["initial_indication"]["indication_id"] = "<!--danger-->"
    comment_response = execute_request(ModuleRunRequest.model_validate(request_data))
    comment_output = MODULE_RESPONSE_ADAPTER.dump_python(comment_response, mode="json")
    assert "<!--danger-->" not in json.dumps(comment_output["interpretability"])
    assert any(
        item["code"] == "HTML_ESCAPED" for item in comment_output["interpretability"]["limitations"]
    )


def test_empty_native_steps_emit_a_limitation_instead_of_failing(success_output) -> None:
    native = AnalysisResult.model_validate(success_output["payload"])
    without_steps = native.model_copy(update={"calculation_steps": ()})
    interpretation = build_interpretability(without_steps)

    assert interpretation.steps == ()
    assert any(item.code == "NO_CALCULATION_STEPS" for item in interpretation.limitations)


def test_citation_only_evidence_stays_visible_in_common_claim(success_output) -> None:
    native = AnalysisResult.model_validate(success_output["payload"])
    citation_only = native.evidence_references[0].model_copy(
        update={
            "source_id": None,
            "source_url": None,
            "citation": "Authoritative citation without a durable identifier",
        }
    )
    interpretation = build_interpretability(
        native.model_copy(update={"evidence_references": (citation_only,)})
    )

    assert len(interpretation.evidence) == 1
    assert "Authoritative citation without a durable identifier" in interpretation.evidence[0].claim
    assert interpretation.evidence[0].source_id is None
    assert interpretation.evidence[0].source_url is None


def test_no_expansion_omits_expansion_lineage_with_limitation() -> None:
    request_data = _strict_load(ROOT / "examples" / "input.json")
    request_data["execution"]["simulations"] = 2
    request_data["program"]["expansion_indications"] = []
    response = execute_request(ModuleRunRequest.model_validate(request_data))
    interpretation = response.interpretability

    assert all(
        item.output_path != "payload.value_decomposition.expansion_increment_discounted_fcf"
        for item in interpretation.lineage
    )
    assert any(item.code == "LABEL_EXPANSION_NOT_APPLICABLE" for item in interpretation.limitations)


def test_degraded_transport_success_is_not_labeled_supported(success_output) -> None:
    assert success_output["status"] == "ok"
    assert success_output["payload"]["decision_grade"] == "NOT_DECISION_GRADE"
    headline = success_output["interpretability"]["headline"]
    assert headline["result"] == "NOT_DECISION_GRADE"
    assert headline["status"] in {"QUALIFIED", "INCONCLUSIVE"}
    assert "SYNTHETIC" in headline["basis"]


def test_all_error_warnings_are_preserved_as_limitations(success_output) -> None:
    expected = Counter(
        (item["code"], item["message"])
        for item in success_output["warnings"]
        if item["severity"] == "ERROR"
    )
    actual = Counter(
        (item["code"], item["message"])
        for item in success_output["interpretability"]["limitations"]
        if item["severity"] == "ERROR"
    )
    assert expected <= actual
    assert all(
        item["field_path"] == "payload.warnings"
        for item in success_output["interpretability"]["limitations"]
        if item["severity"] == "ERROR" and (item["code"], item["message"]) in expected
    )


def test_uncertainty_preserves_percentile_rng_and_correlation_semantics(success_output) -> None:
    payload = success_output["payload"]
    uncertainty = success_output["interpretability"]["uncertainty"]
    intervals = {item["metric_id"]: item for item in uncertainty["intervals"]}
    rnpv = intervals["metric.rnpv.p50"]
    assert (rnpv["low"], rnpv["central"], rnpv["high"]) == (
        payload["uncertainty"]["rnpv"]["p10"],
        payload["uncertainty"]["rnpv"]["p50"],
        payload["uncertainty"]["rnpv"]["p90"],
    )
    assert "scenario percentiles" in uncertainty["method"].lower()
    assert "not confidence intervals" in " ".join(uncertainty["limitations"]).lower()
    assert uncertainty["seed"] == payload["seed"]
    assert uncertainty["draws"] == payload["simulations"]

    design = success_output["interpretability"]["extensions"]["uncertainty_design"]
    assert design["rng_bit_generator"] == payload["uncertainty"]["rng_bit_generator"]
    assert design["numpy_version"] == payload["uncertainty"]["numpy_version"]
    assert (
        design["draw_order_contract_version"]
        == payload["uncertainty"]["draw_order_contract_version"]
    )
    assert (
        design["commercial_driver_correlation"]
        == payload["uncertainty"]["commercial_driver_correlation"]
    )
    assert design["assumptions"] == payload["simulation_assumptions"]


def test_required_rnpv_lineage_and_financial_context_are_present(success_output) -> None:
    interpretation = success_output["interpretability"]
    output_paths = {item["output_path"] for item in interpretation["lineage"]}
    assert {
        "payload.pricing[].annual_net_price_corridor",
        "payload.pricing[].access_estimates[].accessible_patients",
        "payload.access[].annual_persistence_rate",
        "payload.cash_flow.initial_approval_probability",
        "payload.value_decomposition.development_costs",
        "payload.cash_flow.effective_exclusivity_end_year",
        "payload.summary.value_lost_per_launch_delay_year",
        "payload.value_decomposition.expansion_increment_discounted_fcf",
    } <= output_paths

    contexts = interpretation["extensions"]["financial_metric_context"]
    financial_metric_ids = {
        item["id"]
        for item in interpretation["metrics"]
        if item["unit"].startswith(
            success_output["payload"]["input_snapshot"]["program"]["currency"]
        )
    }
    assert financial_metric_ids
    for metric_id in financial_metric_ids:
        assert contexts[metric_id]["currency"]
        assert contexts[metric_id]["valuation_year"] is not None
        assert "price_basis" in contexts[metric_id]
