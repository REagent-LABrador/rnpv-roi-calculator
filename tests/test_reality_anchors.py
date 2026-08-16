from __future__ import annotations

import json
from pathlib import Path

import pytest

from labrador_roi.evaluation import (
    AnchorBucket,
    AnchorStatus,
    evaluate_reality_anchors,
    load_reality_anchors,
)


def test_portable_catalog_preserves_all_four_honest_buckets() -> None:
    anchors = load_reality_anchors()

    assert len(anchors) == 14
    assert len({anchor.anchor_id for anchor in anchors}) == 14
    counts = {bucket: sum(anchor.bucket == bucket for anchor in anchors) for bucket in AnchorBucket}
    assert counts == {
        AnchorBucket.COMPUTATION: 8,
        AnchorBucket.CROSS_SOURCE: 1,
        AnchorBucket.CONTROL: 1,
        AnchorBucket.CONFIG: 4,
    }
    assert all(anchor.claim and anchor.source for anchor in anchors)


def test_supported_live_measurements_pass_without_converting_gaps_to_passes() -> None:
    report = evaluate_reality_anchors()

    assert report.exit_code == 0
    assert report.model_counts.total == 10
    assert report.model_counts.evaluated == 6
    assert report.model_counts.passed == 6
    assert report.model_counts.failed == 0
    assert report.model_counts.skipped == 4
    assert report.bucket_counts[AnchorBucket.CONFIG].passed == 1
    assert report.bucket_counts[AnchorBucket.CONFIG].skipped == 3
    assert all(
        result.reason.strip() for result in report.results if result.status == AnchorStatus.SKIP
    )


def test_expected_ra_actuals_come_through_public_engine_contracts() -> None:
    report = evaluate_reality_anchors()

    assert report.result("pos_loa_chain").actual == pytest.approx(0.106505376144)
    assert report.result("addressable_us").actual == pytest.approx(393_668.1)
    assert report.result("mean_time_on_therapy").actual == pytest.approx(3.6796156341)
    assert report.result("net_price_oral").actual == pytest.approx(35_000)
    assert report.result("config.discount_rate").actual == pytest.approx(0.10)


def test_ten_x_prevalence_control_really_breaches_the_baseline_band() -> None:
    report = evaluate_reality_anchors()
    baseline = report.result("addressable_us")
    control = report.result("prevalence_10x_breaches")

    assert baseline.status == AnchorStatus.PASS
    assert control.status == AnchorStatus.PASS
    assert baseline.actual is not None
    assert control.actual is not None
    assert control.actual == pytest.approx(baseline.actual * 10)
    assert control.actual > baseline.expected[1]


def test_a_bad_live_result_fails_instead_of_widening_the_band(tmp_path: Path) -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "labrador_roi"
        / "evaluation"
        / "reality_anchors.json"
    )
    payload = json.loads(source_path.read_text())
    target = next(anchor for anchor in payload["anchors"] if anchor["id"] == "pos_loa_chain")
    target["expected"] = [0.20, 0.30]
    altered_path = tmp_path / "altered_anchors.json"
    altered_path.write_text(json.dumps(payload))

    report = evaluate_reality_anchors(anchors_path=altered_path)

    assert report.result("pos_loa_chain").status == AnchorStatus.FAIL
    assert report.result("pos_loa_chain").actual == pytest.approx(0.106505376144)
    assert report.exit_code == 1


def test_json_report_keeps_configuration_out_of_model_score() -> None:
    payload = evaluate_reality_anchors().to_dict()

    assert payload["model_counts"] == {
        "total": 10,
        "passed": 6,
        "failed": 0,
        "skipped": 4,
        "evaluated": 6,
    }
    assert payload["bucket_counts"]["config"]["total"] == 4
