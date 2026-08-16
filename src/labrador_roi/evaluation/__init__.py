"""Reality-grounding evaluation API."""

from labrador_roi.evaluation.reality_anchors import (
    AnchorBucket,
    AnchorResult,
    AnchorStatus,
    BucketCounts,
    LabradorRealityAdapter,
    RAEvaluationScenario,
    RealityAnchor,
    RealityReport,
    evaluate_reality_anchors,
    format_reality_report,
    load_reality_anchors,
)

__all__ = [
    "AnchorBucket",
    "AnchorResult",
    "AnchorStatus",
    "BucketCounts",
    "LabradorRealityAdapter",
    "RAEvaluationScenario",
    "RealityAnchor",
    "RealityReport",
    "evaluate_reality_anchors",
    "format_reality_report",
    "load_reality_anchors",
]
