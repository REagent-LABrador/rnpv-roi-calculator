"""Command-line entry point for the portable reality-anchor harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from labrador_roi.evaluation import evaluate_reality_anchors, format_reality_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LABrador against RA/I&I reality anchors")
    parser.add_argument("--anchors", type=Path, help="Optional alternate anchor JSON document")
    parser.add_argument("--json", action="store_true", help="Emit the typed report as JSON")
    args = parser.parse_args()
    report = evaluate_reality_anchors(anchors_path=args.anchors)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_reality_report(report))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
