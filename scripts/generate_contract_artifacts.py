"""Regenerate portable schemas and the canonical synthetic request example."""

from __future__ import annotations

import json
from pathlib import Path

from labrador_roi.contracts import (
    CONTRACT_VERSION,
    MODULE_NAME,
    ModuleRunRequest,
    write_json_schemas,
)
from labrador_roi.simulation import SimulationAssumptions

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    program = json.loads((ROOT / "fixtures" / "demo_program.json").read_text(encoding="utf-8"))
    comparables_document = json.loads(
        (ROOT / "fixtures" / "demo_comparables.json").read_text(encoding="utf-8")
    )
    request = ModuleRunRequest.model_validate(
        {
            "contract_version": CONTRACT_VERSION,
            "module": MODULE_NAME,
            "request_id": "synthetic-roi-demo-001",
            "program": program,
            "comparables": comparables_document["comparables"],
            "execution": {
                "simulations": 128,
                "seed": 42,
                "simulation_assumptions": SimulationAssumptions().model_dump(mode="json"),
            },
        }
    )
    examples = ROOT / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    (examples / "input.json").write_text(
        json.dumps(request.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_json_schemas(ROOT / "schemas")


if __name__ == "__main__":
    main()
