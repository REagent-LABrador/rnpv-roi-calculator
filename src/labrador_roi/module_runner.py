"""File-in/file-out CLI for multi-repository pipeline orchestration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from labrador_roi.contracts import (
    MODULE_RESPONSE_ADAPTER,
    ModuleErrorDetail,
    ModuleRunFailure,
    ModuleRunRequest,
    execute_request,
    write_json_schemas,
)
from labrador_roi.provenance import redact

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run the standalone rNPV/ROI JSON module.",
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_request(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )


def _error_details(exc: Exception) -> tuple[ModuleErrorDetail, ...]:
    if isinstance(exc, ValidationError):
        return tuple(
            ModuleErrorDetail(
                type=str(item["type"]),
                path=tuple(item["loc"]),
                message=str(item["msg"]),
            )
            for item in exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        )
    return (
        ModuleErrorDetail(
            type=type(exc).__name__,
            message=str(redact(str(exc))) or "Module execution failed",
        ),
    )


def _serialize_response(response: Any) -> str:
    payload = MODULE_RESPONSE_ADAPTER.dump_python(response, mode="json")
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(contents)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


@app.command("run")
def run_module(
    input_path: Annotated[Path, typer.Option("--input", help="Versioned request JSON.")],
    output_path: Annotated[Path, typer.Option("--output", help="Response JSON artifact.")],
) -> None:
    """Validate one request, run the engine, and always write a response envelope."""

    request_id = "unknown"
    exit_code = 0
    try:
        raw_request = _read_request(input_path)
        if isinstance(raw_request, dict) and isinstance(raw_request.get("request_id"), str):
            request_id = raw_request["request_id"] or "unknown"
        request = ModuleRunRequest.model_validate(raw_request)
        request_id = request.request_id
        response: Any = execute_request(request)
        serialized = _serialize_response(response)
    except Exception as exc:
        exit_code = 2
        response = ModuleRunFailure(request_id=request_id, errors=_error_details(exc))
        serialized = _serialize_response(response)

    try:
        _write_atomic(output_path, serialized)
    except OSError as exc:
        typer.echo(f"Unable to write module response: {redact(str(exc))}", err=True)
        raise typer.Exit(code=2) from exc

    if exit_code:
        typer.echo(f"ROI module failed; structured error written to {output_path}", err=True)
        raise typer.Exit(code=exit_code)


@app.command("write-schemas")
def write_schemas(
    directory: Annotated[
        Path,
        typer.Option("--directory", help="Directory for generated schema artifacts."),
    ] = Path("schemas"),
) -> None:
    """Regenerate the checked-in Draft 2020-12 schemas."""

    input_path, output_path = write_json_schemas(directory)
    typer.echo(f"Wrote {input_path} and {output_path}")


if __name__ == "__main__":
    app()
