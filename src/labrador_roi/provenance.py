"""Reproducible, secret-safe provenance helpers for analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from pydantic import BaseModel

REDACTED = "[REDACTED]"
SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"bearer[_-]?token|api[_-]?key|client[_-]?secret|cookie|credentials?|password|"
    r"private[_-]?key|secret|signature)(?:$|[_-])|"
    r"(?:^|[_-])authorization(?:$|[_-](?:header|token|credentials?)$)",
    re.IGNORECASE,
)
SECRET_VALUE = re.compile(
    r"(?:"
    r"gxl_[A-Za-z0-9_-]{12,}|"
    r"sk-[A-Za-z0-9_-]{12,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"gh[opsu]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"Bearer\s+[A-Za-z0-9._~+/-]{12,}"
    r")"
)


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    """Create a non-semantic identifier for one analysis run."""

    return f"run_{uuid4().hex}"


def _redact_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc or not parts.query:
        return value
    query = [
        (key, REDACTED if SENSITIVE_KEY.search(key) else item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def redact(value: Any, *, extra_secrets: Sequence[str] = ()) -> Any:
    """Return a JSON-compatible copy with credential names and values removed."""

    extra = tuple(secret for secret in extra_secrets if secret)

    def clean(item: Any, key: str | None = None) -> Any:
        if key is not None and SENSITIVE_KEY.search(key):
            return None if item is None else REDACTED
        if isinstance(item, BaseModel):
            return clean(item.model_dump(mode="json"))
        if is_dataclass(item) and not isinstance(item, type):
            return clean(asdict(item))
        if isinstance(item, Mapping):
            return {
                str(nested_key): clean(nested, str(nested_key))
                for nested_key, nested in item.items()
            }
        if isinstance(item, (list, tuple, set)):
            return [clean(nested) for nested in item]
        if isinstance(item, Enum):
            return clean(item.value)
        if isinstance(item, (datetime, Path)):
            return str(item)
        if isinstance(item, str):
            output = SECRET_VALUE.sub(REDACTED, _redact_url(item))
            for secret in extra:
                output = output.replace(secret, REDACTED)
            return output
        return item

    return clean(value)


def canonical_json(value: Any, *, extra_secrets: Sequence[str] = ()) -> str:
    """Serialize a sanitized object deterministically for hashing and storage."""

    return json.dumps(
        redact(value, extra_secrets=extra_secrets),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_digest(value: Any, *, extra_secrets: Sequence[str] = ()) -> str:
    """Hash the canonical, sanitized representation of an artifact."""

    payload = canonical_json(value, extra_secrets=extra_secrets).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
