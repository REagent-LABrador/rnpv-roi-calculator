from labrador_roi.provenance import REDACTED, canonical_json, redact, sha256_digest


def test_digest_is_independent_of_mapping_order() -> None:
    assert sha256_digest({"a": 1, "b": 2}) == sha256_digest({"b": 2, "a": 1})


def test_redact_covers_secret_keys_values_and_signed_urls() -> None:
    payload = {
        "api_key": "plain-secret",
        "note": "github_" + "pat_123456789012345678901234567890",
        "url": "https://example.test/data?access_token=abc123&limit=1",
    }

    sanitized = redact(payload)
    rendered = canonical_json(sanitized)

    assert sanitized["api_key"] == REDACTED
    assert sanitized["note"] == REDACTED
    assert "abc123" not in rendered
    assert "access_token=%5BREDACTED%5D" in sanitized["url"]


def test_extra_secret_is_removed_before_artifact_storage() -> None:
    rendered = canonical_json(
        {"message": "prefix private-value suffix"},
        extra_secrets=["private-value"],
    )
    assert "private-value" not in rendered
    assert REDACTED in rendered


def test_domain_authorization_fields_are_preserved_and_affect_digest() -> None:
    lower = {"authorization_rate": 0.4, "prior_authorization_pass_fraction": 0.5}
    higher = {"authorization_rate": 0.8, "prior_authorization_pass_fraction": 0.9}

    assert redact(lower) == lower
    assert sha256_digest(lower) != sha256_digest(higher)


def test_authorization_credential_header_is_still_redacted() -> None:
    sanitized = redact({"Authorization": "Bearer actual-credential-value"})

    assert sanitized["Authorization"] == REDACTED
