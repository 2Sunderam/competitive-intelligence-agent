"""Intake requires Name|domain — no slug.com guessing."""

from __future__ import annotations

import pytest

from compete.nodes.intake import intake_node, normalize_domain, parse_name_domain


def test_parse_name_domain_accepts_url_and_host():
    assert parse_name_domain("Jamf|https://www.jamf.com") == ("Jamf", "jamf.com")
    assert parse_name_domain("Kandji|kandji.io") == ("Kandji", "kandji.io")


def test_parse_name_domain_requires_pipe_and_domain():
    with pytest.raises(ValueError, match="Name\\|domain"):
        parse_name_domain("Jamf")
    with pytest.raises(ValueError, match="invalid domain"):
        parse_name_domain("Jamf|notadomain")


def test_normalize_domain_strips_www_and_path():
    assert normalize_domain("https://www.jamf.com/products") == "jamf.com"


def test_intake_uses_provided_domains_not_guesses():
    out = intake_node(
        {
            "company_name": "Kandji|https://www.kandji.io",
            "description": "Apple MDM for fleets.",
            "seed_competitors": ["Jamf|https://www.jamf.com", "Mosyle|mosyle.com"],
        }
    )
    comps = {c.name: c for c in out["competitors"]}
    assert out["company_name"] == "Kandji"
    assert comps["Kandji"].domain == "kandji.io"
    assert comps["Kandji"].is_target is True
    assert comps["Jamf"].domain == "jamf.com"
    assert comps["Mosyle"].urls == ["https://mosyle.com"]
