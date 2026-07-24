from __future__ import annotations

import re
from urllib.parse import urlparse

from compete.logging_utils import get_logger, log_step
from compete.models import Competitor
from compete.state import AgentState

log = get_logger("intake")


def parse_name_domain(raw: str) -> tuple[str, str]:
    """Parse ``Name|https://example.com`` or ``Name|example.com`` into (name, domain).

    Domain is required — we do not guess ``name.com`` from the label.
    """
    text = (raw or "").strip()
    if "|" not in text:
        raise ValueError(
            f"Expected 'Name|domain' (e.g. 'Jamf|https://www.jamf.com'), got: {raw!r}"
        )
    name, domain_raw = text.split("|", 1)
    name = name.strip()
    domain = normalize_domain(domain_raw)
    if not name:
        raise ValueError(f"Missing company name before '|': {raw!r}")
    if not domain:
        raise ValueError(f"Missing or invalid domain after '|': {raw!r}")
    return name, domain


def normalize_domain(raw: str) -> str | None:
    """Strip scheme/path/www → apex-ish host (``jamf.com``)."""
    value = (raw or "").strip()
    if not value:
        return None
    if "://" not in value:
        value = f"https://{value}"
    try:
        host = urlparse(value).netloc or urlparse(value).path.split("/")[0]
    except Exception:  # noqa: BLE001
        return None
    host = host.strip(".").casefold()
    if host.startswith("www."):
        host = host[4:]
    if "." not in host:
        return None
    return host


def intake_node(state: AgentState) -> dict:
    """Validate & clean intake → Competitor objects (target = #0).

    ``company_name`` and each ``seed_competitors`` entry must be
    ``Name|domain`` so homepage URLs are explicit, not guessed.
    """
    company_raw = (state.get("company_name") or "").strip()
    description = (state.get("description") or "").strip()
    seeds = state.get("seed_competitors") or []

    if not company_raw:
        raise ValueError("company_name is required (format: Name|domain)")
    if not description:
        raise ValueError("description is required")

    seen: set[str] = set()
    competitors: list[Competitor] = []

    def add(raw: str, *, is_target: bool, desc: str = "") -> None:
        name, domain = parse_name_domain(raw)
        key = re.sub(r"\s+", " ", name.lower())
        if key in seen:
            return
        if len(key) < 2 or key in {"n/a", "none", "unknown", "tbd"}:
            return
        seen.add(key)
        competitors.append(
            Competitor(
                name=name,
                description=desc,
                domain=domain,
                urls=[f"https://{domain}"],
                is_target=is_target,
            )
        )

    add(company_raw, is_target=True, desc=description)
    for seed in seeds:
        add(str(seed), is_target=False)

    if len(competitors) < 2:
        raise ValueError("Need the target company plus at least one distinct competitor")

    # Persist the bare target name for downstream prompts / report titles
    target = competitors[0]

    log_step(
        log,
        "intake.done",
        target=target.name,
        competitors=len(competitors),
        names=",".join(f"{c.name}({c.domain})" for c in competitors),
    )
    return {"competitors": competitors, "company_name": target.name}
