from __future__ import annotations

import re
from urllib.parse import urlparse

from compete.logging_utils import get_logger, log_step
from compete.models import Competitor
from compete.state import AgentState

log = get_logger("intake")


def _guess_domain(name: str) -> str | None:
    slug = re.sub(r"[^a-z0-9]+", "", name.lower())
    if not slug:
        return None
    return f"{slug}.com"


def _canonical_url(name: str, domain: str | None) -> list[str]:
    if domain:
        host = domain if "://" in domain else f"https://{domain}"
        return [host]
    guessed = _guess_domain(name)
    return [f"https://{guessed}"] if guessed else []


def intake_node(state: AgentState) -> dict:
    """Validate & clean intake → Competitor objects (target = #0)."""
    company_name = (state.get("company_name") or "").strip()
    description = (state.get("description") or "").strip()
    seeds = state.get("seed_competitors") or []

    if not company_name:
        raise ValueError("company_name is required")
    if not description:
        raise ValueError("description is required")

    seen: set[str] = set()
    competitors: list[Competitor] = []

    def add(name: str, *, is_target: bool, desc: str = "") -> None:
        key = re.sub(r"\s+", " ", name.strip().lower())
        if not key or key in seen:
            return
        # Flag obviously empty / non-entity tokens
        if len(key) < 2 or key in {"n/a", "none", "unknown", "tbd"}:
            return
        seen.add(key)
        domain = _guess_domain(name)
        competitors.append(
            Competitor(
                name=name.strip(),
                description=desc,
                domain=domain,
                urls=_canonical_url(name, domain),
                is_target=is_target,
            )
        )

    add(company_name, is_target=True, desc=description)
    for seed in seeds:
        add(str(seed), is_target=False)

    if len(competitors) < 2:
        raise ValueError("Need the target company plus at least one distinct competitor")

    log_step(
        log,
        "intake.done",
        target=company_name,
        competitors=len(competitors),
        names=",".join(c.name for c in competitors),
    )
    return {"competitors": competitors}


def resolve_domain_hint(url: str) -> str | None:
    try:
        host = urlparse(url).netloc
        return host or None
    except Exception:  # noqa: BLE001
        return None
