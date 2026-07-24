from __future__ import annotations

import re

from compete.config import get_settings
from compete.llm.client import make_reason_llm, resolve_entities_with_llm
from compete.logging_utils import get_logger, log_step
from compete.models import Competitor, ResolvedEntityDraft
from compete.state import AgentState

log = get_logger("resolve")


def _slug_from_domain_or_name(domain: str | None, name: str) -> str:
    """Sanitize empty LLM slug: first domain label, else a hyphenated name."""
    if domain:
        label = domain.split(".", 1)[0]
        cleaned = re.sub(r"[^a-z0-9-]+", "-", label.casefold()).strip("-")
        if cleaned:
            return cleaned
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    return cleaned or "unknown"


def _clean_slug(raw: str, *, domain: str | None, name: str) -> str:
    value = (raw or "").strip().casefold()
    value = value.replace("https://", "").replace("http://", "")
    if "g2.com/products/" in value:
        value = value.split("g2.com/products/", 1)[1]
    if "linkedin.com/company/" in value:
        value = value.split("linkedin.com/company/", 1)[1]
    value = value.split("/")[0]
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return value or _slug_from_domain_or_name(domain, name)


def _apply_resolution(
    competitors: list[Competitor],
    drafts_by_name: dict[str, ResolvedEntityDraft],
) -> list[Competitor]:
    """Keep user domains; attach LLM-guessed G2 / LinkedIn / aliases / keywords."""
    updated: list[Competitor] = []
    for comp in competitors:
        key = re.sub(r"\s+", " ", comp.name.strip().casefold())
        draft = drafts_by_name[key]
        domain = comp.domain  # never overwritten — intake already locked it
        g2_slug = _clean_slug(draft.g2_product_slug, domain=domain, name=comp.name)
        li_slug = _clean_slug(draft.linkedin_company_slug, domain=domain, name=comp.name)
        aliases = list(draft.aliases)
        keywords = list(draft.category_keywords)
        if not keywords and comp.description:
            tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", comp.description.casefold())
            keywords = list(dict.fromkeys(tokens))[:6]

        g2_url = f"https://www.g2.com/products/{g2_slug}/reviews"
        linkedin_url = f"https://www.linkedin.com/company/{li_slug}"
        urls: list[str] = []
        if domain:
            urls.append(f"https://{domain}")
        urls.extend([g2_url, linkedin_url])

        updated.append(
            comp.model_copy(
                update={
                    "domain": domain,
                    "urls": list(dict.fromkeys(urls)),
                    "g2_url": g2_url,
                    "linkedin_url": linkedin_url,
                    "aliases": aliases,
                    "category_keywords": keywords,
                }
            )
        )
    return updated


async def resolve_entities_node(state: AgentState) -> dict:
    """Guess G2 / LinkedIn / aliases via LLM; domains stay as intake provided.

    Requires ``OPENAI_API_KEY`` — without a model there is nothing useful to do.
    """
    competitors: list[Competitor] = list(state.get("competitors") or [])
    description = state.get("description") or ""
    company_name = state.get("company_name") or ""
    log_step(log, "resolve_entities.start", companies=len(competitors))

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. This agent cannot resolve entities or "
            "extract claims without an LLM."
        )

    llm = make_reason_llm(settings)
    result = await resolve_entities_with_llm(
        llm,
        target_name=company_name,
        description=description,
        entities=[
            {"name": c.name, "domain": c.domain or ""}
            for c in competitors
        ],
    )
    drafts_by_name: dict[str, ResolvedEntityDraft] = {}
    for draft in result.entities:
        key = re.sub(r"\s+", " ", draft.name.strip().casefold())
        drafts_by_name[key] = draft

    missing = [
        c.name
        for c in competitors
        if re.sub(r"\s+", " ", c.name.strip().casefold()) not in drafts_by_name
    ]
    if missing:
        raise RuntimeError(
            "Entity resolution LLM omitted companies: " + ", ".join(missing)
        )

    log_step(log, "resolve_entities.llm_ok", resolved=len(drafts_by_name))
    resolved = _apply_resolution(competitors, drafts_by_name)
    for c in resolved:
        log_step(
            log,
            "resolve_entities.entity",
            name=c.name,
            domain=c.domain,
            g2=c.g2_url,
            linkedin=c.linkedin_url,
            keywords=",".join(c.category_keywords[:4]),
        )
    log_step(log, "resolve_entities.done", companies=len(resolved))
    return {"competitors": resolved}
