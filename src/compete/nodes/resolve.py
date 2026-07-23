from __future__ import annotations

import re
from urllib.parse import urlparse

from compete.config import get_settings
from compete.llm.client import make_reason_llm, resolve_entities_with_llm
from compete.logging_utils import get_logger, log_step
from compete.models import Competitor, ResolvedEntityDraft
from compete.sources.jina import company_slug
from compete.state import AgentState

log = get_logger("resolve")


def _clean_domain(raw: str) -> str | None:
    value = (raw or "").strip().casefold()
    if not value:
        return None
    value = value.replace("https://", "").replace("http://", "")
    value = value.split("/")[0].strip(".")
    if "." not in value:
        return None
    return value


def _clean_slug(raw: str, fallback_name: str) -> str:
    value = (raw or "").strip().casefold()
    value = value.replace("https://", "").replace("http://", "")
    if "g2.com/products/" in value:
        value = value.split("g2.com/products/", 1)[1]
    if "linkedin.com/company/" in value:
        value = value.split("linkedin.com/company/", 1)[1]
    value = value.split("/")[0]
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return value or company_slug(fallback_name)


def _apply_resolution(
    competitors: list[Competitor],
    drafts_by_name: dict[str, ResolvedEntityDraft],
) -> list[Competitor]:
    updated: list[Competitor] = []
    for comp in competitors:
        key = re.sub(r"\s+", " ", comp.name.strip().casefold())
        draft = drafts_by_name.get(key)
        domain = _clean_domain(draft.official_domain if draft else "") or comp.domain
        g2_slug = _clean_slug(draft.g2_product_slug if draft else "", comp.name)
        li_slug = _clean_slug(draft.linkedin_company_slug if draft else "", comp.name)
        aliases = list(draft.aliases) if draft else []
        keywords = list(draft.category_keywords) if draft else []
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
    """Thin LLM wrapper: lock competitors to canonical domain / G2 / LinkedIn URLs."""
    competitors: list[Competitor] = list(state.get("competitors") or [])
    description = state.get("description") or ""
    company_name = state.get("company_name") or ""
    log_step(log, "resolve_entities.start", companies=len(competitors))

    settings = get_settings()
    drafts_by_name: dict[str, ResolvedEntityDraft] = {}

    if settings.openai_api_key:
        try:
            llm = make_reason_llm(settings)
            result = await resolve_entities_with_llm(
                llm,
                target_name=company_name,
                description=description,
                competitor_names=[c.name for c in competitors],
            )
            for draft in result.entities:
                key = re.sub(r"\s+", " ", draft.name.strip().casefold())
                drafts_by_name[key] = draft
            log_step(log, "resolve_entities.llm_ok", resolved=len(drafts_by_name))
        except Exception as exc:  # noqa: BLE001
            log_step(log, "resolve_entities.llm_fallback", error=str(exc)[:160])
    else:
        log_step(log, "resolve_entities.no_llm", reason="OPENAI_API_KEY missing")

    for comp in competitors:
        key = re.sub(r"\s+", " ", comp.name.strip().casefold())
        if key not in drafts_by_name:
            drafts_by_name[key] = ResolvedEntityDraft(
                name=comp.name,
                official_domain=comp.domain or "",
                g2_product_slug=company_slug(comp.name),
                linkedin_company_slug=company_slug(comp.name),
            )

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
