from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from compete.config import get_settings
from compete.llm.client import extract_claims_from_document, make_extract_llm
from compete.logging_utils import get_logger, log_step
from compete.models import Claim, FetchedDocument
from compete.state import AgentState, ExtractTask
from compete.store.evidence import get_evidence_store

log = get_logger("extract")


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").casefold())


def resolve_company(
    raw: str, roster: list[str], *, fallback: str | None = None
) -> str | None:
    """Map the model's ``company`` label onto a canonical roster name.

    Returns ``None`` when the label names something outside the roster — those
    claims are dropped rather than misfiled. An empty label falls back to the
    company the document was fetched for.
    """
    if not raw or not raw.strip():
        return fallback
    key = _key(raw)
    if not key:
        return fallback
    by_key = {_key(name): name for name in roster}
    if key in by_key:
        return by_key[key]
    # Tolerate "Linear (app)" / "Atlassian Jira" style answers
    for name_key, name in by_key.items():
        if name_key and (name_key in key or key in name_key):
            return name
    return None


async def extract_document(task: ExtractTask) -> dict:
    """One LLM extraction call per document → evidence.jsonl."""
    settings = get_settings()
    doc: FetchedDocument = task["document"]
    run_dir = Path(task["run_dir"])
    research_description = task.get("description") or ""
    target_company = task.get("company_name") or ""
    category_keywords = list(task.get("category_keywords") or [])
    roster = list(task.get("roster") or []) or [doc.competitor]
    # Shared instance: concurrent extract tasks must not each allocate ev_0001
    store = get_evidence_store(run_dir / "evidence.jsonl")
    log_step(
        log,
        "extract.start",
        competitor=doc.competitor,
        platform=doc.platform,
        url=(doc.url or "")[:90],
        chars=len(doc.text or ""),
        keywords=",".join(category_keywords[:5]),
    )

    evidence_ids: list[str] = []

    if not doc.text.strip():
        log_step(log, "extract.skip_empty", competitor=doc.competitor)
        return {"evidence_ids": evidence_ids}

    if not settings.openai_api_key:
        log_step(log, "extract.skip_no_key", competitor=doc.competitor)
        return {"evidence_ids": evidence_ids, "errors": [f"OPENAI_API_KEY missing; skipped extract for {doc.url}"]}

    try:
        llm = make_extract_llm(settings)
        result = await extract_claims_from_document(
            llm,
            competitor=doc.competitor,
            document_text=doc.text,
            url=doc.url,
            research_description=research_description,
            target_company=target_company,
            category_keywords=category_keywords,
            roster=roster,
        )

        # A "Linear vs Jira" thread produces claims about both. File each claim
        # under the company it is about, so a complaint about ClickUp inside a
        # Linear thread does not end up on Linear's profile.
        by_company: dict[str, list[Claim]] = defaultdict(list)
        unattributed = 0
        for draft in result.claims:
            company = resolve_company(draft.company, roster, fallback=doc.competitor)
            if company is None:
                unattributed += 1
                continue
            by_company[company].append(
                Claim(
                    claim_id="tmp",
                    dimension=draft.dimension,
                    text=draft.text,
                    quote=draft.quote,
                    sentiment=draft.sentiment,
                )
            )

        for company, drafts in by_company.items():
            record = store.append(
                competitor=company,
                source=doc.source.value,
                url=doc.url,
                platform=doc.platform,
                date=doc.date,
                source_text=doc.text,
                draft_claims=drafts,
            )
            if record:
                evidence_ids.append(record.evidence_id)
                log_step(
                    log,
                    "extract.done",
                    fetched_for=doc.competitor,
                    attributed_to=company,
                    evidence_id=record.evidence_id,
                    claims=len(record.claims),
                )
            else:
                log_step(
                    log,
                    "extract.no_valid_claims",
                    fetched_for=doc.competitor,
                    attributed_to=company,
                    drafts=len(drafts),
                )

        if not by_company:
            log_step(
                log,
                "extract.no_claims",
                competitor=doc.competitor,
                dropped_unattributed=unattributed,
            )
    except Exception as exc:  # noqa: BLE001
        log_step(log, "extract.error", competitor=doc.competitor, error=str(exc)[:160])
        return {"evidence_ids": evidence_ids, "errors": [f"extract failed for {doc.url}: {exc}"]}

    return {"evidence_ids": evidence_ids}


def fanout_extract(state: AgentState) -> list:
    from langgraph.types import Send

    run_dir = state["run_dir"]
    description = state.get("description") or ""
    company_name = state.get("company_name") or ""
    docs = state.get("documents") or []
    keywords_by_name = {
        c.name: list(c.category_keywords or [])
        for c in (state.get("competitors") or [])
    }
    if not docs:
        return []
    return [
        Send(
            "extract_map",
            {
                "document": d,
                "run_dir": run_dir,
                "description": description,
                "company_name": company_name,
                "category_keywords": keywords_by_name.get(d.competitor, []),
            },
        )
        for d in docs
    ]
