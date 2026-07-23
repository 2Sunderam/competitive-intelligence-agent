from __future__ import annotations

from langchain_openai import ChatOpenAI

from compete.config import Settings
from compete.models import (
    ClusteringResult,
    EntityResolutionResult,
    ExtractionResult,
    GapAnalysisResult,
)


EXTRACT_SYSTEM = """You extract grounded competitive-intelligence claims from ONE document.

Research focus (from intake — this defines the product category we care about):
{research_focus}

Companies under analysis (attribute every claim to exactly one of these):
{roster}

Rules:
- Only extract claims about pricing, features, UX, support, or positioning.
- Every claim MUST include a verbatim quote copied exactly from the document.
- Never invent quotes. If you cannot find a supporting quote, skip the claim.
- Set `company` to the company the claim is ABOUT, spelled exactly as in the list
  above. A comparison thread discusses several products: file each claim under the
  product it describes, not under the document's main subject.
- If a claim is about a company that is not in the list, skip it entirely.
- Do not mix information that is not in this document.
- Keep claim text concise (one sentence).
- Only extract claims about these companies as products in THIS research focus /
  category — not a different org that shares a similar name.
- DROP off-topic content even if a company name appears: wrong industry, nonprofit
  vs SaaS product, agency/marketing firm vs software product, job postings/salary bands
  presented as product pricing, unrelated LinkedIn company pages, or any page that is
  clearly not about the product category in the research focus.
- If the document is mostly or entirely off-category, return an empty claims list.
- Prefer a few high-relevance claims over many noisy ones.
"""


def build_extract_prompt(
    *,
    competitor: str,
    document_text: str,
    url: str,
    research_description: str,
    target_company: str = "",
    category_keywords: list[str] | None = None,
    roster: list[str] | None = None,
) -> str:
    keywords = [k.strip() for k in (category_keywords or []) if k and k.strip()]
    focus_parts = [
        f"Target company: {target_company or 'n/a'}",
        f"Product / category description: {research_description.strip() or 'n/a'}",
    ]
    if keywords:
        focus_parts.append("Category keywords: " + ", ".join(keywords))
    research_focus = "\n".join(focus_parts)
    names = [n for n in (roster or []) if n] or [competitor]
    system = EXTRACT_SYSTEM.format(
        research_focus=research_focus,
        roster="\n".join(f"- {n}" for n in names),
    )
    return (
        f"{system}\n\n"
        f"This document was retrieved while researching: {competitor}\n"
        f"Source URL: {url}\n\n"
        f"Document:\n{document_text[:12000]}\n\n"
        "Return structured claims only, each attributed via `company`. "
        "Omit anything that does not match the research focus."
    )


ENTITY_RESOLVE_SYSTEM = """You resolve competitor entities for competitive research.
Given a target company description and a list of competitor names, return canonical
identifiers so search does not pick the wrong org with the same name.

For each competitor (including the target):
- official_domain: apex domain only, e.g. fireflies.ai (no scheme/path)
- g2_product_slug: G2 slug if known/likely, e.g. fireflies-ai (else best guess)
- linkedin_company_slug: LinkedIn /company/ slug if known/likely (else best guess)
- aliases: alternate product/company names used in reviews
- category_keywords: 3-6 keywords from the target description that disambiguate
  this product category (shared across competitors is fine)

Rules:
- Prefer well-known SaaS product identities matching the description category.
- Do not invent unrelated industries.
- If unsure of a slug, still provide the best lowercase hyphenated guess from the name.
"""


SYNTH_SYSTEM = """You synthesize a company profile from structured claims.
Rules:
- Group by dimension.
- Keep conflicting sentiments side by side — never resolve them into one verdict.
- Reference claim_ids only; do not invent new claims or quotes.
"""


COMPARE_SYSTEM = """You compare a target company against its competitors using
only the grounded claims supplied below.

Produce two lists of gaps:
- competitor_has_target_lacks: a capability, pricing model, integration or
  strength evidenced for one or more COMPETITORS with no equivalent claim for
  the target company.
- target_has_competitor_lacks: the same in reverse.

Rules:
- Every gap MUST cite the claim_ids that evidence it. A gap with no claim_ids
  is invalid — drop it instead.
- Only cite claim_ids that appear in the input. Never invent an ID.
- Describe the substantive difference ("competitors ship a native Git branch
  integration; no such claim exists for the target"), not the shape of the
  data ("has more claims in this dimension").
- Absence of evidence is not evidence of absence — phrase gaps as "no grounded
  claim found for X", never as "X does not exist".
- Prefer 3-8 well-evidenced gaps per direction over an exhaustive list.
"""


CLUSTER_SYSTEM = """You cluster complaint / unmet-need claims across companies
into recurring pain points.

Rules:
- Group claims that describe the SAME underlying problem even when the wording
  differs ("pricing jumps at 10 seats" and "gets expensive fast as the team
  grows" are one cluster).
- Every cluster MUST list the claim_ids of its members. Only use claim_ids from
  the input; never invent one.
- A cluster needs at least one claim. Prefer fewer, meaningful clusters over
  one cluster per claim — a cluster containing a single claim is only worth
  emitting when the complaint is substantive on its own.
- label: a short noun phrase naming the pain point.
- summary: one sentence stating the shared problem.
- Do not invent complaints that are not in the claims provided.
"""


REPORT_SYSTEM = """You write a competitive brief in Markdown.
Rules:
- Narrate only what is already computed in the inputs.
- Do not introduce new claims, quotes, or facts.
- Include sections: Overview, Per-company profiles, Comparison, Market gaps / pain points, Evidence notes.
"""


def make_extract_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_extract_model,
        api_key=settings.openai_api_key or None,
        temperature=0,
    )


def make_reason_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_reason_model,
        api_key=settings.openai_api_key or None,
        temperature=0,
    )


async def extract_claims_from_document(
    llm: ChatOpenAI,
    *,
    competitor: str,
    document_text: str,
    url: str,
    research_description: str = "",
    target_company: str = "",
    category_keywords: list[str] | None = None,
    roster: list[str] | None = None,
) -> ExtractionResult:
    structured = llm.with_structured_output(ExtractionResult)
    prompt = build_extract_prompt(
        competitor=competitor,
        document_text=document_text,
        url=url,
        research_description=research_description,
        target_company=target_company,
        category_keywords=category_keywords,
        roster=roster,
    )
    result = await structured.ainvoke(prompt)
    if isinstance(result, ExtractionResult):
        return result
    return ExtractionResult.model_validate(result)


async def cluster_pain_points_with_llm(
    llm: ChatOpenAI,
    *,
    claims: list[dict],
    domain_description: str = "",
) -> ClusteringResult:
    """Group complaint claims into pain points. Input is already filtered to
    negative/mixed sentiment claims across every company."""
    structured = llm.with_structured_output(ClusteringResult)
    lines = "\n".join(
        f"- {c['claim_id']} | company={c['competitor']} | dimension={c['dimension']} | "
        f"sentiment={c['sentiment']} | {c['text']}"
        for c in claims
    )
    prompt = (
        f"{CLUSTER_SYSTEM}\n\n"
        f"Domain: {domain_description or 'n/a'}\n\n"
        f"Complaint claims:\n{lines}\n\n"
        "Return clusters grouping these claim_ids."
    )
    result = await structured.ainvoke(prompt)
    if isinstance(result, ClusteringResult):
        return result
    return ClusteringResult.model_validate(result)


async def analyze_gaps_with_llm(
    llm: ChatOpenAI,
    *,
    target_company: str,
    domain_description: str,
    target_claims: list[dict],
    competitor_claims: dict[str, list[dict]],
) -> GapAnalysisResult:
    """Feature-level gap analysis in both directions, grounded in claim_ids."""
    structured = llm.with_structured_output(GapAnalysisResult)

    def render(claims: list[dict]) -> str:
        if not claims:
            return "  (no grounded claims)"
        return "\n".join(
            f"  - {c['claim_id']} | {c['dimension']} | {c['sentiment']} | {c['text']}"
            for c in claims
        )

    blocks = [f"TARGET — {target_company}:\n{render(target_claims)}"]
    for name, claims in competitor_claims.items():
        blocks.append(f"COMPETITOR — {name}:\n{render(claims)}")

    prompt = (
        f"{COMPARE_SYSTEM}\n\n"
        f"Domain: {domain_description or 'n/a'}\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn gaps in both directions, each citing claim_ids."
    )
    result = await structured.ainvoke(prompt)
    if isinstance(result, GapAnalysisResult):
        return result
    return GapAnalysisResult.model_validate(result)


async def resolve_entities_with_llm(
    llm: ChatOpenAI,
    *,
    target_name: str,
    description: str,
    competitor_names: list[str],
) -> EntityResolutionResult:
    structured = llm.with_structured_output(EntityResolutionResult)
    names = ", ".join(competitor_names)
    prompt = (
        f"{ENTITY_RESOLVE_SYSTEM}\n\n"
        f"Target company: {target_name}\n"
        f"Target description: {description}\n"
        f"All entities to resolve (include target): {names}\n\n"
        "Return one entity object per name."
    )
    result = await structured.ainvoke(prompt)
    if isinstance(result, EntityResolutionResult):
        return result
    return EntityResolutionResult.model_validate(result)
