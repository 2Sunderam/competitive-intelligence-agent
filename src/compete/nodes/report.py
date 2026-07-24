from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from compete.logging_utils import get_logger, log_step
from compete.models import (
    Claim,
    CompanyProfile,
    ComparisonTable,
    EvidenceRecord,
    PainPointCluster,
    Sentiment,
)
from compete.state import AgentState
from compete.store.evidence import get_evidence_store

log = get_logger("report")

SENTIMENT_MARK = {
    Sentiment.POSITIVE: "+",
    Sentiment.NEGATIVE: "-",
    Sentiment.MIXED: "~",
    Sentiment.NEUTRAL: "·",
}


def _short_quote(quote: str, limit: int = 220) -> str:
    q = " ".join((quote or "").split())
    if len(q) > limit:
        q = q[: limit - 1].rstrip() + "…"
    return q


def _cite(claim_id: str, lookup: dict[str, tuple[EvidenceRecord, Claim]]) -> str:
    """Render one claim as an evidence-backed bullet with its verbatim quote."""
    entry = lookup.get(claim_id)
    if not entry:
        return f"  - `{claim_id}` (claim not found)"
    record, claim = entry
    mark = SENTIMENT_MARK.get(claim.sentiment, "·")
    return (
        f"  - {mark} {claim.text} — “{_short_quote(claim.quote)}” "
        f"([{record.platform}]({record.url}) `{claim_id}`)"
    )


def _render_brief(
    *,
    company_name: str,
    description: str,
    profiles: list[CompanyProfile],
    comparison: ComparisonTable | None,
    clusters: list[PainPointCluster],
    skips: list,
    lookup: dict[str, tuple[EvidenceRecord, Claim]],
) -> str:
    records = {record.evidence_id: record for record, _claim in lookup.values()}
    platforms = Counter(record.platform for record, _claim in lookup.values())
    target = next((p for p in profiles if p.is_target), None)

    lines: list[str] = [
        f"# Competitive Brief: {company_name}",
        "",
        "## Company overview",
        "",
        f"**{company_name}** — {description or 'no description supplied'}",
        "",
        f"Analyzed **{len(profiles)}** companies across **{len(records)}** source documents "
        f"yielding **{len(lookup)}** grounded claims. Every claim below carries a verbatim "
        "quote and a source link; nothing in this report is asserted without one.",
        "",
    ]
    if platforms:
        spread = ", ".join(f"{name} ({count})" for name, count in platforms.most_common())
        lines.extend([f"Evidence by platform: {spread}.", ""])

    lines.extend(["## Competitor landscape", ""])
    for profile in profiles:
        tag = " *(target)*" if profile.is_target else ""
        claims = len(profile.claim_ids)
        dims = ", ".join(sorted(d.dimension.value for d in profile.dimensions)) or "none"
        lines.append(f"- **{profile.competitor}**{tag} — {claims} claims across: {dims}")
    lines.append("")

    lines.extend(["## Per-company findings", ""])
    for profile in profiles:
        tag = " (target)" if profile.is_target else ""
        lines.append(f"### {profile.competitor}{tag}")
        if not profile.dimensions:
            lines.extend(["", "_No grounded claims extracted — see source skips below._", ""])
            continue
        for dim in profile.dimensions:
            header = f"**{dim.dimension.value}**"
            if dim.conflicting_sentiments:
                sentiments = ", ".join(s.value for s in dim.conflicting_sentiments)
                header += f" — ⚠️ conflicting opinions preserved ({sentiments})"
            lines.append(f"- {header}")
            for claim_id in dim.claim_ids:
                lines.append(_cite(claim_id, lookup))
        lines.append("")

    lines.extend(["## Feature / positioning comparison", ""])
    if comparison and comparison.rows:
        dims = [d.value for d in comparison.dimensions]
        lines.append("Claim counts per dimension (cell = number of grounded claims).")
        lines.append("")
        lines.append("| Company | " + " | ".join(dims) + " |")
        lines.append("|---|" + "|".join(["---"] * len(dims)) + "|")
        for company, row in comparison.rows.items():
            cells = [str(len(row.get(d, []))) or "0" for d in dims]
            marker = " **(target)**" if target and company == target.competitor else ""
            lines.append(f"| {company}{marker} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.extend([f"### Gaps — competitors have, {company_name} lacks", ""])
        if comparison.gaps_competitor_has_target_lacks:
            for gap in comparison.gaps_competitor_has_target_lacks:
                lines.append(f"- **[{gap.dimension.value}]** {gap.description}")
                for claim_id in gap.claim_ids:
                    lines.append(_cite(claim_id, lookup))
        else:
            lines.append("_None identified from the collected evidence._")
        lines.append("")

        lines.extend([f"### Gaps — {company_name} has, competitors lack", ""])
        if comparison.gaps_target_has_competitor_lacks:
            for gap in comparison.gaps_target_has_competitor_lacks:
                lines.append(f"- **[{gap.dimension.value}]** {gap.description}")
                for claim_id in gap.claim_ids:
                    lines.append(_cite(claim_id, lookup))
        else:
            lines.append("_None identified from the collected evidence._")
        lines.append("")
    else:
        lines.extend(["_No comparison data._", ""])

    domain_wide = [c for c in clusters if c.scope == "domain_wide"]
    company_specific = [c for c in clusters if c.scope == "company_specific"]

    lines.extend(["## Domain-wide pain points", ""])
    if domain_wide:
        lines.append(
            "Complaints that recur across more than one company — these are unsolved "
            "problems in the category, not one vendor's bug."
        )
        lines.append("")
        for cluster in domain_wide:
            lines.append(f"- **{cluster.label}** — affects: {', '.join(cluster.companies)}")
            if cluster.summary:
                lines.append(f"  - {cluster.summary}")
            for claim_id in cluster.claim_ids:
                lines.append(_cite(claim_id, lookup))
    else:
        lines.append("_No complaint appeared across more than one company in this run._")
    lines.append("")

    lines.extend(["## Company-specific weaknesses", ""])
    if company_specific:
        for cluster in company_specific:
            lines.append(f"- **{cluster.label}** — {', '.join(cluster.companies)}")
            if cluster.summary:
                lines.append(f"  - {cluster.summary}")
            for claim_id in cluster.claim_ids:
                lines.append(_cite(claim_id, lookup))
    else:
        lines.append("_No company-specific complaints extracted._")
    lines.append("")

    lines.extend([f"## Opportunities for {company_name}", ""])
    opportunities = [c for c in domain_wide if target is None or target.competitor not in c.companies]
    if opportunities:
        lines.append(
            f"Pain points recurring across competitors with no matching complaint "
            f"recorded against {company_name} — the openings this evidence supports:"
        )
        lines.append("")
        for cluster in opportunities:
            lines.append(
                f"- **{cluster.label}** — unresolved at {', '.join(cluster.companies)} "
                f"(`{', '.join(cluster.claim_ids)}`)"
            )
    elif comparison and comparison.gaps_competitor_has_target_lacks:
        lines.append(
            f"No domain-wide pain point avoided {company_name}. The closest openings are "
            "the competitor capabilities listed under gaps above."
        )
    else:
        lines.append("_Not enough evidence collected to support an opportunity claim._")
    lines.append("")

    if skips:
        lines.extend(
            [
                "## Source skips",
                "",
                "Sources that were empty, blocked, rate-limited or off-topic. Logged and "
                "skipped — never substituted with invented content.",
                "",
            ]
        )
        for skip in skips:
            lines.append(f"- **{skip.competitor}** / `{skip.source.value}`: {skip.reason}")
        lines.append("")

    lines.extend(
        [
            "## Evidence notes",
            "",
            "`evidence.json` carries every claim and pain point with its verbatim quote, "
            "source URL, platform and the company it applies to. `evidence.jsonl` is the "
            "append-only log written during extraction, one line per source document.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(state: AgentState) -> dict:
    """Reduce L4: assemble competitive_brief.md + evidence.json (no new claims)."""
    run_dir = Path(state["run_dir"])
    store = get_evidence_store(run_dir / "evidence.jsonl")
    lookup = store.claim_lookup()
    profiles = state.get("company_profiles") or []
    comparison = state.get("comparison")
    clusters = state.get("pain_clusters") or []
    skips = state.get("skips") or []
    company_name = state.get("company_name") or "Target"

    # Target first, then companies with the most evidence
    profiles = sorted(profiles, key=lambda p: (not p.is_target, -len(p.claim_ids)))

    brief = _render_brief(
        company_name=company_name,
        description=state.get("description") or "",
        profiles=profiles,
        comparison=comparison,
        clusters=clusters,
        skips=skips,
        lookup=lookup,
    )
    brief_path = run_dir / "competitive_brief.md"
    brief_path.write_text(brief, encoding="utf-8")

    evidence_payload: dict = {
        "run_id": state.get("run_id"),
        "company_name": company_name,
        "description": state.get("description") or "",
        "claims": [],
        "pain_points": [c.model_dump() for c in clusters],
        "skips": [s.model_dump() for s in skips],
    }
    for record, claim in store.all_claims():
        evidence_payload["claims"].append(
            {
                "claim_id": claim.claim_id,
                "evidence_id": record.evidence_id,
                "competitor": record.competitor,
                "dimension": claim.dimension.value,
                "text": claim.text,
                "quote": claim.quote,
                "sentiment": claim.sentiment.value,
                "url": record.url,
                "platform": record.platform,
                "source": record.source.value,
                "date": record.date,
            }
        )

    evidence_json_path = run_dir / "evidence.json"
    evidence_json_path.write_text(json.dumps(evidence_payload, indent=2), encoding="utf-8")

    log_step(
        log,
        "report.done",
        brief=str(brief_path),
        evidence=str(evidence_json_path),
        claims=len(evidence_payload["claims"]),
        pain_points=len(clusters),
    )
    return {
        "brief_path": str(brief_path),
        "evidence_json_path": str(evidence_json_path),
    }
