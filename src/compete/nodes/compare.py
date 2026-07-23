from __future__ import annotations

from pathlib import Path

from compete.config import get_settings
from compete.llm.client import analyze_gaps_with_llm, make_reason_llm
from compete.logging_utils import get_logger, log_step
from compete.models import (
    Claim,
    CompanyProfile,
    ComparisonTable,
    Dimension,
    EvidenceRecord,
    GapItem,
)
from compete.state import AgentState
from compete.store.evidence import get_evidence_store

log = get_logger("compare")


def _claim_payload(pairs: list[tuple[EvidenceRecord, Claim]]) -> list[dict]:
    return [
        {
            "claim_id": claim.claim_id,
            "dimension": claim.dimension.value,
            "sentiment": claim.sentiment.value,
            "text": claim.text,
        }
        for _record, claim in pairs
    ]


def _gaps_from_drafts(drafts, valid_ids: set[str]) -> list[GapItem]:
    """Keep only gaps that cite real claim_ids — an ungrounded gap is dropped."""
    gaps: list[GapItem] = []
    for draft in drafts:
        cited = [cid for cid in dict.fromkeys(draft.claim_ids) if cid in valid_ids]
        if not cited:
            continue
        gaps.append(
            GapItem(
                dimension=draft.dimension,
                description=draft.description.strip(),
                claim_ids=cited,
            )
        )
    return gaps


def _coverage_gaps(
    target: CompanyProfile, competitors: list[CompanyProfile]
) -> tuple[list[GapItem], list[GapItem]]:
    """Fallback: which dimensions have evidence for one side but not the other.

    This is a statement about crawl coverage, not about the products, so the
    wording says so explicitly. Used only when the LLM pass is unavailable.
    """
    gaps_comp: list[GapItem] = []
    gaps_target: list[GapItem] = []
    target_dims = {d.dimension for d in target.dimensions}

    for comp in competitors:
        comp_dims = {d.dimension for d in comp.dimensions}
        for dim in sorted(comp_dims - target_dims, key=lambda d: d.value):
            summary = next(d for d in comp.dimensions if d.dimension == dim)
            gaps_comp.append(
                GapItem(
                    dimension=dim,
                    description=(
                        f"{comp.competitor} has grounded {dim.value} claims; "
                        f"no {dim.value} evidence was collected for {target.competitor}"
                    ),
                    claim_ids=summary.claim_ids,
                )
            )
        for dim in sorted(target_dims - comp_dims, key=lambda d: d.value):
            summary = next(d for d in target.dimensions if d.dimension == dim)
            gaps_target.append(
                GapItem(
                    dimension=dim,
                    description=(
                        f"{target.competitor} has grounded {dim.value} claims; "
                        f"no {dim.value} evidence was collected for {comp.competitor}"
                    ),
                    claim_ids=summary.claim_ids,
                )
            )
    return gaps_comp, gaps_target


async def compare_node(state: AgentState) -> dict:
    """Reduce L2: cross-company comparison + feature-level gaps in both directions."""
    settings = get_settings()
    profiles: list[CompanyProfile] = state.get("company_profiles") or []
    log_step(log, "compare.start", profiles=len(profiles))

    if not profiles:
        return {
            "comparison": ComparisonTable(
                dimensions=[],
                rows={},
                gaps_competitor_has_target_lacks=[],
                gaps_target_has_competitor_lacks=[],
            )
        }

    dims = sorted(
        {d.dimension for p in profiles for d in p.dimensions},
        key=lambda d: d.value,
    )
    rows: dict[str, dict[str, list[str]]] = {}
    for profile in profiles:
        dim_map = {d.dimension: d for d in profile.dimensions}
        rows[profile.competitor] = {
            dim.value: (dim_map[dim].claim_ids if dim in dim_map else [])
            for dim in dims
        }

    target = next((p for p in profiles if p.is_target), profiles[0])
    competitors = [p for p in profiles if not p.is_target]

    store = get_evidence_store(Path(state["run_dir"]) / "evidence.jsonl")
    gaps_comp: list[GapItem] = []
    gaps_target: list[GapItem] = []

    if settings.openai_api_key and any(p.claim_ids for p in profiles):
        try:
            llm = make_reason_llm(settings)
            result = await analyze_gaps_with_llm(
                llm,
                target_company=target.competitor,
                domain_description=state.get("description") or "",
                target_claims=_claim_payload(store.claims_for_competitor(target.competitor)),
                competitor_claims={
                    c.competitor: _claim_payload(store.claims_for_competitor(c.competitor))
                    for c in competitors
                },
            )
            valid_ids = set(store.claim_lookup())
            gaps_comp = _gaps_from_drafts(result.competitor_has_target_lacks, valid_ids)
            gaps_target = _gaps_from_drafts(result.target_has_competitor_lacks, valid_ids)
            log_step(log, "compare.llm_done", gaps_out=len(gaps_comp), gaps_in=len(gaps_target))
        except Exception as exc:  # noqa: BLE001 - fall back, never fail the run
            log_step(log, "compare.llm_error", error=str(exc)[:160])

    if not gaps_comp and not gaps_target:
        gaps_comp, gaps_target = _coverage_gaps(target, competitors)
        log_step(log, "compare.fallback_gaps", gaps_out=len(gaps_comp), gaps_in=len(gaps_target))

    log_step(log, "compare.done", companies=len(rows), gaps_out=len(gaps_comp), gaps_in=len(gaps_target))
    return {
        "comparison": ComparisonTable(
            dimensions=dims or list(Dimension),
            rows=rows,
            gaps_competitor_has_target_lacks=gaps_comp,
            gaps_target_has_competitor_lacks=gaps_target,
        )
    }
