from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from compete.config import get_settings
from compete.llm.client import cluster_pain_points_with_llm, make_reason_llm
from compete.logging_utils import get_logger, log_step
from compete.models import Claim, EvidenceRecord, PainPointCluster, Sentiment
from compete.state import AgentState
from compete.store.evidence import EvidenceStore, get_evidence_store
from compete.store.validate import normalize_text

log = get_logger("cluster")

COMPLAINT_SENTIMENTS = {Sentiment.NEGATIVE, Sentiment.MIXED}


def _complaint_claims(store: EvidenceStore) -> list[tuple[EvidenceRecord, Claim]]:
    """Raw complaint claims across ALL companies.

    Deliberately reads the flat evidence log rather than the per-company
    profiles: a pain point repeated across three competitors is the domain
    signal, and per-company grouping would hide it.
    """
    return [
        (record, claim)
        for record, claim in store.all_claims()
        if claim.sentiment in COMPLAINT_SENTIMENTS
    ]


def _cluster_deterministic(store: EvidenceStore) -> list[PainPointCluster]:
    """Fallback clustering by exact normalized claim text within a dimension.

    Only groups near-identical wording, so it is a floor, not a substitute for
    the LLM pass — it exists so the pipeline still produces pain points when no
    API key is configured or the model call fails.
    """
    buckets: dict[str, list[tuple[Claim, EvidenceRecord]]] = defaultdict(list)
    for record, claim in _complaint_claims(store):
        key = f"{claim.dimension.value}|{normalize_text(claim.text)[:80]}"
        buckets[key].append((claim, record))

    clusters: list[PainPointCluster] = []
    for i, members in enumerate(buckets.values(), start=1):
        companies = sorted({record.competitor for _claim, record in members})
        clusters.append(
            PainPointCluster(
                cluster_id=f"pp_{i:03d}",
                label=members[0][0].text,
                summary="",
                scope="domain_wide" if len(companies) > 1 else "company_specific",
                companies=companies,
                claim_ids=[claim.claim_id for claim, _record in members],
                sources=sorted({record.platform for _claim, record in members}),
            )
        )
    return clusters


def _clusters_from_drafts(drafts, lookup: dict[str, tuple[EvidenceRecord, Claim]]) -> list[PainPointCluster]:
    """Turn LLM drafts into clusters, dropping any hallucinated claim_ids."""
    clusters: list[PainPointCluster] = []
    for i, draft in enumerate(drafts, start=1):
        valid_ids = [cid for cid in dict.fromkeys(draft.claim_ids) if cid in lookup]
        if not valid_ids:
            continue
        records = [lookup[cid][0] for cid in valid_ids]
        companies = sorted({r.competitor for r in records})
        clusters.append(
            PainPointCluster(
                cluster_id=f"pp_{i:03d}",
                label=draft.label.strip() or lookup[valid_ids[0]][1].text,
                summary=draft.summary.strip(),
                # Scope is decided from the evidence, not from the model
                scope="domain_wide" if len(companies) > 1 else "company_specific",
                companies=companies,
                claim_ids=valid_ids,
                sources=sorted({r.platform for r in records}),
            )
        )
    return clusters


async def cluster_pain_points(state: AgentState) -> dict:
    """Reduce L3: cluster complaints across every company into pain points."""
    settings = get_settings()
    store = get_evidence_store(Path(state["run_dir"]) / "evidence.jsonl")
    complaints = _complaint_claims(store)

    if not complaints:
        log_step(log, "cluster.no_complaints")
        return {"pain_clusters": []}

    payload = [
        {
            "claim_id": claim.claim_id,
            "competitor": record.competitor,
            "dimension": claim.dimension.value,
            "sentiment": claim.sentiment.value,
            "text": claim.text,
        }
        for record, claim in complaints
    ]

    if settings.openai_api_key:
        try:
            llm = make_reason_llm(settings)
            result = await cluster_pain_points_with_llm(
                llm,
                claims=payload,
                domain_description=state.get("description") or "",
            )
            clusters = _clusters_from_drafts(result.clusters, store.claim_lookup())
            if clusters:
                log_step(
                    log,
                    "cluster.llm_done",
                    clusters=len(clusters),
                    domain_wide=sum(1 for c in clusters if c.scope == "domain_wide"),
                    claims_in=len(payload),
                )
                return {"pain_clusters": clusters}
            log_step(log, "cluster.llm_empty", claims_in=len(payload))
        except Exception as exc:  # noqa: BLE001 - fall back, never fail the run
            log_step(log, "cluster.llm_error", error=str(exc)[:160])

    clusters = _cluster_deterministic(store)
    log_step(log, "cluster.fallback_done", clusters=len(clusters))
    return {"pain_clusters": clusters}
