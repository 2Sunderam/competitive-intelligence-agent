from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from compete.logging_utils import get_logger, log_step
from compete.models import CompanyProfile, Dimension, DimensionSummary, Sentiment
from compete.state import SynthTask
from compete.store.evidence import EvidenceStore, get_evidence_store

log = get_logger("synthesize")


def _synthesize_deterministic(
    competitor_name: str, is_target: bool, store: EvidenceStore
) -> CompanyProfile:
    """Group one company's claims by dimension.

    Purely deterministic on purpose: the claims are already grounded, so an
    LLM pass here would add cost and hallucination risk without adding
    information. Conflicting sentiments inside a dimension are recorded rather
    than resolved.
    """
    by_dim: dict[Dimension, list[str]] = defaultdict(list)
    sentiments: dict[Dimension, set[Sentiment]] = defaultdict(set)
    notes: dict[Dimension, list[str]] = defaultdict(list)
    all_ids: list[str] = []

    for _record, claim in store.claims_for_competitor(competitor_name):
        by_dim[claim.dimension].append(claim.claim_id)
        sentiments[claim.dimension].add(claim.sentiment)
        notes[claim.dimension].append(claim.text)
        all_ids.append(claim.claim_id)

    dimensions: list[DimensionSummary] = []
    for dim, claim_ids in by_dim.items():
        unique_sentiments = sorted(sentiments[dim], key=lambda s: s.value)
        dimensions.append(
            DimensionSummary(
                dimension=dim,
                claim_ids=claim_ids,
                notes=list(dict.fromkeys(notes[dim])),  # preserve order, dedupe
                conflicting_sentiments=unique_sentiments if len(unique_sentiments) > 1 else [],
            )
        )

    return CompanyProfile(
        competitor=competitor_name,
        is_target=is_target,
        dimensions=dimensions,
        claim_ids=all_ids,
    )


async def synthesize_company(task: SynthTask) -> dict:
    """Reduce L1: one company profile from that company's grounded claims."""
    competitor = task["competitor"]
    store = get_evidence_store(Path(task["run_dir"]) / "evidence.jsonl")
    profile = _synthesize_deterministic(competitor.name, competitor.is_target, store)
    log_step(
        log,
        "synthesize.done",
        competitor=competitor.name,
        claims=len(profile.claim_ids),
        dimensions=len(profile.dimensions),
        conflicts=sum(1 for d in profile.dimensions if d.conflicting_sentiments),
    )
    return {"company_profiles": [profile]}
