from __future__ import annotations

from pathlib import Path

from compete.models import (
    Claim,
    Dimension,
    Sentiment,
    SourceKind,
)
from compete.nodes.cluster import _cluster_deterministic
from compete.nodes.compare import compare_node
from compete.nodes.report import write_report
from compete.nodes.synthesize import _synthesize_deterministic
from compete.store.evidence import EvidenceStore, get_evidence_store, reset_evidence_stores


def _seed(store: EvidenceStore) -> None:
    store.append(
        competitor="Acme CRM",
        source=SourceKind.REDDIT.value,
        url="https://reddit.com/1",
        platform="reddit",
        date=None,
        source_text="Support is amazing and replies within an hour.",
        draft_claims=[
            Claim(
                claim_id="tmp",
                dimension=Dimension.SUPPORT,
                text="support replies within an hour",
                quote="replies within an hour",
                sentiment=Sentiment.POSITIVE,
            )
        ],
    )
    store.append(
        competitor="Acme CRM",
        source=SourceKind.REDDIT.value,
        url="https://reddit.com/2",
        platform="reddit",
        date=None,
        source_text="Support is terrible and tickets sit for days.",
        draft_claims=[
            Claim(
                claim_id="tmp",
                dimension=Dimension.SUPPORT,
                text="tickets sit for days",
                quote="tickets sit for days",
                sentiment=Sentiment.NEGATIVE,
            )
        ],
    )
    store.append(
        competitor="HubSpot",
        source=SourceKind.HACKER_NEWS.value,
        url="https://hn.com/1",
        platform="hacker_news",
        date=None,
        source_text="HubSpot pricing jumps after 10 seats.",
        draft_claims=[
            Claim(
                claim_id="tmp",
                dimension=Dimension.PRICING,
                text="pricing jumps after 10 seats",
                quote="pricing jumps after 10 seats",
                sentiment=Sentiment.NEGATIVE,
            )
        ],
    )


def test_synthesize_keeps_conflicting_sentiments(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    _seed(store)
    profile = _synthesize_deterministic("Acme CRM", True, store)
    support = next(d for d in profile.dimensions if d.dimension == Dimension.SUPPORT)
    assert len(support.claim_ids) == 2
    assert Sentiment.POSITIVE in support.conflicting_sentiments
    assert Sentiment.NEGATIVE in support.conflicting_sentiments


async def test_compare_and_cluster_and_report(tmp_path: Path):
    store = get_evidence_store(tmp_path / "evidence.jsonl")
    _seed(store)
    acme = _synthesize_deterministic("Acme CRM", True, store)
    hub = _synthesize_deterministic("HubSpot", False, store)
    state = {
        "company_name": "Acme CRM",
        "run_id": "test",
        "run_dir": str(tmp_path),
        "company_profiles": [acme, hub],
        "skips": [],
    }
    comparison = (await compare_node(state))["comparison"]
    assert "Acme CRM" in comparison.rows
    assert "HubSpot" in comparison.rows

    clusters = _cluster_deterministic(store)
    assert any(c.scope == "company_specific" for c in clusters)

    state["comparison"] = comparison
    state["pain_clusters"] = clusters
    out = write_report(state)
    assert Path(out["brief_path"]).exists()
    assert Path(out["evidence_json_path"]).exists()
    brief = Path(out["brief_path"]).read_text(encoding="utf-8")
    assert "Competitive Brief" in brief
    assert "conflicting" in brief
