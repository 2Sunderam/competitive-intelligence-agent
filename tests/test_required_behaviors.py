"""The three behaviors the assignment requires.

    uv run pytest

No network and no HTTP mocking: two stub source clients cover the failure case,
and the other two tests run directly against the evidence store.
"""

from __future__ import annotations

from pathlib import Path

from compete.models import (
    Claim,
    Competitor,
    Dimension,
    FetchedDocument,
    Sentiment,
    SourceKind,
)
from compete.nodes.ingest import ingest_competitor
from compete.sources.base import BaseSourceClient, SoftSourceError, skip_record
from compete.store.evidence import EvidenceStore


def _claim(*, text: str, quote: str, dimension: Dimension, sentiment: Sentiment) -> Claim:
    """A claim as the extractor produces it, before the store assigns an id."""
    return Claim(
        claim_id="tmp",
        dimension=dimension,
        text=text,
        quote=quote,
        sentiment=sentiment,
    )


# --------------------------------------------------------------------------
# 1. Unreachable, blocked, or empty source handling
# --------------------------------------------------------------------------


class _BlockedSource(BaseSourceClient):
    """Stands in for a source that is rate-limited, blocked, or returns nothing."""

    source = SourceKind.REDDIT

    async def fetch(self, ctx):
        raise SoftSourceError(
            skip_record(
                competitor=ctx.competitor,
                source=SourceKind.REDDIT,
                reason="Reddit search HTTP 403: blocked",
            )
        )


class _WorkingSource(BaseSourceClient):
    source = SourceKind.HACKER_NEWS

    async def fetch(self, ctx):
        return [
            FetchedDocument(
                competitor=ctx.competitor,
                source=SourceKind.HACKER_NEWS,
                url="https://news.ycombinator.com/item?id=1",
                platform="hacker_news",
                title="Acme thread",
                text="Acme pricing is steep for small teams.",
            )
        ]


async def test_blocked_source_is_skipped_and_the_run_continues(monkeypatch, tmp_path: Path):
    """A failing source is logged as a skip; it neither crashes the run nor is
    replaced by fabricated content, and the healthy source still returns docs."""
    monkeypatch.setattr(
        "compete.nodes.ingest.build_source_clients",
        lambda settings: [_BlockedSource(), _WorkingSource()],
    )

    result = await ingest_competitor(
        {
            "competitor": Competitor(name="Acme CRM", description="B2B CRM"),
            "run_dir": str(tmp_path),
            "description": "B2B CRM for sales teams",
            "peers": [],
        }
    )

    assert len(result["skips"]) == 1
    assert result["skips"][0].source == SourceKind.REDDIT
    assert "403" in result["skips"][0].reason  # the real reason is preserved
    assert len(result["documents"]) == 1  # the working source is unaffected


# --------------------------------------------------------------------------
# 2. Deduplication of the same claim appearing across multiple sources
# --------------------------------------------------------------------------


async def test_same_claim_from_two_sources_is_linked_not_duplicated(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    source_text = "Pricing jumps sharply above 10 seats for growing teams."
    draft = [
        _claim(
            text="pricing jumps sharply above 10 seats",
            quote="Pricing jumps sharply above 10 seats",
            dimension=Dimension.PRICING,
            sentiment=Sentiment.NEGATIVE,
        )
    ]

    first = store.append(
        competitor="Acme CRM",
        source=SourceKind.REDDIT.value,
        url="https://reddit.com/r/saas/1",
        platform="reddit",
        date=None,
        source_text=source_text,
        draft_claims=draft,
    )
    second = store.append(
        competitor="Acme CRM",
        source=SourceKind.HACKER_NEWS.value,
        url="https://news.ycombinator.com/item?id=1",
        platform="hacker_news",
        date=None,
        source_text=source_text,
        draft_claims=draft,
    )

    assert first is not None
    assert second is None  # the duplicate is not written again

    records = store.read_all()
    assert len(records) == 1
    assert len(records[0].claims) == 1
    # The second source is attached to the existing claim instead
    assert (
        "https://news.ycombinator.com/item?id=1"
        in records[0].claims[0].linked_source_urls
    )


# --------------------------------------------------------------------------
# 3. Preservation of conflicting viewpoints about the same company
# --------------------------------------------------------------------------


async def test_conflicting_viewpoints_are_both_preserved(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.jsonl")

    store.append(
        competitor="Acme CRM",
        source=SourceKind.REDDIT.value,
        url="https://reddit.com/1",
        platform="reddit",
        date=None,
        source_text="Support is amazing and replies within an hour.",
        draft_claims=[
            _claim(
                text="support replies within an hour",
                quote="replies within an hour",
                dimension=Dimension.SUPPORT,
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
            _claim(
                text="tickets sit for days",
                quote="tickets sit for days",
                dimension=Dimension.SUPPORT,
                sentiment=Sentiment.NEGATIVE,
            )
        ],
    )

    claims = [claim for _record, claim in store.claims_for_competitor("Acme CRM")]

    # Both survive on the same dimension — neither is resolved away into a verdict
    assert len(claims) == 2
    assert {c.sentiment for c in claims} == {Sentiment.POSITIVE, Sentiment.NEGATIVE}
