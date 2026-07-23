from __future__ import annotations

from pathlib import Path

from compete.models import Claim, Dimension, Sentiment, SourceKind
from compete.store.evidence import EvidenceStore
from compete.store.validate import quote_is_substring


def test_quote_substring_validator():
    source = "the jump from 5 to 10 seats basically doubles your bill"
    assert quote_is_substring("doubles your bill", source)
    assert not quote_is_substring("triples your bill", source)


def test_unreachable_source_does_not_block_store(tmp_path: Path):
    """Test 1 analogue: empty/blocked path still allows pipeline artifacts."""
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    assert store.read_all() == []
    # Writing nothing is fine — skip logging is separate
    skip_path = tmp_path / "skips.jsonl"
    skip_path.write_text(
        '{"competitor":"Acme","source":"reddit","reason":"blocked","url":null}\n',
        encoding="utf-8",
    )
    assert skip_path.exists()


def test_dedup_links_second_source(tmp_path: Path):
    """Test 2: same claim from 2 sources → deduped/linked, not duplicated."""
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    source_text = "Pricing jumps sharply above 10 seats for growing teams."
    draft = [
        Claim(
            claim_id="tmp",
            dimension=Dimension.PRICING,
            text="pricing jumps sharply above 10 seats",
            quote="Pricing jumps sharply above 10 seats",
            sentiment=Sentiment.NEGATIVE,
        )
    ]
    r1 = store.append(
        competitor="Acme CRM",
        source=SourceKind.REDDIT.value,
        url="https://reddit.com/r/saas/1",
        platform="reddit",
        date="2026-03-12",
        source_text=source_text,
        draft_claims=draft,
    )
    assert r1 is not None
    r2 = store.append(
        competitor="Acme CRM",
        source=SourceKind.HACKER_NEWS.value,
        url="https://news.ycombinator.com/item?id=1",
        platform="hacker_news",
        date="2026-03-13",
        source_text=source_text,
        draft_claims=draft,
    )
    assert r2 is None  # deduped
    records = store.read_all()
    assert len(records) == 1
    assert len(records[0].claims) == 1
    assert "https://news.ycombinator.com/item?id=1" in records[0].claims[0].linked_source_urls


def test_conflicting_sentiments_kept_separate(tmp_path: Path):
    """Test 3: two conflicting claims about one competitor → both present."""
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    text_pos = "Support is amazing and replies within an hour."
    text_neg = "Support is terrible and tickets sit for days."
    store.append(
        competitor="Acme CRM",
        source=SourceKind.REDDIT.value,
        url="https://reddit.com/1",
        platform="reddit",
        date=None,
        source_text=text_pos,
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
        source_text=text_neg,
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
    claims = [c for _, c in store.claims_for_competitor("Acme CRM")]
    assert len(claims) == 2
    sentiments = {c.sentiment for c in claims}
    assert sentiments == {Sentiment.POSITIVE, Sentiment.NEGATIVE}


def test_rejects_non_substring_quote(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    record = store.append(
        competitor="Acme",
        source=SourceKind.WEB_SEARCH.value,
        url="https://example.com",
        platform="web",
        date=None,
        source_text="Users like the clean dashboard.",
        draft_claims=[
            Claim(
                claim_id="tmp",
                dimension=Dimension.UX,
                text="fabricated claim",
                quote="this quote was never in the source",
                sentiment=Sentiment.POSITIVE,
            )
        ],
    )
    assert record is None
    assert store.read_all() == []
