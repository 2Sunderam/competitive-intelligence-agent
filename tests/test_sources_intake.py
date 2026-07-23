from __future__ import annotations

import httpx
import pytest

from compete.config import Settings
from compete.models import SourceKind
from compete.nodes.intake import intake_node
from compete.sources.base import FetchContext, SoftSourceError
from compete.sources.hacker_news import HackerNewsClient
from compete.sources.reddit import RedditClient


def ctx(competitor: str = "Acme", **kwargs) -> FetchContext:
    kwargs.setdefault("description", "B2B CRM for sales teams")
    kwargs.setdefault("keywords", ["crm", "sales"])
    return FetchContext(competitor=competitor, **kwargs)


def test_intake_dedupes_and_sets_target():
    out = intake_node(
        {
            "company_name": "Acme CRM",
            "description": "B2B CRM",
            "seed_competitors": ["HubSpot", "hubspot", "Salesforce", "n/a"],
        }
    )
    names = [c.name for c in out["competitors"]]
    assert names[0] == "Acme CRM"
    assert out["competitors"][0].is_target
    assert "HubSpot" in names
    assert "Salesforce" in names
    assert names.count("HubSpot") == 1


@pytest.mark.asyncio
async def test_reddit_missing_token_fails_soft():
    client = RedditClient(Settings(reddit_access_token=""))
    with pytest.raises(SoftSourceError) as exc:
        await client.fetch(ctx())
    assert exc.value.skip.source == SourceKind.REDDIT
    assert "REDDIT_ACCESS_TOKEN" in exc.value.skip.reason


@pytest.mark.asyncio
async def test_reddit_bearer_search_flow():
    seen_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "/comments/" in request.url.path:
            return httpx.Response(
                200,
                json=[
                    {},
                    {
                        "data": {
                            "children": [
                                {"kind": "t1", "data": {"author": "dev", "body": "Support is slow too"}},
                                {"kind": "t1", "data": {"author": "AutoModerator", "body": "read the rules"}},
                            ]
                        }
                    },
                ],
            )
        assert request.url.path.endswith("/search")
        assert request.headers.get("Authorization") == "Bearer tok_from_browser"
        seen_queries.append(request.url.params.get("q", ""))
        return httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "id": "abc",
                                "title": "Acme pricing discussion",
                                "selftext": "Acme crm pricing jumps after 10 seats and support is slow",
                                "subreddit": "saas",
                                "num_comments": 4,
                                "permalink": "/r/saas/comments/abc/acme/",
                                "created_utc": 1710000000,
                            },
                        }
                    ]
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        settings = Settings(reddit_access_token="tok_from_browser")
        client = RedditClient(settings, client=http)
        docs = await client.fetch(ctx(peers=["HubSpot"]))

    assert len(docs) == 1
    assert docs[0].source == SourceKind.REDDIT
    assert "pricing jumps" in docs[0].text
    # Query decomposition: several short queries, never one long description blob
    assert len(seen_queries) > 1
    assert all(len(q) < 80 for q in seen_queries)
    assert any("vs HubSpot" in q for q in seen_queries)
    # Comments are pulled into the document; AutoModerator noise is dropped
    assert "Support is slow too" in docs[0].text
    assert "read the rules" not in docs[0].text
    assert docs[0].date == "2024-03-09"


@pytest.mark.asyncio
async def test_reddit_expired_token_fails_soft():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = RedditClient(Settings(reddit_access_token="expired"), client=http)
        with pytest.raises(SoftSourceError) as exc:
            await client.fetch(ctx())
    assert "401" in exc.value.skip.reason
    assert exc.value.skip.competitor == "Acme"


@pytest.mark.asyncio
async def test_reddit_offtopic_results_are_filtered():
    """Reddit search is fuzzy — results that never mention the company are dropped."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "id": "zzz",
                                "title": "AITA for telling my roommate to move out",
                                "selftext": "Nothing to do with software at all.",
                                "subreddit": "AmItheAsshole",
                                "num_comments": 400,
                                "permalink": "/r/AmItheAsshole/comments/zzz/x/",
                            },
                        }
                    ]
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = RedditClient(Settings(reddit_access_token="tok"), client=http)
        with pytest.raises(SoftSourceError) as exc:
            await client.fetch(ctx())
    assert "relevance filter" in exc.value.skip.reason


@pytest.mark.asyncio
async def test_hn_parses_algolia_hits():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "objectID": "1",
                        "title": "Acme launch",
                        "url": "https://example.com/acme",
                        "story_text": "Acme pricing is high for startups using this crm",
                        "created_at": "2026-01-01T10:00:00Z",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = HackerNewsClient(Settings(), client=http)
        docs = await client.fetch(ctx())
    assert len(docs) == 1
    assert docs[0].source == SourceKind.HACKER_NEWS
    assert "pricing is high" in docs[0].text
    assert docs[0].date == "2026-01-01"


@pytest.mark.asyncio
async def test_hn_strips_html_from_comments():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "objectID": "2",
                        "story_title": "Ask HN: CRM tools",
                        "comment_text": "<p>Acme crm is great but the pricing &quot;tiers&quot; are painful</p>",
                        "created_at": "2026-02-02T10:00:00Z",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = HackerNewsClient(Settings(), client=http)
        docs = await client.fetch(ctx())
    assert '"tiers"' in docs[0].text
    assert "<p>" not in docs[0].text
