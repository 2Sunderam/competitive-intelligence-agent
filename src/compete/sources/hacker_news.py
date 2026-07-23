from __future__ import annotations

import html
import re

import httpx

from compete.config import Settings
from compete.models import FetchedDocument, SourceKind
from compete.query import build_hn_queries, is_relevant, relevance_score
from compete.sources.base import (
    BaseSourceClient,
    FetchContext,
    SoftSourceError,
    skip_record,
)


def _strip_html(text: str) -> str:
    """Algolia returns comment bodies as HTML fragments."""
    if not text:
        return ""
    text = re.sub(r"<p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


class HackerNewsClient(BaseSourceClient):
    """Hacker News via the public Algolia search API (no auth, no API key).

    Searches both ``tags=story`` and ``tags=comment``. Comments matter more:
    a story is usually a launch announcement, while the comment threads carry
    the actual comparisons and complaints.

    Algolia's matching is loose - a bare product name can report hundreds of
    thousands of hits, most of which never mention it. Every hit is therefore
    relevance-scored against the competitor name and category keywords before
    it becomes a document.
    """

    source = SourceKind.HACKER_NEWS

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    async def _search(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        query: str,
        tags: str,
    ) -> list[dict]:
        resp = await client.get(
            api_url,
            params={
                "query": query,
                "tags": tags,
                "hitsPerPage": self.settings.hn_results_per_query,
            },
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 429:
            raise SoftSourceError(
                skip_record(
                    competitor="",
                    source=SourceKind.HACKER_NEWS,
                    reason="HN Algolia rate-limited (HTTP 429)",
                    url=api_url,
                )
            )
        if resp.status_code >= 400:
            raise SoftSourceError(
                skip_record(
                    competitor="",
                    source=SourceKind.HACKER_NEWS,
                    reason=f"HN API HTTP {resp.status_code}",
                    url=api_url,
                )
            )
        payload = resp.json()
        hits = payload.get("hits") or []
        return [h for h in hits if isinstance(h, dict)]

    @staticmethod
    def _hit_text(hit: dict) -> tuple[str, str]:
        """Return ``(title, body)`` for a story or comment hit."""
        title = hit.get("title") or hit.get("story_title") or ""
        body = _strip_html(
            hit.get("comment_text") or hit.get("story_text") or ""
        )
        return title, body

    @staticmethod
    def _hit_url(hit: dict) -> str:
        object_id = hit.get("objectID")
        if object_id:
            return f"https://news.ycombinator.com/item?id={object_id}"
        return hit.get("url") or ""

    async def fetch(self, ctx: FetchContext) -> list[FetchedDocument]:
        api_url = self.settings.hacker_news_api_url or "https://hn.algolia.com/api/v1/search"
        pairs = build_hn_queries(
            ctx.competitor,
            keywords=ctx.keywords,
            peers=ctx.peers,
            description=ctx.description,
        )
        client = self._client or httpx.AsyncClient(timeout=self.settings.http_timeout_seconds)
        owns = self._client is None

        try:
            hits: dict[str, dict] = {}
            errors: list[str] = []
            for query, tags in pairs:
                try:
                    for hit in await self._search(client, api_url, query, tags):
                        object_id = str(hit.get("objectID") or "")
                        if object_id and object_id not in hits:
                            hits[object_id] = hit
                except SoftSourceError as exc:
                    errors.append(exc.skip.reason)
                    if "429" in exc.skip.reason:
                        break

            if not hits:
                reason = (
                    "; ".join(dict.fromkeys(errors))
                    if errors
                    else f"HN search returned no results across {len(pairs)} queries"
                )
                raise SoftSourceError(
                    skip_record(
                        competitor=ctx.competitor,
                        source=SourceKind.HACKER_NEWS,
                        reason=reason,
                        url=api_url,
                    )
                )

            scored: list[tuple[int, dict, str, str]] = []
            for hit in hits.values():
                title, body = self._hit_text(hit)
                score = relevance_score(
                    text=body,
                    title=title,
                    competitor=ctx.competitor,
                    aliases=ctx.aliases,
                    keywords=ctx.keywords,
                    peers=ctx.peers,
                )
                if is_relevant(score):
                    scored.append((score, hit, title, body))
            scored.sort(key=lambda row: row[0], reverse=True)
            keep = scored[: self.settings.max_docs_per_competitor]

            if not keep:
                raise SoftSourceError(
                    skip_record(
                        competitor=ctx.competitor,
                        source=SourceKind.HACKER_NEWS,
                        reason=(
                            f"HN returned {len(hits)} hits but none were on-topic "
                            f"for {ctx.competitor} (relevance filter)"
                        ),
                        url=api_url,
                    )
                )

            docs: list[FetchedDocument] = []
            for _score, hit, title, body in keep:
                text = "\n\n".join(p for p in (title, body) if p).strip()
                if not text:
                    continue
                created = hit.get("created_at") or ""
                docs.append(
                    FetchedDocument(
                        competitor=ctx.competitor,
                        source=SourceKind.HACKER_NEWS,
                        url=self._hit_url(hit) or f"hn://{ctx.competitor}",
                        platform="hacker_news",
                        title=title,
                        text=text[: self.settings.max_document_chars],
                        date=created[:10] or None,
                    )
                )

            if not docs:
                raise SoftSourceError(
                    skip_record(
                        competitor=ctx.competitor,
                        source=SourceKind.HACKER_NEWS,
                        reason="HN results had no usable text",
                        url=api_url,
                    )
                )
            return docs
        except SoftSourceError as exc:
            if not exc.skip.competitor:
                exc.skip.competitor = ctx.competitor
            raise
        except Exception as exc:  # noqa: BLE001
            raise SoftSourceError(
                skip_record(
                    competitor=ctx.competitor,
                    source=SourceKind.HACKER_NEWS,
                    reason=f"HN API error: {exc}",
                    url=api_url,
                )
            ) from exc
        finally:
            if owns:
                await client.aclose()
