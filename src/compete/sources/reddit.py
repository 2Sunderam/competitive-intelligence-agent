from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from compete.config import Settings
from compete.models import FetchedDocument, SourceKind
from compete.query import build_reddit_queries, is_relevant, relevance_score
from compete.sources.base import (
    BaseSourceClient,
    FetchContext,
    SoftSourceError,
    skip_record,
)

SEARCH_URL = "https://oauth.reddit.com/search"
COMMENTS_URL = "https://oauth.reddit.com/comments/{post_id}"


class RedditClient(BaseSourceClient):
    """Reddit search over several intent-shaped queries, plus thread comments.

    Auth: unauthenticated ``reddit.com/*.json`` endpoints return HTTP 403 for
    search, subreddit listings and old.reddit alike, whatever User-Agent is
    sent. So this uses a browser session access token (``token_v2`` from
    Chrome DevTools -> Application) as ``Authorization: Bearer``. The token
    expires; when it does, every Reddit fetch degrades to a logged skip rather
    than a crash.

    Why several queries: one long query built from the company description
    returns an empty listing. Short queries like ``Linear vs Jira`` are what
    surface opinionated threads.

    Why comments: the post body is often a one-line question. The complaints,
    pricing gripes and feature gaps live in the replies.
    """

    source = SourceKind.REDDIT

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "User-Agent": self.settings.reddit_user_agent,
            "Accept": "application/json",
        }

    @staticmethod
    def _post_url(data: dict) -> str:
        permalink = data.get("permalink") or ""
        if permalink:
            return f"https://www.reddit.com{permalink}"
        url = data.get("url") or ""
        if url.startswith("/"):
            return f"https://www.reddit.com{url}"
        return url

    @staticmethod
    def _iso_date(data: dict) -> str | None:
        created = data.get("created_utc")
        if isinstance(created, (int, float)):
            return datetime.fromtimestamp(created, tz=timezone.utc).date().isoformat()
        return None

    async def _search(self, client: httpx.AsyncClient, token: str, query: str) -> list[dict]:
        resp = await client.get(
            SEARCH_URL,
            params={
                "q": query,
                "limit": self.settings.reddit_results_per_query,
                "sort": "relevance",
                "type": "link",
                "t": self.settings.reddit_time_filter,
                "raw_json": 1,
            },
            headers=self._headers(token),
        )
        if resp.status_code == 401:
            raise SoftSourceError(
                skip_record(
                    competitor="",
                    source=SourceKind.REDDIT,
                    reason="Reddit token expired or invalid (HTTP 401) - refresh REDDIT_ACCESS_TOKEN",
                    url=SEARCH_URL,
                )
            )
        if resp.status_code == 429:
            raise SoftSourceError(
                skip_record(
                    competitor="",
                    source=SourceKind.REDDIT,
                    reason="Reddit rate-limited (HTTP 429)",
                    url=SEARCH_URL,
                )
            )
        if resp.status_code >= 400:
            raise SoftSourceError(
                skip_record(
                    competitor="",
                    source=SourceKind.REDDIT,
                    reason=f"Reddit search HTTP {resp.status_code}: {resp.text[:160]}",
                    url=SEARCH_URL,
                )
            )
        children = resp.json().get("data", {}).get("children", [])
        return [c.get("data", {}) for c in children if isinstance(c, dict)]

    async def _comments(self, client: httpx.AsyncClient, token: str, post_id: str) -> list[str]:
        """Top-level comment bodies for one post. Any failure returns empty."""
        try:
            resp = await client.get(
                COMMENTS_URL.format(post_id=post_id),
                params={
                    "limit": self.settings.reddit_comment_limit,
                    "sort": "top",
                    "depth": 1,
                    "raw_json": 1,
                },
                headers=self._headers(token),
            )
            if resp.status_code >= 400:
                return []
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2:
                return []
            children = payload[1].get("data", {}).get("children", [])
        except Exception:  # noqa: BLE001 - comments are a bonus, never fatal
            return []

        bodies: list[str] = []
        for child in children:
            if not isinstance(child, dict) or child.get("kind") != "t1":
                continue
            data = child.get("data", {})
            body = (data.get("body") or "").strip()
            author = (data.get("author") or "").casefold()
            if not body or body in {"[deleted]", "[removed]"}:
                continue
            if author == "automoderator":
                continue
            bodies.append(body)
        return bodies

    async def fetch(self, ctx: FetchContext) -> list[FetchedDocument]:
        token = (self.settings.reddit_access_token or "").strip()
        if not token:
            raise SoftSourceError(
                skip_record(
                    competitor=ctx.competitor,
                    source=SourceKind.REDDIT,
                    reason=(
                        "Reddit not configured - set REDDIT_ACCESS_TOKEN. "
                        "Anonymous reddit.com JSON endpoints return HTTP 403."
                    ),
                )
            )

        queries = build_reddit_queries(
            ctx.competitor,
            keywords=ctx.keywords,
            peers=ctx.peers,
            description=ctx.description,
        )
        client = self._client or httpx.AsyncClient(timeout=self.settings.http_timeout_seconds)
        owns = self._client is None

        try:
            posts: dict[str, dict] = {}
            errors: list[str] = []
            for query in queries:
                try:
                    for data in await self._search(client, token, query):
                        post_id = data.get("id")
                        if post_id and post_id not in posts:
                            posts[post_id] = data
                except SoftSourceError as exc:
                    errors.append(exc.skip.reason)
                    # Auth / rate-limit failures apply to every query - stop early
                    if "401" in exc.skip.reason or "429" in exc.skip.reason:
                        break

            if not posts:
                reason = (
                    "; ".join(dict.fromkeys(errors))
                    if errors
                    else f"Reddit search returned no results across {len(queries)} queries"
                )
                raise SoftSourceError(
                    skip_record(
                        competitor=ctx.competitor,
                        source=SourceKind.REDDIT,
                        reason=reason,
                        url=SEARCH_URL,
                    )
                )

            # Rank by relevance before spending comment fetches or LLM calls
            scored: list[tuple[int, dict]] = []
            for data in posts.values():
                score = relevance_score(
                    text=data.get("selftext") or "",
                    title=data.get("title") or "",
                    competitor=ctx.competitor,
                    aliases=ctx.aliases,
                    keywords=ctx.keywords,
                    peers=ctx.peers,
                )
                if is_relevant(score):
                    scored.append((score, data))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            keep = scored[: self.settings.max_docs_per_competitor]

            if not keep:
                raise SoftSourceError(
                    skip_record(
                        competitor=ctx.competitor,
                        source=SourceKind.REDDIT,
                        reason=(
                            f"Reddit returned {len(posts)} posts but none were on-topic "
                            f"for {ctx.competitor} (relevance filter)"
                        ),
                        url=SEARCH_URL,
                    )
                )

            # Pull comments for the strongest threads - that is where opinions are
            comment_targets = [
                data
                for _score, data in keep[: self.settings.reddit_threads_with_comments]
                if (data.get("num_comments") or 0) > 0 and data.get("id")
            ]
            comment_map: dict[str, list[str]] = {}
            if comment_targets:
                results = await asyncio.gather(
                    *(self._comments(client, token, d["id"]) for d in comment_targets),
                    return_exceptions=True,
                )
                for data, result in zip(comment_targets, results, strict=False):
                    if isinstance(result, list):
                        comment_map[data["id"]] = result

            docs: list[FetchedDocument] = []
            for _score, data in keep:
                title = data.get("title") or ""
                body = (data.get("selftext") or "").strip()
                parts = [title, body]
                comments = comment_map.get(data.get("id", ""), [])
                if comments:
                    parts.append("--- Comments ---")
                    parts.extend(comments)
                text = "\n\n".join(p for p in parts if p).strip()
                if not text:
                    continue
                subreddit = data.get("subreddit") or ""
                docs.append(
                    FetchedDocument(
                        competitor=ctx.competitor,
                        source=SourceKind.REDDIT,
                        url=self._post_url(data) or f"reddit://{ctx.competitor}",
                        platform=f"reddit/r/{subreddit}" if subreddit else "reddit",
                        title=title,
                        text=text[: self.settings.max_document_chars],
                        date=self._iso_date(data),
                    )
                )

            if not docs:
                raise SoftSourceError(
                    skip_record(
                        competitor=ctx.competitor,
                        source=SourceKind.REDDIT,
                        reason="Reddit results had no usable text",
                        url=SEARCH_URL,
                    )
                )
            return docs
        except SoftSourceError as exc:
            # Stamp the competitor onto skips raised by the helpers
            if not exc.skip.competitor:
                exc.skip.competitor = ctx.competitor
            raise
        except Exception as exc:  # noqa: BLE001
            raise SoftSourceError(
                skip_record(
                    competitor=ctx.competitor,
                    source=SourceKind.REDDIT,
                    reason=f"Reddit API error: {exc}",
                    url=SEARCH_URL,
                )
            ) from exc
        finally:
            if owns:
                await client.aclose()
