from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import httpx

from compete.config import Settings
from compete.models import FetchedDocument, SourceKind
from compete.query import build_review_search_queries, is_relevant, relevance_score
from compete.sources.base import (
    BaseSourceClient,
    FetchContext,
    SoftSourceError,
    skip_record,
)


def company_slug(name: str) -> str:
    """Guess a G2-style slug: lowercase hyphenated company name."""
    slug = name.casefold()
    slug = slug.replace(".ai", "-ai").replace(".com", "")
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def g2_search_query(company: str, keywords: list[str] | None = None) -> str:
    base = f'site:g2.com/products "{company}" reviews'
    if keywords:
        # Disambiguate ambiguous names (e.g. Fathom meeting AI vs nonprofit)
        kw = " OR ".join(f'"{k}"' for k in keywords[:3])
        return f"{base} ({kw})"
    return base


def linkedin_search_query(company: str, keywords: list[str] | None = None) -> str:
    base = f'site:linkedin.com/company "{company}"'
    if keywords:
        kw = " OR ".join(f'"{k}"' for k in keywords[:3])
        return f"{base} ({kw})"
    return base


def is_g2_product_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        path = urlparse(url).path.lower()
    except Exception:  # noqa: BLE001
        return False
    if "g2.com" not in host:
        return False
    return "/products/" in path


def is_linkedin_company_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        path = urlparse(url).path.lower()
    except Exception:  # noqa: BLE001
        return False
    if "linkedin.com" not in host:
        return False
    return "/company/" in path


def linkedin_slug(url: str) -> str | None:
    """Extract the ``/company/<slug>`` segment, ignoring regional subdomains."""
    if not is_linkedin_company_url(url):
        return None
    parts = [p for p in urlparse(url).path.split("/") if p]
    try:
        idx = parts.index("company")
        return parts[idx + 1].casefold()
    except (ValueError, IndexError):
        return None


def score_g2_url(url: str, company: str) -> int:
    """Higher is better — prefer product/reviews pages matching the company slug."""
    if not is_g2_product_url(url):
        return -100
    path = urlparse(url).path.lower()
    slug = company_slug(company)
    score = 10
    if slug and slug in path:
        score += 50
    if path.rstrip("/").endswith("/reviews") or "/reviews" in path:
        score += 30
    if "/compare/" in path:
        score -= 5
    if "/discussions/" in path or "/answers/" in path:
        score -= 10
    return score


def canonical_g2_product_url(url: str) -> str | None:
    """Normalize to https://www.g2.com/products/<slug> or .../reviews when possible."""
    if not is_g2_product_url(url):
        return None
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    # Expect: products / <slug> / optional(reviews|...)
    try:
        idx = parts.index("products")
        slug = parts[idx + 1]
    except (ValueError, IndexError):
        return None
    if len(parts) > idx + 2 and parts[idx + 2] == "reviews":
        return f"https://www.g2.com/products/{slug}/reviews"
    return f"https://www.g2.com/products/{slug}"


def pick_best_g2_urls(docs: list[FetchedDocument], company: str, *, limit: int = 3) -> list[str]:
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for doc in docs:
        raw = (doc.url or "").strip()
        url = canonical_g2_product_url(raw) or raw
        if not url or url in seen:
            continue
        seen.add(url)
        score = score_g2_url(url, company)
        if score < 0:
            continue
        ranked.append((score, url))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [url for _, url in ranked[:limit]]


def linkedin_url_matches_company(
    url: str,
    company: str,
    *,
    linkedin_url: str | None = None,
    aliases: list[str] | None = None,
) -> bool:
    """Accept a LinkedIn company URL only when its slug IS the company's slug.

    A substring test is not enough: searching for "Notion" surfaces
    ``/company/notion-setup`` and ``/company/revive-notion``, which are other
    businesses entirely. Exact slug matching against the resolved URL plus
    name/alias slugs keeps those out, while regional subdomains
    (``iq.linkedin.com/company/linearapp``) still dedupe onto one entity.
    """
    slug = linkedin_slug(url)
    if not slug:
        return False
    accepted = {company_slug(company)}
    for alias in aliases or []:
        if alias and alias.strip():
            accepted.add(company_slug(alias))
    resolved = linkedin_slug(linkedin_url or "")
    if resolved:
        accepted.add(resolved)
    # "linearapp" for "Linear", "shortcutsoftware" for "Shortcut"
    compact = {a.replace("-", "") for a in accepted}
    return slug in accepted or slug.replace("-", "") in compact


def linkedin_matches_company(
    url: str,
    title: str,
    text: str,
    company: str,
    *,
    linkedin_url: str | None = None,
    keywords: list[str] | None = None,
) -> bool:
    """Compatibility wrapper: slug match first, then name+keyword evidence."""
    if linkedin_url_matches_company(url, company, linkedin_url=linkedin_url):
        return True
    blob = f"{title}\n{text}".casefold()
    if company.casefold() not in blob:
        return False
    return bool(keywords) and any(k.casefold() in blob for k in keywords)


BLOCKED_MARKERS = (
    "please wait for verification",
    "just a moment",
    "enable javascript",
    "cf-browser-verification",
    "access denied",
    "requiring captcha",
    "checking your browser",
    "attention required",
)


def looks_blocked(text: str) -> str | None:
    """Return the bot-wall marker found in ``text``, or None if it looks clean."""
    low = (text or "").casefold()
    for marker in BLOCKED_MARKERS:
        if marker in low:
            return marker
    return None


def reader_looks_usable(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 800:
        return False
    return looks_blocked(t) is None


def parse_reader_payload(raw: str) -> tuple[str, str, str | None]:
    """Unwrap a Jina reader response into ``(title, content, block_reason)``.

    With ``Accept: application/json`` the reader replies with an envelope like
    ``{"code":200,"data":{"title":...,"content":...,"warning":...}}``. Passing
    that envelope straight to the extractor makes the model quote JSON-escaped
    text, so it is unwrapped here. Plain-markdown replies pass through.
    """
    text = (raw or "").strip()
    if not text:
        return "", "", "empty response"

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return "", text, looks_blocked(text)
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            content = (data.get("content") or data.get("text") or "").strip()
            title = (data.get("title") or "").strip()
            warning = (data.get("warning") or "").strip()
            if not content:
                reason = looks_blocked(warning) or warning or "reader returned empty content"
                return title, "", reason
            return title, content, looks_blocked(content) or looks_blocked(warning)
        return "", text, looks_blocked(text)

    return "", text, looks_blocked(text)


class JinaClient(BaseSourceClient):
    """Jina AI search (``s.jina.ai``) + reader (``r.jina.ai``) as the web layer.

    Ladder per competitor, stopping once enough usable documents are found:

    1. **LinkedIn** — read the resolved ``/company/<slug>`` page for positioning.
    2. **G2** — site-operator search, then reader on the best product/review URLs.
    3. **Review fallback** — when G2 answers with a CAPTCHA wall (it usually
       does), search for review / pros-and-cons pages on any other reachable
       host and read those instead.

    Every bot-wall is captured as a reason string and surfaced in the skip log,
    so a blocked source is visible in the output rather than silently missing.
    """

    source = SourceKind.WEB_SEARCH

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.jina_api_key:
            headers["Authorization"] = f"Bearer {self.settings.jina_api_key}"
        return headers

    def _parse_search_payload(
        self,
        *,
        data: object | None,
        raw_text: str,
        query: str,
        limit: int,
        platform: str,
        source: SourceKind,
    ) -> list[FetchedDocument]:
        docs: list[FetchedDocument] = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("results") or data.get("items") or []
            for item in items[:limit]:
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or item.get("link") or ""
                title = item.get("title") or ""
                text = item.get("content") or item.get("description") or item.get("snippet") or title
                if not url and not text:
                    continue
                docs.append(
                    FetchedDocument(
                        competitor="",
                        source=source,
                        url=url or f"jina-search:{query}",
                        platform=platform,
                        title=title,
                        text=text,
                    )
                )
        elif raw_text.strip():
            docs.append(
                FetchedDocument(
                    competitor="",
                    source=source,
                    url=f"{self.settings.jina_search_url}?q={query}",
                    platform=platform,
                    title=query,
                    text=raw_text[: self.settings.max_document_chars],
                )
            )
        return docs

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        platform: str = "web",
        source: SourceKind = SourceKind.WEB_SEARCH,
    ) -> list[FetchedDocument]:
        limit = limit or self.settings.max_docs_per_source
        client = self._client or httpx.AsyncClient(timeout=self.settings.http_timeout_seconds)
        owns = self._client is None
        try:
            resp = await client.get(
                self.settings.jina_search_url,
                params={"q": query},
                headers=self._headers(),
            )
            if resp.status_code >= 400:
                raise SoftSourceError(
                    skip_record(
                        competitor="",
                        source=SourceKind.WEB_SEARCH,
                        reason=f"Jina search HTTP {resp.status_code}: {resp.text[:160]}",
                    )
                )
            data = resp.json() if "application/json" in resp.headers.get("content-type", "") else None
            return self._parse_search_payload(
                data=data,
                raw_text=resp.text,
                query=query,
                limit=limit,
                platform=platform,
                source=source,
            )
        except SoftSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SoftSourceError(
                skip_record(
                    competitor="",
                    source=SourceKind.WEB_SEARCH,
                    reason=f"Jina search error: {exc}",
                )
            ) from exc
        finally:
            if owns:
                await client.aclose()

    async def read(self, url: str, *, platform: str = "web") -> FetchedDocument:
        """Fetch one page as clean text. Raises SoftSourceError when blocked."""
        client = self._client or httpx.AsyncClient(timeout=self.settings.http_timeout_seconds)
        owns = self._client is None
        reader_url = f"{self.settings.jina_reader_base.rstrip('/')}/{url}"
        try:
            resp = await client.get(reader_url, headers=self._headers())
            if resp.status_code >= 400:
                raise SoftSourceError(
                    skip_record(
                        competitor="",
                        source=SourceKind.JINA_READER,
                        reason=f"Jina reader HTTP {resp.status_code}",
                        url=url,
                    )
                )
            title, content, blocked = parse_reader_payload(resp.text)
            if blocked or not content.strip():
                raise SoftSourceError(
                    skip_record(
                        competitor="",
                        source=SourceKind.JINA_READER,
                        reason=f"blocked by bot protection ({blocked or 'empty content'})",
                        url=url,
                    )
                )
            return FetchedDocument(
                competitor="",
                source=SourceKind.JINA_READER,
                url=url,
                platform=platform,
                title=title or url,
                text=content[: self.settings.max_document_chars],
            )
        except SoftSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SoftSourceError(
                skip_record(
                    competitor="",
                    source=SourceKind.JINA_READER,
                    reason=f"Jina reader error: {exc}",
                    url=url,
                )
            ) from exc
        finally:
            if owns:
                await client.aclose()

    def _keep_if_relevant(
        self,
        doc: FetchedDocument,
        ctx: FetchContext,
        *,
        require_keywords: bool = True,
    ) -> bool:
        score = relevance_score(
            text=doc.text,
            title=doc.title,
            competitor=ctx.competitor,
            aliases=ctx.aliases,
            keywords=ctx.keywords if require_keywords else None,
            peers=ctx.peers,
            # A resolved LinkedIn page is already known to be the right company
            require_context=require_keywords,
        )
        return is_relevant(score)

    async def _linkedin_docs(self, ctx: FetchContext, notes: list[str]) -> list[FetchedDocument]:
        docs: list[FetchedDocument] = []
        seen_slugs: set[str] = set()

        candidates: list[str] = []
        if ctx.linkedin_url:
            candidates.append(ctx.linkedin_url)
        try:
            hits = await self.search(
                linkedin_search_query(ctx.competitor, ctx.keywords or None),
                limit=max(self.settings.jina_linkedin_result_limit * 2, 5),
                platform="linkedin",
            )
            candidates.extend(h.url for h in hits if h.url)
        except SoftSourceError as exc:
            notes.append(f"linkedin search: {exc.skip.reason}")

        for url in candidates:
            if len(docs) >= self.settings.jina_linkedin_result_limit:
                break
            if not linkedin_url_matches_company(
                url, ctx.competitor, linkedin_url=ctx.linkedin_url, aliases=ctx.aliases
            ):
                continue
            slug = linkedin_slug(url)
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            try:
                page = await self.read(url, platform="linkedin")
            except SoftSourceError as exc:
                notes.append(f"linkedin/{slug}: {exc.skip.reason}")
                continue
            page.competitor = ctx.competitor
            # LinkedIn "About" copy is short; do not require category keywords
            if self._keep_if_relevant(page, ctx, require_keywords=False):
                docs.append(page)
        return docs

    async def _g2_docs(self, ctx: FetchContext, notes: list[str]) -> list[FetchedDocument]:
        slug = company_slug(ctx.competitor)
        candidates: list[FetchedDocument] = [
            FetchedDocument(
                competitor=ctx.competitor,
                source=SourceKind.WEB_SEARCH,
                url=seed,
                platform="g2",
                title=seed,
                text="",
            )
            for seed in dict.fromkeys(
                [
                    ctx.g2_url or "",
                    f"https://www.g2.com/products/{slug}/reviews",
                    f"https://www.g2.com/products/{slug}",
                ]
            )
            if seed
        ]
        try:
            candidates.extend(
                await self.search(
                    g2_search_query(ctx.competitor, ctx.keywords or None),
                    limit=10,
                    platform="g2",
                )
            )
        except SoftSourceError as exc:
            notes.append(f"g2 search: {exc.skip.reason}")

        docs: list[FetchedDocument] = []
        for url in pick_best_g2_urls(
            candidates, ctx.competitor, limit=self.settings.jina_g2_reader_limit
        ):
            try:
                page = await self.read(url, platform="g2")
            except SoftSourceError as exc:
                notes.append(f"g2{urlparse(url).path}: {exc.skip.reason}")
                continue
            page.competitor = ctx.competitor
            if self._keep_if_relevant(page, ctx):
                docs.append(page)
        return docs

    async def _review_fallback_docs(self, ctx: FetchContext, notes: list[str]) -> list[FetchedDocument]:
        """Third-party review prose from any reachable host.

        This is what keeps a second public source alive when G2 is walled off.
        """
        docs: list[FetchedDocument] = []
        seen_hosts: set[str] = set()
        limit = self.settings.jina_g2_reader_limit
        for query in build_review_search_queries(
            ctx.competitor, keywords=ctx.keywords, description=ctx.description
        ):
            if len(docs) >= limit:
                break
            try:
                hits = await self.search(query, limit=8, platform="web")
            except SoftSourceError as exc:
                notes.append(f"review search: {exc.skip.reason}")
                continue

            for hit in hits:
                if len(docs) >= limit:
                    break
                url = hit.url or ""
                host = urlparse(url).netloc.casefold()
                if not host or host in seen_hosts:
                    continue
                # g2 / linkedin are handled by their own steps
                if "g2.com" in host or "linkedin.com" in host:
                    continue
                seen_hosts.add(host)
                try:
                    page = await self.read(url, platform=host)
                except SoftSourceError as exc:
                    notes.append(f"{host}: {exc.skip.reason}")
                    continue
                page.competitor = ctx.competitor
                if self._keep_if_relevant(page, ctx):
                    docs.append(page)
        return docs

    async def fetch_g2_and_linkedin(
        self,
        competitor: str,
        *,
        g2_url: str | None = None,
        linkedin_url: str | None = None,
        category_keywords: list[str] | None = None,
        aliases: list[str] | None = None,
    ) -> list[FetchedDocument]:
        """Compatibility entry point for older callers/tests."""
        return await self.fetch(
            FetchContext(
                competitor=competitor,
                keywords=list(category_keywords or []),
                aliases=list(aliases or []),
                g2_url=g2_url,
                linkedin_url=linkedin_url,
            )
        )

    async def fetch(self, ctx: FetchContext) -> list[FetchedDocument]:
        notes: list[str] = []
        docs: list[FetchedDocument] = []

        docs.extend(await self._linkedin_docs(ctx, notes))
        docs.extend(await self._g2_docs(ctx, notes))

        if not any(d.platform == "g2" for d in docs):
            docs.extend(await self._review_fallback_docs(ctx, notes))

        if not docs:
            detail = "; ".join(dict.fromkeys(notes))[:400] or "no usable results"
            raise SoftSourceError(
                skip_record(
                    competitor=ctx.competitor,
                    source=SourceKind.WEB_SEARCH,
                    reason=f"Jina web layer produced no usable documents ({detail})",
                )
            )
        return docs[: self.settings.max_docs_per_competitor]
