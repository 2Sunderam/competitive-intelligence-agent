from __future__ import annotations

from compete.config import get_settings
from compete.logging_utils import get_logger, log_step
from compete.models import Competitor, FetchedDocument, SourceKind, SourceSkip
from compete.query import keywords_from_description
from compete.sources import build_source_clients
from compete.sources.base import FetchContext, SoftSourceError
from compete.state import IngestTask

log = get_logger("ingest")


def _source_name(client: object) -> str:
    source = getattr(client, "source", None)
    return source.value if isinstance(source, SourceKind) else client.__class__.__name__


async def ingest_competitor(task: IngestTask) -> dict:
    """Fetch documents from every source for one competitor. Fail soft per source.

    Each client gets a :class:`FetchContext` instead of a pre-baked query string,
    so it can decompose the request its own way: Reddit wants several short
    intent queries, Algolia wants 2-3 tokens, Jina wants site operators.
    """
    settings = get_settings()
    competitor: Competitor = task["competitor"]
    description = task.get("description") or competitor.description or ""
    peers = [p for p in (task.get("peers") or []) if p and p != competitor.name]

    keywords = list(competitor.category_keywords or []) or keywords_from_description(description)
    ctx = FetchContext(
        competitor=competitor.name,
        description=description,
        aliases=list(competitor.aliases or []),
        keywords=keywords,
        peers=peers,
        urls=list(competitor.urls or []),
        g2_url=competitor.g2_url,
        linkedin_url=competitor.linkedin_url,
    )

    log_step(
        log,
        "ingest.start",
        competitor=competitor.name,
        keywords=",".join(keywords[:4]),
        peers=len(peers),
    )

    documents: list[FetchedDocument] = []
    skips: list[SourceSkip] = []

    seen_urls: set[str] = set()

    for client in build_source_clients(settings):
        name = _source_name(client)
        try:
            docs = await client.fetch(ctx)
            # The web-search fallback can surface a Reddit thread the Reddit
            # client already returned. Extracting it twice would double the LLM
            # cost and produce duplicate claims, so dedupe on URL here.
            fresh = []
            for doc in docs:
                key = doc.url.split("?")[0].rstrip("/").casefold()
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                fresh.append(doc)
            docs = fresh
            documents.extend(docs)
            log_step(
                log,
                "ingest.source_ok",
                competitor=competitor.name,
                source=name,
                docs=len(docs),
                platforms=",".join(sorted({d.platform for d in docs})[:4]),
            )
        except SoftSourceError as exc:
            skip = exc.skip
            skips.append(
                SourceSkip(
                    competitor=skip.competitor or competitor.name,
                    source=skip.source,
                    reason=skip.reason,
                    url=skip.url,
                )
            )
            log_step(
                log,
                "ingest.source_skip",
                competitor=competitor.name,
                source=skip.source.value,
                reason=skip.reason[:160],
            )
        except Exception as exc:  # noqa: BLE001 - a broken client must not kill the run
            skips.append(
                SourceSkip(
                    competitor=competitor.name,
                    source=getattr(client, "source", SourceKind.WEB_SEARCH),
                    reason=f"unexpected error: {exc}",
                )
            )
            log_step(log, "ingest.source_error", competitor=competitor.name, error=str(exc)[:160])

    log_step(
        log,
        "ingest.done",
        competitor=competitor.name,
        docs=len(documents),
        skips=len(skips),
    )
    return {"documents": documents, "skips": skips}
