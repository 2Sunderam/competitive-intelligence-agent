from __future__ import annotations

from dataclasses import dataclass, field

from compete.models import FetchedDocument, SourceKind, SourceSkip


class SoftSourceError(Exception):
    """Raised when a source fails but the pipeline should continue."""

    def __init__(self, skip: SourceSkip) -> None:
        self.skip = skip
        super().__init__(skip.reason)


def skip_record(
    *,
    competitor: str,
    source: SourceKind,
    reason: str,
    url: str | None = None,
) -> SourceSkip:
    return SourceSkip(competitor=competitor, source=source, reason=reason, url=url)


@dataclass
class FetchContext:
    """Everything a source client needs to build good queries and filter noise.

    ``peers`` are the other companies in the run — a document mentioning both
    the competitor and a peer is almost always a genuine comparison thread,
    which is exactly the kind of document that yields grounded claims.
    """

    competitor: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    peers: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    g2_url: str | None = None
    linkedin_url: str | None = None


class BaseSourceClient:
    source: SourceKind

    async def fetch(self, ctx: FetchContext) -> list[FetchedDocument]:
        raise NotImplementedError
