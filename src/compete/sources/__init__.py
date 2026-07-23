from __future__ import annotations

from compete.config import Settings
from compete.sources.base import BaseSourceClient
from compete.sources.hacker_news import HackerNewsClient
from compete.sources.jina import JinaClient
from compete.sources.reddit import RedditClient


def build_source_clients(settings: Settings) -> list[BaseSourceClient]:
    return [
        RedditClient(settings),
        HackerNewsClient(settings),
        JinaClient(settings),
    ]
