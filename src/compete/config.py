from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_extract_model: str = "gpt-5.4-nano"
    openai_reason_model: str = "gpt-5.4-mini"

    # Jina AI — search + reader (r.jina.ai / s.jina.ai)
    jina_api_key: str = ""
    jina_search_url: str = "https://s.jina.ai/"
    jina_reader_base: str = "https://r.jina.ai/"
    jina_g2_reader_limit: int = 3
    jina_linkedin_result_limit: int = 3

    # Reddit — browser session access token (Application storage / cookie: token_v2)
    # Unauthenticated reddit.com/*.json returns HTTP 403 for every endpoint
    # (search, subreddit listing, old.reddit) regardless of User-Agent, so a
    # bearer token is the only working path. See README for the trade-off.
    reddit_access_token: str = ""
    reddit_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    )
    reddit_results_per_query: int = 6
    reddit_comment_limit: int = 15
    reddit_threads_with_comments: int = 3
    reddit_time_filter: str = "all"

    # Hacker News — public Algolia search (no API key)
    hacker_news_api_url: str = "https://hn.algolia.com/api/v1/search"
    hn_results_per_query: int = 6

    http_timeout_seconds: float = 30.0
    max_docs_per_source: int = 5
    # Cap on documents kept per competitor per source after relevance ranking.
    max_docs_per_competitor: int = 6
    max_document_chars: int = 14000
    # Source text kept alongside each evidence record. Quotes are validated at
    # write time against the full document; storing a wide excerpt lets anyone
    # re-verify them from the artifact alone.
    evidence_excerpt_chars: int = 14000
    data_dir: Path = Field(default=Path("data/runs"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
