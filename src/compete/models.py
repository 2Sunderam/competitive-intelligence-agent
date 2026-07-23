from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Dimension(str, Enum):
    PRICING = "pricing"
    FEATURES = "features"
    UX = "ux"
    SUPPORT = "support"
    POSITIONING = "positioning"
    OTHER = "other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class SourceKind(str, Enum):
    REDDIT = "reddit"
    HACKER_NEWS = "hacker_news"
    WEB_SEARCH = "web_search"
    JINA_READER = "jina_reader"


class Competitor(BaseModel):
    """Canonical competitor entity (target company is index 0)."""

    name: str
    description: str = ""
    domain: str | None = None
    urls: list[str] = Field(default_factory=list)
    is_target: bool = False
    # Resolved during intake (LLM + rules) for verified ingest
    g2_url: str | None = None
    linkedin_url: str | None = None
    aliases: list[str] = Field(default_factory=list)
    category_keywords: list[str] = Field(default_factory=list)


class ResolvedEntityDraft(BaseModel):
    name: str
    official_domain: str = ""
    g2_product_slug: str = ""
    linkedin_company_slug: str = ""
    aliases: list[str] = Field(default_factory=list)
    category_keywords: list[str] = Field(default_factory=list)


class EntityResolutionResult(BaseModel):
    entities: list[ResolvedEntityDraft] = Field(default_factory=list)


class IntakeInput(BaseModel):
    company_name: str
    description: str
    seed_competitors: list[str] = Field(min_length=1, max_length=8)


class Claim(BaseModel):
    claim_id: str
    dimension: Dimension
    text: str
    quote: str
    sentiment: Sentiment
    linked_source_urls: list[str] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    evidence_id: str
    competitor: str
    source: SourceKind
    url: str
    platform: str
    date: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    raw_excerpt: str | None = None


class SourceSkip(BaseModel):
    competitor: str
    source: SourceKind
    reason: str
    url: str | None = None


class FetchedDocument(BaseModel):
    competitor: str
    source: SourceKind
    url: str
    platform: str
    title: str = ""
    text: str
    date: str | None = None


class DimensionSummary(BaseModel):
    dimension: Dimension
    claim_ids: list[str]
    notes: list[str] = Field(default_factory=list)
    conflicting_sentiments: list[Sentiment] = Field(default_factory=list)


class CompanyProfile(BaseModel):
    competitor: str
    is_target: bool = False
    dimensions: list[DimensionSummary] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class GapItem(BaseModel):
    dimension: Dimension
    description: str
    claim_ids: list[str] = Field(default_factory=list)


class ComparisonTable(BaseModel):
    dimensions: list[Dimension]
    rows: dict[str, dict[str, list[str]]]
    gaps_competitor_has_target_lacks: list[GapItem] = Field(default_factory=list)
    gaps_target_has_competitor_lacks: list[GapItem] = Field(default_factory=list)


class PainPointCluster(BaseModel):
    cluster_id: str
    label: str
    summary: str = ""
    scope: Literal["company_specific", "domain_wide"]
    companies: list[str]
    claim_ids: list[str]
    sources: list[str] = Field(default_factory=list)


class ExtractedClaimDraft(BaseModel):
    """LLM structured output before validation."""

    company: str = ""
    """Which company the claim is about.

    A "Linear vs Jira" thread yields claims about both, so the extractor
    attributes each claim rather than filing everything under whichever
    competitor triggered the fetch.
    """

    dimension: Dimension
    text: str
    quote: str
    sentiment: Sentiment


class ExtractionResult(BaseModel):
    claims: list[ExtractedClaimDraft] = Field(default_factory=list)


class ClusterDraft(BaseModel):
    """LLM pain-point cluster before claim_id validation."""

    label: str
    summary: str = ""
    claim_ids: list[str] = Field(default_factory=list)


class ClusteringResult(BaseModel):
    clusters: list[ClusterDraft] = Field(default_factory=list)


class GapDraft(BaseModel):
    """LLM gap finding before claim_id validation."""

    dimension: Dimension
    description: str
    claim_ids: list[str] = Field(default_factory=list)


class GapAnalysisResult(BaseModel):
    competitor_has_target_lacks: list[GapDraft] = Field(default_factory=list)
    target_has_competitor_lacks: list[GapDraft] = Field(default_factory=list)
