from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from compete.models import (
    CompanyProfile,
    ComparisonTable,
    Competitor,
    FetchedDocument,
    PainPointCluster,
    SourceSkip,
)


class AgentState(TypedDict, total=False):
    # Intake
    company_name: str
    description: str
    seed_competitors: list[str]
    competitors: list[Competitor]
    run_id: str
    run_dir: str

    # Ingest / map
    documents: Annotated[list[FetchedDocument], operator.add]
    skips: Annotated[list[SourceSkip], operator.add]
    evidence_ids: Annotated[list[str], operator.add]

    # Reduce outputs
    company_profiles: Annotated[list[CompanyProfile], operator.add]
    comparison: ComparisonTable | None
    pain_clusters: list[PainPointCluster]
    brief_path: str
    evidence_json_path: str
    errors: Annotated[list[str], operator.add]


class IngestTask(TypedDict):
    competitor: Competitor
    run_dir: str
    description: str
    # Other companies in the run — used to spot comparison threads
    peers: list[str]


class ExtractTask(TypedDict):
    document: FetchedDocument
    run_dir: str
    description: str
    company_name: str
    category_keywords: list[str]
    # Every company in the run — the extractor attributes each claim to one
    roster: list[str]


class SynthTask(TypedDict):
    competitor: Competitor
    run_dir: str
