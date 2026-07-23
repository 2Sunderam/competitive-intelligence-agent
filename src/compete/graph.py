from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from compete.config import get_settings
from compete.logging_utils import get_logger, log_step, setup_logging
from compete.nodes.cluster import cluster_pain_points
from compete.nodes.compare import compare_node
from compete.nodes.extract import extract_document
from compete.nodes.ingest import ingest_competitor
from compete.nodes.intake import intake_node
from compete.nodes.report import write_report
from compete.nodes.resolve import resolve_entities_node
from compete.nodes.synthesize import synthesize_company
from compete.state import AgentState
from compete.store.evidence import reset_evidence_stores

log = get_logger("graph")


def prepare_run(state: AgentState) -> dict:
    setup_logging()
    settings = get_settings()
    run_id = state.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path(state.get("run_dir") or (settings.data_dir / run_id))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "skips.jsonl").touch()
    # Drop any store cached from a previous run in the same process
    reset_evidence_stores()
    log_step(log, "prepare_run.done", run_id=run_id, run_dir=str(run_dir))
    return {"run_id": run_id, "run_dir": str(run_dir)}


def fanout_ingest(state: AgentState) -> list[Send]:
    run_dir = state["run_dir"]
    description = state.get("description") or ""
    companies = state.get("competitors") or []
    log_step(log, "fanout_ingest", companies=len(companies))
    return [
        Send(
            "ingest",
            {
                "competitor": c,
                "run_dir": run_dir,
                "description": description,
                # Peers let each source spot genuine comparison threads
                "peers": [o.name for o in companies if o.name != c.name],
            },
        )
        for c in companies
    ]


def dispatch_extract(state: AgentState) -> list[Send] | str:
    """After all ingest tasks finish: map extract, or skip to after_map."""
    docs = state.get("documents") or []
    run_dir = state["run_dir"]
    description = state.get("description") or ""
    company_name = state.get("company_name") or ""
    competitors = state.get("competitors") or []
    keywords_by_name = {c.name: list(c.category_keywords or []) for c in competitors}
    roster = [c.name for c in competitors]
    log_step(log, "dispatch_extract", documents=len(docs))
    if not docs:
        return "after_map"
    return [
        Send(
            "extract_map",
            {
                "document": d,
                "run_dir": run_dir,
                "description": description,
                "company_name": company_name,
                "category_keywords": keywords_by_name.get(d.competitor, []),
                "roster": roster,
            },
        )
        for d in docs
    ]


def after_ingest(state: AgentState) -> dict:
    """Join node: runs once after all parallel ingest tasks complete."""
    log_step(
        log,
        "after_ingest",
        documents=len(state.get("documents") or []),
        skips=len(state.get("skips") or []),
    )
    return {}


def after_map(state: AgentState) -> dict:
    """Join node after extract (or empty ingest). Persists skip log."""
    run_dir = Path(state["run_dir"])
    skips = state.get("skips") or []
    if skips:
        with (run_dir / "skips.jsonl").open("a", encoding="utf-8") as fh:
            for skip in skips:
                fh.write(skip.model_dump_json() + "\n")
    log_step(
        log,
        "after_map",
        evidence_ids=len(state.get("evidence_ids") or []),
        skips_logged=len(skips),
    )
    return {}


def fanout_synthesize(state: AgentState) -> list[Send]:
    run_dir = state["run_dir"]
    companies = state.get("competitors") or []
    log_step(log, "fanout_synthesize", companies=len(companies))
    return [
        Send("synthesize_company", {"competitor": c, "run_dir": run_dir})
        for c in companies
    ]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("prepare_run", prepare_run)
    graph.add_node("intake", intake_node)
    graph.add_node("resolve_entities", resolve_entities_node)
    graph.add_node("ingest", ingest_competitor)
    graph.add_node("after_ingest", after_ingest)
    graph.add_node("extract_map", extract_document)
    graph.add_node("after_map", after_map)
    graph.add_node("synthesize_company", synthesize_company)
    graph.add_node("compare", compare_node)
    graph.add_node("cluster_pain_points", cluster_pain_points)
    graph.add_node("write_report", write_report)

    graph.add_edge(START, "prepare_run")
    graph.add_edge("prepare_run", "intake")
    graph.add_edge("intake", "resolve_entities")
    graph.add_conditional_edges("resolve_entities", fanout_ingest, ["ingest"])
    graph.add_edge("ingest", "after_ingest")
    graph.add_conditional_edges("after_ingest", dispatch_extract, ["extract_map", "after_map"])
    graph.add_edge("extract_map", "after_map")

    # Reduce stages run in sequence: cluster -> per-company synthesis -> compare
    # -> report. They could fan out in parallel, but then write_report would sit
    # behind two branches that finish in different supersteps and LangGraph would
    # fire it twice — once with a half-filled state. One trigger, one report.
    graph.add_edge("after_map", "cluster_pain_points")
    graph.add_conditional_edges("cluster_pain_points", fanout_synthesize, ["synthesize_company"])
    graph.add_edge("synthesize_company", "compare")
    graph.add_edge("compare", "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()


def run_agent(
    *,
    company_name: str,
    description: str,
    seed_competitors: list[str],
    run_dir: str | Path | None = None,
) -> AgentState:
    import asyncio

    from compete.config import get_settings

    setup_logging()
    get_settings.cache_clear()
    app = build_graph()
    payload: AgentState = {
        "company_name": company_name,
        "description": description,
        "seed_competitors": seed_competitors,
    }
    if run_dir is not None:
        payload["run_dir"] = str(run_dir)
    log_step(log, "run_agent.start", company=company_name, seeds=len(seed_competitors))
    result = asyncio.run(app.ainvoke(payload))
    log_step(
        log,
        "run_agent.done",
        run_dir=result.get("run_dir"),
        brief=result.get("brief_path"),
        docs=len(result.get("documents") or []),
    )
    return result
