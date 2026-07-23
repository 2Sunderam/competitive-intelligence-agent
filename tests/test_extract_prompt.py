from compete.llm.client import build_extract_prompt
from compete.nodes.extract import resolve_company


def test_extract_prompt_includes_research_focus():
    prompt = build_extract_prompt(
        competitor="Fathom",
        document_text="Fathom is a nonprofit AI policy org.",
        url="https://example.com",
        research_description="AI meeting notetaker for sales teams",
        target_company="Fireflies.ai",
        category_keywords=["meeting", "notetaker", "transcription"],
        roster=["Fireflies.ai", "Otter.ai", "Fathom"],
    )
    assert "AI meeting notetaker for sales teams" in prompt
    assert "Fireflies.ai" in prompt
    assert "meeting, notetaker, transcription" in prompt
    assert "This document was retrieved while researching: Fathom" in prompt
    assert "DROP off-topic content" in prompt
    assert "empty claims list" in prompt


def test_extract_prompt_lists_roster_for_attribution():
    prompt = build_extract_prompt(
        competitor="Linear",
        document_text="Linear vs Jira thread",
        url="https://example.com",
        research_description="issue tracking",
        roster=["Linear", "Jira", "Asana"],
    )
    for entry in ("- Linear", "- Jira", "- Asana"):
        assert entry in prompt
    assert "attribute every claim to exactly one" in prompt


def test_resolve_company_maps_labels_onto_roster():
    roster = ["Linear", "Jira", "ClickUp"]
    assert resolve_company("Linear", roster) == "Linear"
    assert resolve_company("clickup", roster) == "ClickUp"
    # Tolerates decorated answers
    assert resolve_company("Atlassian Jira", roster) == "Jira"
    # An empty label falls back to the company the document was fetched for
    assert resolve_company("", roster, fallback="Linear") == "Linear"
    # A company outside the roster is dropped, never misfiled
    assert resolve_company("Trello", roster) is None
