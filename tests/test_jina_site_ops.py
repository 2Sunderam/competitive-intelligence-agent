from __future__ import annotations

from compete.models import FetchedDocument, SourceKind
from compete.sources.jina import (
    company_slug,
    g2_search_query,
    linkedin_search_query,
    pick_best_g2_urls,
    score_g2_url,
)


def test_company_slug():
    assert company_slug("Fireflies.ai") == "fireflies-ai"
    assert company_slug("Otter.ai") == "otter-ai"
    assert company_slug("tl;dv") == "tl-dv"


def test_site_operator_queries():
    assert g2_search_query("Fireflies.ai") == 'site:g2.com/products "Fireflies.ai" reviews'
    assert linkedin_search_query("Fireflies.ai") == 'site:linkedin.com/company "Fireflies.ai"'
    assert 'meeting' in g2_search_query("Fathom", ["meeting", "notetaker"])
    assert 'meeting' in linkedin_search_query("Fathom", ["meeting", "AI"])


def test_score_prefers_product_reviews_matching_slug():
    company = "Fireflies.ai"
    good = "https://www.g2.com/products/fireflies-ai/reviews"
    ok = "https://www.g2.com/products/fireflies-ai"
    weak = "https://www.g2.com/products/other-tool/reviews"
    junk = "https://example.com/fireflies"
    assert score_g2_url(good, company) > score_g2_url(ok, company)
    assert score_g2_url(ok, company) > score_g2_url(weak, company)
    assert score_g2_url(junk, company) < 0


def test_pick_best_g2_urls_orders_and_limits():
    company = "Fireflies.ai"
    docs = [
        FetchedDocument(
            competitor=company,
            source=SourceKind.WEB_SEARCH,
            url="https://www.g2.com/products/other-tool/reviews",
            platform="g2",
            text="x",
        ),
        FetchedDocument(
            competitor=company,
            source=SourceKind.WEB_SEARCH,
            url="https://www.g2.com/products/fireflies-ai",
            platform="g2",
            text="x",
        ),
        FetchedDocument(
            competitor=company,
            source=SourceKind.WEB_SEARCH,
            url="https://www.g2.com/products/fireflies-ai/reviews?page=6",
            platform="g2",
            text="x",
        ),
        FetchedDocument(
            competitor=company,
            source=SourceKind.WEB_SEARCH,
            url="https://www.g2.com/products/fireflies-ai/reviews?qs=pros-and-cons",
            platform="g2",
            text="x",
        ),
        FetchedDocument(
            competitor=company,
            source=SourceKind.WEB_SEARCH,
            url="https://linkedin.com/company/fireflies",
            platform="linkedin",
            text="x",
        ),
    ]
    picked = pick_best_g2_urls(docs, company, limit=2)
    assert picked[0] == "https://www.g2.com/products/fireflies-ai/reviews"
    assert picked[1] == "https://www.g2.com/products/fireflies-ai"
    assert len(picked) == 2
