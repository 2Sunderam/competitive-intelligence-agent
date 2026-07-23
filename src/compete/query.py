"""Query decomposition + relevance gating.

Two problems this module solves:

1. **Query decomposition.** A single long query built from the company
   description returns zero results on both Reddit and HN's Algolia index
   (verified: ``nbHits=0``). Search engines here want short, intent-shaped
   queries. So one competitor fans out into several targeted queries
   ("X vs Y", "X pricing", "X alternative") which is also what surfaces
   opinionated threads rather than marketing pages.

2. **Relevance gating.** Both APIs are fuzzy. Reddit's search for
   ``"Linear" issue tracker`` returns r/BestofRedditorUpdates; Algolia
   reports 313k hits for ``ClickUp``. Every fetched document is scored
   against the competitor name + category keywords before it costs an
   LLM call.
"""

from __future__ import annotations

import re

# Words that signal an opinionated / evaluative document rather than a
# product landing page. Used both to build queries and to score results.
INTENT_TERMS: tuple[str, ...] = (
    "review",
    "reviews",
    "pricing",
    "price",
    "expensive",
    "cost",
    "alternative",
    "alternatives",
    "switch",
    "switched",
    "switching",
    "migrate",
    "migrated",
    "vs",
    "versus",
    "compare",
    "comparison",
    "complaint",
    "complaints",
    "frustrating",
    "slow",
    "buggy",
    "bug",
    "missing",
    "lacks",
    "wish",
    "annoying",
    "support",
    "onboarding",
    "worth it",
    "pros and cons",
)

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "who",
    "its", "it's", "their", "them", "they", "you", "your", "our", "not", "but",
    "into", "than", "then", "when", "what", "which", "while", "have", "has",
    "had", "been", "being", "will", "would", "can", "could", "should", "more",
    "most", "some", "such", "also", "like", "over", "under", "about", "built",
    "offers", "targeting", "teams", "team", "tool", "tools", "software",
    "platform", "product", "products", "users", "user", "company", "companies",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def keywords_from_description(description: str, *, limit: int = 6) -> list[str]:
    """Fallback category keywords when the LLM resolver gives us none."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", description or "")
    out: list[str] = []
    for token in tokens:
        low = token.casefold()
        if low in _STOPWORDS or low in out:
            continue
        out.append(low)
        if len(out) >= limit:
            break
    return out


def primary_keyword(keywords: list[str] | None, description: str = "") -> str:
    """Best single category phrase to append to a bare company name."""
    for kw in keywords or []:
        cleaned = kw.strip()
        if cleaned and " " in cleaned:  # multi-word keywords disambiguate best
            return cleaned
    for kw in keywords or []:
        if kw.strip():
            return kw.strip()
    fallback = keywords_from_description(description, limit=1)
    return fallback[0] if fallback else ""


def build_reddit_queries(
    competitor: str,
    *,
    keywords: list[str] | None = None,
    peers: list[str] | None = None,
    description: str = "",
    max_queries: int = 6,
) -> list[str]:
    """Short, intent-shaped Reddit queries for one competitor.

    Reddit's relevance search rewards a quoted brand plus a couple of terms.
    Long descriptive queries return an empty listing.
    """
    name = competitor.strip()
    if not name:
        return []
    category = primary_keyword(keywords, description)
    quoted = f'"{name}"'

    queries: list[str] = []
    if category:
        queries.append(f"{quoted} {category}")
    else:
        queries.append(quoted)
    queries.append(f"{quoted} review")
    queries.append(f"{quoted} pricing")

    for peer in (peers or [])[:2]:
        peer = peer.strip()
        if peer and _norm(peer) != _norm(name):
            queries.append(f"{name} vs {peer}")

    queries.append(f"{name} alternative")

    deduped: list[str] = []
    for q in queries:
        if q not in deduped:
            deduped.append(q)
    return deduped[:max_queries]


def build_hn_queries(
    competitor: str,
    *,
    keywords: list[str] | None = None,
    peers: list[str] | None = None,
    description: str = "",
    max_queries: int = 4,
) -> list[tuple[str, str]]:
    """HN queries as ``(query, tags)`` pairs.

    ``tags=comment`` is where opinions live; ``tags=story`` gives launches and
    announcements. Algolia ANDs the terms, so 2-3 tokens is the sweet spot.
    """
    name = competitor.strip()
    if not name:
        return []
    category = primary_keyword(keywords, description)

    pairs: list[tuple[str, str]] = []
    pairs.append((f"{name} {category}".strip(), "story"))
    pairs.append((f"{name} {category}".strip(), "comment"))
    for peer in (peers or [])[:1]:
        peer = peer.strip()
        if peer and _norm(peer) != _norm(name):
            pairs.append((f"{name} {peer}", "comment"))

    deduped: list[tuple[str, str]] = []
    for pair in pairs:
        if pair[0] and pair not in deduped:
            deduped.append(pair)
    return deduped[:max_queries]


def build_review_search_queries(
    competitor: str,
    *,
    keywords: list[str] | None = None,
    description: str = "",
) -> list[str]:
    """Web-search queries aimed at third-party review/opinion pages.

    Used with Jina search when G2 itself is CAPTCHA-walled — the goal is
    review-style prose from any reachable host, not one specific domain.
    """
    name = competitor.strip()
    if not name:
        return []
    category = primary_keyword(keywords, description)
    return [
        f'"{name}" review pros and cons {category}'.strip(),
        f'"{name}" {category} pricing complaints'.strip(),
    ]


def _name_variants(competitor: str, aliases: list[str] | None = None) -> list[str]:
    variants = {competitor.strip().casefold()}
    for alias in aliases or []:
        if alias and alias.strip():
            variants.add(alias.strip().casefold())
    # "Fireflies.ai" should also match a bare "fireflies" mention
    base = re.sub(r"\.(ai|io|com|dev|app)$", "", competitor.strip().casefold())
    if len(base) >= 4:
        variants.add(base)
    return [v for v in variants if v]


def _mentions(blob: str, term: str) -> int:
    if not term:
        return 0
    pattern = re.escape(term)
    # Word-boundary match so "linear" does not fire inside "linearly"
    if re.match(r"^[a-z0-9]", term) and re.search(r"[a-z0-9]$", term):
        pattern = rf"\b{pattern}\b"
    return len(re.findall(pattern, blob))


def relevance_score(
    *,
    text: str,
    title: str = "",
    competitor: str,
    aliases: list[str] | None = None,
    keywords: list[str] | None = None,
    peers: list[str] | None = None,
    require_context: bool = True,
) -> int:
    """Score a fetched document for on-topic-ness. Higher is better.

    Returns ``-1`` when the competitor is not mentioned at all — a hard
    reject, since a name-less document can only produce ungrounded claims.

    Mentioning the name is not sufficient either. Many product names are also
    ordinary words ("Linear", "Notion", "Shortcut"), so with ``require_context``
    the document must additionally mention a category keyword or one of the
    peer products. That is what separates a Jira-vs-Linear thread from a
    r/DestinyTheGame post describing a linear weapon progression.

    Set ``require_context=False`` for pages already known to be about the
    company — a resolved LinkedIn company page, for instance, where the "About"
    copy may never use the category vocabulary.
    """
    blob = _norm(f"{title}\n{text}")
    if not blob:
        return -1

    name_hits = sum(_mentions(blob, v) for v in _name_variants(competitor, aliases))
    if name_hits == 0:
        return -1

    has_keyword = any(_mentions(blob, k.casefold()) for k in (keywords or []) if k)
    has_peer = any(_mentions(blob, p.casefold()) for p in (peers or []) if p)

    if require_context and (keywords or peers) and not (has_keyword or has_peer):
        # Name appears, but nothing ties the document to this product category
        return 10

    score = min(name_hits, 3) * 10
    if has_keyword:
        score += 15
    if has_peer:
        score += 10
    if any(_mentions(blob, term) for term in INTENT_TERMS):
        score += 10

    if len(blob) < 200:
        score -= 5
    if len(blob) > 1500:
        score += 5

    return score


KEEP_THRESHOLD = 20


def is_relevant(score: int, *, threshold: int = KEEP_THRESHOLD) -> bool:
    return score >= threshold
