# Competitive Intelligence & Market Gap Agent

An autonomous agent that takes a company name plus a one-paragraph description, reads what
real people say about it and its competitors on public platforms, and produces an
evidence-backed brief: how the company compares, and where the unmet pain points in its
domain are.

Every claim in the output carries a **verbatim quote and a source URL**. Nothing is asserted
without one — that constraint is enforced in code, not just requested in a prompt.

Built with **LangGraph** for orchestration, **OpenAI** for extraction and reasoning, and a flat
append-only **JSONL** evidence store. There is no vector DB: this is a traceability problem, not
a retrieval-at-scale problem.

---

## Quickstart

```bash
uv sync
cp env.example .env      # fill in OPENAI_API_KEY and REDDIT_ACCESS_TOKEN
uv run pytest            # 21 tests, no network required
```

Run the agent:

```bash
uv run compete \
  --company "Kandji" \
  --description "Kandji is an Apple device management (MDM) platform for IT teams, automating device setup, security compliance, patching and endpoint hardening for Mac, iPhone and iPad fleets." \
  --competitor "Jamf" \
  --competitor "Mosyle" \
  --competitor "Addigy" \
  --competitor "Hexnode" \
  --competitor "JumpCloud"
```

Output lands in an auto-named folder, `data/runs/<UTC-timestamp>-<uuid8>/`:

| File | Contents |
|---|---|
| `competitive_brief.md` | The report — overview, landscape, per-company findings, comparison, gaps both directions, domain-wide pain points, opportunities |
| `evidence.json` | Every claim and pain point with quote, source URL, platform, sentiment, and the company it applies to |
| `evidence.jsonl` | Append-only extraction log, one line per source document, with the source excerpt so any quote can be re-verified |
| `skips.jsonl` | Sources that were empty, blocked, rate-limited or off-topic — logged, never substituted |

`--run-dir <path>` overrides the auto-generated folder name if you want a stable path.

### Environment

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes | Extraction + reduce reasoning |
| `OPENAI_EXTRACT_MODEL` | no | Per-document extraction. Default `gpt-5.4-nano` |
| `OPENAI_REASON_MODEL` | no | Entity resolution, clustering, gap analysis. Default `gpt-5.4-mini` |
| `REDDIT_ACCESS_TOKEN` | yes | Browser session `token_v2` — see [Reddit access](#reddit-access-why-a-token-is-required) |
| `REDDIT_USER_AGENT` | no | Sent with Reddit requests |
| `JINA_API_KEY` | no | Raises Jina rate limits; the free tier works without it |
| `HACKER_NEWS_API_URL` | no | Public Algolia endpoint, no key needed |

Without `OPENAI_API_KEY` the pipeline still runs end to end — extraction is skipped and the
reduce stages fall back to deterministic grouping, so you get structure but no claims.

---

## Sample input (test company profile)

The two committed runs under `data/runs/` use these profiles.

**Primary — well-discussed product:**

```json
{
  "company_name": "Linear",
  "description": "Linear is a project and issue tracking tool built for software product teams. It offers fast keyboard-driven issue tracking, sprint and cycle planning, roadmaps, and Git integrations, targeting startups and engineering teams who find traditional trackers slow and heavyweight.",
  "seed_competitors": ["Jira", "Asana", "ClickUp", "Shortcut", "Notion"]
}
```

**Secondary — deliberately thin online discussion, to test sparse-signal handling:**

```json
{
  "company_name": "Kandji",
  "description": "Kandji is an Apple device management (MDM) platform for IT teams, automating device setup, security compliance, patching and endpoint hardening for Mac, iPhone and iPad fleets.",
  "seed_competitors": ["Jamf", "Mosyle", "Addigy", "Hexnode", "JumpCloud"]
}
```

The Kandji run produced 223 grounded claims from 37 unique source URLs across 64 documents,
9 pain-point clusters, and 7 + 6 gaps in both directions — with community signal concentrated
in niche subreddits (r/macsysadmin, r/msp, r/JumpCloud) rather than mainstream traffic.

---

## How the system works

```
intake → resolve_entities → ingest (fan-out per competitor)
   → extract_map (fan-out per document) → cluster_pain_points
   → synthesize_company (fan-out per competitor) → compare → write_report
```

LangGraph `Send` fan-out parallelises the per-competitor and per-document work. The reduce
stages run in sequence deliberately: when `cluster` and `compare` ran as parallel branches,
`write_report` fired twice — once with a half-filled state — because the branches complete in
different supersteps. One trigger, one report.

| Node | File | Job |
|---|---|---|
| `intake` | `nodes/intake.py` | Validate and dedupe competitors; target becomes competitor #0 |
| `resolve_entities` | `nodes/resolve.py` | One LLM call → canonical domain, G2 slug, LinkedIn slug, aliases, category keywords per company |
| `ingest` | `nodes/ingest.py` | Fan out per competitor; call every source client; fail soft per source |
| `extract_map` | `nodes/extract.py` | One LLM call per document → grounded claims, attributed per company |
| `cluster_pain_points` | `nodes/cluster.py` | Reduce L3: cluster complaint claims across all companies |
| `synthesize_company` | `nodes/synthesize.py` | Reduce L1: group one company's claims by dimension |
| `compare` | `nodes/compare.py` | Reduce L2: comparison table + feature-level gaps both directions |
| `write_report` | `nodes/report.py` | Reduce L4: assemble the brief and `evidence.json`; introduces no new claims |

Dimensions are fixed: `pricing`, `features`, `ux`, `support`, `positioning`, `other`.

---

## How the agent ingests and processes each competitor's sources, step by step

For each of the N companies (target included), `ingest` builds one `FetchContext` — name,
aliases, category keywords, **peer names**, resolved URLs — and hands it to each source client.
Clients receive that context rather than a pre-baked query string, because each backend wants a
different query shape. Peers matter: a document mentioning both the competitor and a peer is
almost always a genuine comparison thread, which is where the useful claims live.

**1. Reddit** (`sources/reddit.py`)

1. Decompose into ~6 short intent-shaped queries: `"Kandji" device management`, `"Kandji" review`,
   `"Kandji" pricing`, `Kandji vs Jamf`, `Kandji vs Mosyle`, `Kandji alternative`.
2. Search each via `oauth.reddit.com/search`, dedupe posts by ID across queries.
3. Score every post for relevance and discard anything off-topic (see below).
4. For the top-ranked threads, fetch top-level comments from `oauth.reddit.com/comments/<id>`,
   dropping AutoModerator and `[deleted]`. The post body is often a one-line question — the
   complaints are in the replies.
5. Emit documents as `title + selftext + comments`, tagged with the subreddit
   (`reddit/r/macsysadmin`) and an ISO date.

**2. Hacker News** (`sources/hacker_news.py`)

Public Algolia API, no auth. Queries run against both `tags=story` and `tags=comment`; comments
carry the opinions, stories carry launches. HTML fragments in comment bodies are unescaped.
Same relevance gate applies — Algolia's matching is loose enough that a bare product name can
report hundreds of thousands of hits, almost none of which mention it.

**3. Web layer via Jina** (`sources/jina.py`) — a ladder, stopping when enough usable documents are found:

1. **LinkedIn** — read the resolved `/company/<slug>` page for positioning.
2. **G2** — site-operator search (`site:g2.com/products "<name>" reviews`), rank the product/review
   URLs, then read the best ones.
3. **Review fallback** — when G2 answers with a CAPTCHA wall (it usually does), search for
   review / pros-and-cons pages on any other reachable host and read those instead. This is what
   keeps a second public source alive.

**4. Normalisation and dedupe.** The reader returns a JSON envelope; it is unwrapped to plain
markdown before extraction, otherwise the model quotes JSON-escaped text. Documents are deduped
by URL across sources within a competitor, since the web-search fallback can resurface a Reddit
thread the Reddit client already returned. Claims are deduped at write time (below).

**5. Relevance gating** (`query.py`). Every fetched document is scored before it costs an LLM call.
A document that never names the company is rejected outright. A name mention alone is not enough
either — many product names are ordinary words ("Linear", "Notion", "Shortcut"), so the document
must additionally mention a category keyword or a peer product. That single rule is what
separates a Jira-vs-Linear thread from an r/DestinyTheGame post about linear weapon progression.

**6. Failure handling.** Every client raises `SoftSourceError` carrying a `SourceSkip` with a
specific reason — expired token (401), rate limit (429), CAPTCHA wall, empty result, or
"returned N posts but none were on-topic". `ingest` catches it, records the skip and moves on.
A bare `except` catches anything unforeseen so a broken client can never kill a run. Skips are
written to `skips.jsonl` and reproduced in the brief.

---

## How claims and pain points are grounded in source evidence

**Extraction is per document.** One LLM call per fetched document, never batched — batching is
what causes hallucinated cross-contamination between sources. The prompt supplies the research
focus, the category keywords and the full company roster.

**Attribution is per claim.** A "Linear vs Jira" thread produces claims about both, so the model
attributes each claim to one company from the roster. A claim naming a company outside the
roster is dropped rather than misfiled. Without this, everything found while researching X gets
filed under X — in an early run a ClickUp complaint ended up on Linear's profile.

**Quotes are validated in code, not trusted from the prompt.** `store/validate.py` requires the
quote to be a verbatim substring of the source text, allowing whitespace normalisation only. A
claim whose quote fails that check is dropped before it is ever written. In the committed Kandji
run, **223 of 223 quotes re-verify** against the excerpt stored alongside each record — you can
check this yourself without re-fetching anything:

```python
import json
for line in open("data/runs/<run>/evidence.jsonl"):
    r = json.loads(line)
    src = " ".join(r["raw_excerpt"].split())
    for c in r["claims"]:
        assert " ".join(c["quote"].split()) in src, c["claim_id"]
```

**Deduplication happens at write time.** Claims are hashed on
`(competitor, dimension, normalised text)`. On a hash hit the new source URL is appended to the
existing claim's `linked_source_urls` instead of writing a second claim. All concurrent extract
tasks share one locked store instance, so IDs are allocated once and the dedup index is global.

**Conflicting opinions are preserved, never resolved.** Sentiment is recorded per claim. When one
dimension holds more than one sentiment, `DimensionSummary.conflicting_sentiments` records that
and the brief renders `⚠️ conflicting opinions preserved (negative, positive)` with both sides
listed underneath. There is no code path that collapses disagreement into a verdict.

**Pain points cite their members.** Complaint claims (negative or mixed sentiment) across *all*
companies go to a single clustering call — deliberately reading the flat evidence log rather than
per-company profiles, since a complaint repeated across three competitors is the domain signal.
Returned cluster IDs are validated against the known claim set and hallucinated IDs are dropped;
a cluster left with no valid members is discarded. Scope (`domain_wide` vs `company_specific`) is
computed from the evidence, not taken from the model.

**Gaps cite their evidence.** Gap analysis receives each company's claims and must cite claim IDs.
A gap citing no valid ID is discarded entirely. The prompt requires gaps to be phrased as
"no grounded claim found for X" rather than "X does not exist" — absence of evidence is not
evidence of absence.

**The report stage introduces nothing.** `write_report` only narrates what earlier stages computed
and renders each claim with its quote, platform and link.

---

## How LinkedIn's access restrictions were handled

LinkedIn blocks automated scraping and offers no open API for this use case, so it is **never
scraped directly**. Two indirect routes are used:

1. **Search-engine results surfacing LinkedIn company pages** — `site:linkedin.com/company "<name>"`
   via Jina search, exactly the substitution the assignment describes as acceptable.
2. **Jina Reader on the resolved company URL**, which returns whatever is publicly rendered.

Accepting a page requires an **exact slug match** against the resolved LinkedIn URL or the
company/alias slug. A substring test is not enough: searching for "Notion" surfaces
`/company/notion-setup` and `/company/revive-notion`, which are unrelated businesses. Regional
subdomains (`iq.linkedin.com/company/linearapp`) dedupe onto one entity by slug.

**Limits of this approach, stated plainly:**

- LinkedIn yields **company self-description only** — positioning and marketing copy, never user
  opinion. It contributes to the positioning dimension and essentially nothing to pain points.
- Coverage is inconsistent. What the reader returns depends on what LinkedIn renders publicly to
  an unauthenticated request, which varies by company and over time.
- When the page is blocked or empty, it is logged as a skip. No substitute is fabricated.

**G2 is subject to the same reality** and is worth naming, since it was the intended primary
review source. G2 sits behind a CAPTCHA wall; the reader returns
`"This page maybe requiring CAPTCHA"` with empty content. Rather than pretending otherwise, the
client detects the wall, records the specific marker, and falls through to third-party review
pages on reachable hosts. In practice that is where the substantive review prose comes from.

### Reddit access: why a token is required

The assignment states Reddit's public JSON endpoints need no auth. That is no longer true.
Tested directly, with fresh clients and no cookie carry-over:

```
https://www.reddit.com/search.json              → HTTP 403
https://old.reddit.com/search.json              → HTTP 403
https://www.reddit.com/r/<sub>/search.json      → HTTP 403
https://www.reddit.com/r/<sub>/hot.json         → HTTP 403
https://oauth.reddit.com/search  + Bearer token → HTTP 200
```

403 on every anonymous endpoint regardless of User-Agent (browser or script style). So the client
uses a browser session `token_v2` as a bearer token — the only path that works.

**The trade-off, honestly:** the token expires, so this is not reproducible for a reviewer without
supplying their own. On expiry the client returns a specific skip
(`"Reddit token expired or invalid (HTTP 401) — refresh REDDIT_ACCESS_TOKEN"`) and stops early
rather than burning every query, and the run completes on the remaining sources. A registered
OAuth script app with a client-credentials refresh flow is the correct production answer.

---

## One limitation

**Vendor-owned marketing pages enter the corpus as evidence, and nothing distinguishes them from
user reports.**

When third-party review prose is thin, the review-fallback ladder keeps widening until it finds
something — and a vendor's own domain always ranks well. In the Kandji run, **25 of 223 claims**
came from `jumpcloud.com` and `jamf.com`: statements like *"Jamf Pro supports over 150 criteria"*
are spec-sheet copy, not what real people are saying. The brief weights them identically to a
r/macsysadmin complaint. Two consequences follow:

- **Sentiment skews positive** (138 positive vs 46 negative), because marketing copy is uniformly
  positive and outvotes thin community signal.
- **Silent uneven coverage.** Addigy drew 20 claims and **zero** negative ones. That does not mean
  Addigy has no weaknesses; it means too few people discuss it. The brief renders that identically
  to a genuine clean record, and nothing flags the difference. This is the more dangerous half of
  the problem, because it fails invisibly rather than obviously.

A related sampling artifact: JumpCloud shows the worst negative ratio (16 of 33) partly because
`r/JumpCloud` is effectively a support channel and structurally over-produces complaints, while
Kandji and Addigy have no equivalent subreddit. Companies are not sampled comparably.

## One improvement with more time

**Source tiering, with pain points restricted to independent sources.**

Tag every document at fetch time with a tier — `community` (Reddit, HN), `third_party_review`
(independent review sites), `vendor_owned` (the company's own domain, its LinkedIn page) — carry
the tier through to each claim, and then:

1. **Never let a `vendor_owned` claim ground a pain point or a weakness.** Marketing copy can
   establish positioning and advertised features; it cannot establish that users are unhappy.
2. **Show the source mix per company in the brief**, so a reader sees at a glance that Addigy's
   20 claims are mostly vendor copy while Hexnode's are community reports.
3. **Emit an explicit coverage warning** when a company's community-source count falls below a
   threshold — "insufficient independent signal" is a finding in its own right, and far more
   honest than an empty weaknesses section.

This is roughly 40 lines and it addresses the limitation above, the positive skew, and the silent
uneven coverage together. Beyond that, the next most valuable change is **semantic claim
deduplication**: the current hash-based dedup is exact-match, so two sources making the same point
in different words are never linked — in practice `linked_source_urls` stays empty on real data,
and corroboration across sources goes unrecorded.

---

## Design guarantees enforced in code

1. **Quote must be a verbatim substring** of the source text before a claim is written
   (`store/validate.py`) — not a prompt instruction, a hard filter.
2. **Source failures produce a `SourceSkip`**, never a crash and never a fabricated substitute
   (`sources/base.py`).
3. **Claim dedup by hash at write time**; corroborating sources are linked, not duplicated
   (`store/evidence.py`).
4. **Conflicting sentiments are recorded, never merged** (`nodes/synthesize.py`).
5. **One shared, locked evidence store** per run — concurrent extract tasks cannot allocate
   colliding IDs, the dedup index is global, and the JSONL rewrite cannot interleave with an append.
6. **LLM output is validated against known IDs** — hallucinated claim IDs in clusters and gaps are
   dropped, and an item left with no valid evidence is discarded.

## Tests

```bash
uv run pytest        # 21 tests, no network
```

Covering the three required scenarios and more:

| Scenario | Test |
|---|---|
| Unreachable / blocked / empty source | `test_reddit_missing_token_fails_soft`, `test_reddit_expired_token_fails_soft`, `test_unreachable_source_does_not_block_store` |
| Dedup of the same claim across sources | `test_dedup_links_second_source` |
| Conflicting viewpoints preserved | `test_conflicting_sentiments_kept_separate`, `test_synthesize_keeps_conflicting_sentiments` |
| Quote grounding | `test_quote_substring_validator`, `test_rejects_non_substring_quote` |
| Query decomposition + relevance gating | `test_reddit_bearer_search_flow`, `test_reddit_offtopic_results_are_filtered` |
| Per-claim company attribution | `test_resolve_company_maps_labels_onto_roster`, `test_extract_prompt_lists_roster_for_attribution` |
| Source parsing | `test_hn_parses_algolia_hits`, `test_hn_strips_html_from_comments`, G2/LinkedIn URL scoring |

## Project layout

```
src/compete/
  graph.py            LangGraph wiring, fan-out, run preparation
  query.py            Query decomposition + relevance scoring
  models.py           Pydantic models — claims, evidence, profiles, clusters, gaps
  state.py            LangGraph state and per-task payloads
  config.py           Settings (env-driven)
  sources/            reddit.py · hacker_news.py · jina.py · base.py
  nodes/              intake · resolve · ingest · extract · synthesize · compare · cluster · report
  store/              evidence.py (append-only JSONL) · validate.py (quote + hash)
  llm/client.py       Prompts and structured-output calls
tests/                21 tests
data/runs/            Committed run outputs
```

## Optional stretch items implemented

- **Configurable limits** — results per query, documents per competitor, comment depth, reader
  limits, relevance threshold, all in `config.py`.
- **Pluggable source layer** — every client implements `BaseSourceClient.fetch(FetchContext)`;
  adding a source means writing one class and appending it in `sources/__init__.py`.
- **Skip logging as a first-class artifact** — `skips.jsonl` plus a section in the brief.

Not implemented: competitor auto-discovery without a seed list, per-claim confidence scores,
visualisations, and cross-run caching.
