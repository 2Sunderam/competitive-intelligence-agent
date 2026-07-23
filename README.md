# Competitive Intelligence & Market Gap Agent

Takes a company name, a one-paragraph description and a seed list of competitors. Reads what
real people say about all of them on public platforms, and produces an evidence-backed brief:
how the company compares, and where the unmet pain points in its domain are.

Every claim carries a **verbatim quote and a source URL**. That constraint is enforced in code,
not requested in a prompt.

**LangGraph** for orchestration, **OpenAI** for extraction and reasoning, append-only **JSONL**
for evidence. No vector DB — this is a traceability problem, not a retrieval problem.

---

## Setup

```bash
uv sync
cp env.example .env
```

Three keys are needed:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Claim extraction, entity resolution, clustering, gap analysis |
| `REDDIT_ACCESS_TOKEN` | Reddit search — browser session `token_v2` from DevTools → Application. See [Reddit access](#reddit-access) |
| `JINA_API_KEY` | Web search (`s.jina.ai`) and page reader (`r.jina.ai`). Free tier at [jina.ai](https://jina.ai) |

Jina technically works unauthenticated, but a run issues one search plus several reader calls per
company across the LinkedIn → G2 → review-fallback ladder, which exceeds the anonymous rate limit
partway through and turns the remainder into logged skips.

Everything else has a working default in `env.example` — model names, endpoints, result limits.
Hacker News needs no key.

## Run

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

Output goes to an auto-named folder, `data/runs/<UTC-timestamp>-<uuid8>/`:

| File | Contents |
|---|---|
| `competitive_brief.md` | Overview, competitor landscape, per-company findings, comparison table, gaps in both directions, domain-wide pain points, opportunities |
| `evidence.json` | Every claim and pain point with quote, source URL, platform, sentiment, company |
| `evidence.jsonl` | Append-only extraction log, one line per source document, with the source excerpt so any quote can be re-verified |
| `skips.jsonl` | Sources that were blocked, empty, rate-limited or off-topic |

`--run-dir <path>` overrides the generated folder name.

## Sample input

Two runs are committed under `data/runs/`.

```json
{
  "company_name": "Kandji",
  "description": "Kandji is an Apple device management (MDM) platform for IT teams, automating device setup, security compliance, patching and endpoint hardening for Mac, iPhone and iPad fleets.",
  "seed_competitors": ["Jamf", "Mosyle", "Addigy", "Hexnode", "JumpCloud"]
}
```

Chosen deliberately for thin online discussion — signal lives in niche subreddits
(r/macsysadmin, r/msp) rather than mainstream traffic. It produced 223 grounded claims from
37 unique source URLs across 64 documents, 9 pain-point clusters, and 7 + 6 gaps.

The second run uses Linear vs Jira / Asana / ClickUp / Shortcut / Notion — the well-discussed
case, for contrast.

---

## Pipeline

```
intake → resolve_entities → ingest ⇉ extract_map ⇉ cluster_pain_points
       → synthesize_company ⇉ compare → write_report
```

`⇉` marks LangGraph `Send` fan-out — per competitor for ingest and synthesis, per document for
extraction. The reduce stages run in sequence on purpose: as parallel branches they complete in
different supersteps and `write_report` fires twice, once with half-filled state.

| Node | Job |
|---|---|
| `intake` | Validate and dedupe competitors; target becomes competitor #0 |
| `resolve_entities` | One LLM call → canonical domain, G2 slug, LinkedIn slug, aliases, category keywords |
| `ingest` | Call every source client per competitor; fail soft per source |
| `extract_map` | One LLM call per document → grounded claims, attributed per company |
| `cluster_pain_points` | Cluster complaint claims across all companies |
| `synthesize_company` | Group one company's claims by dimension |
| `compare` | Comparison table + feature-level gaps both directions |
| `write_report` | Assemble the brief and `evidence.json`; introduces no new claims |

Dimensions: `pricing`, `features`, `ux`, `support`, `positioning`, `other`.

---

## How each competitor's sources are ingested and processed

Per competitor, `ingest` builds one `FetchContext` — name, aliases, category keywords, **peer
names**, resolved URLs — and hands it to every source client. Clients receive context rather
than a finished query string, because each backend wants a different query shape. Peers matter:
a document naming both the competitor and a peer is almost always a comparison thread, which is
where the useful claims are.

**Reddit**

1. Decompose into ~6 short intent-shaped queries: `"Kandji" device management`, `"Kandji" review`,
   `"Kandji" pricing`, `Kandji vs Jamf`, `Kandji vs Mosyle`, `Kandji alternative`.
2. Search each, dedupe posts by ID across queries.
3. Relevance-score every post, discard off-topic ones.
4. For the top threads, fetch top-level comments, dropping AutoModerator and deleted bodies. The
   post is often a one-line question — the complaints are in the replies.
5. Emit `title + selftext + comments`, tagged with the subreddit and an ISO date.

One long query built from the company description returns an empty listing; short intent queries
are what surface opinionated threads.

**Hacker News** — public Algolia API, no auth. Runs against `tags=story` and `tags=comment`;
comments carry the opinions, stories carry launches. HTML in comment bodies is unescaped.

**Web layer (Jina)** — a ladder, stopping once enough usable documents are found:

1. LinkedIn — read the resolved `/company/<slug>` page for positioning.
2. G2 — site-operator search, rank product/review URLs, read the best ones.
3. Review fallback — when G2 returns a CAPTCHA wall (it usually does), search review and
   pros-and-cons pages on any other reachable host and read those. This is what keeps a second
   public source alive.

**Normalisation** — the reader returns a JSON envelope; it is unwrapped to plain markdown before
extraction, or the model quotes JSON-escaped text. Documents are deduped by URL across sources,
since the web fallback can resurface a Reddit thread already fetched.

**Relevance gating** — every document is scored before it costs an LLM call. A document that
never names the company is rejected. A name mention alone is not enough either, since many
product names are ordinary words ("Linear", "Notion", "Shortcut"): it must also mention a
category keyword or a peer product. That rule is what separates a Jira-vs-Linear thread from an
r/DestinyTheGame post about linear weapon progression.

**Failure handling** — every client raises `SoftSourceError` carrying a specific reason: expired
token (401), rate limit (429), CAPTCHA wall, empty result, or "returned N posts, none on-topic".
`ingest` records the skip and moves on; a bare `except` catches anything unforeseen so a broken
client cannot kill a run. Skips are written to `skips.jsonl` and reproduced in the brief.

## How claims and pain points are grounded

**One LLM call per document**, never batched — batching is what causes hallucinated
cross-contamination between sources.

**Attribution is per claim.** A "Linear vs Jira" thread yields claims about both, so the model
attributes each to one company from the roster; a claim naming a company outside the roster is
dropped rather than misfiled. Without this, everything found while researching X lands on X — in
an early run a ClickUp complaint ended up on Linear's profile.

**Quotes are validated in code.** The quote must be a verbatim substring of the source text,
whitespace normalisation only. A claim failing that check is discarded before it is written. In
the committed Kandji run, 223 of 223 quotes re-verify against the stored excerpt:

```python
import json
for line in open("data/runs/<run>/evidence.jsonl"):
    r = json.loads(line)
    src = " ".join(r["raw_excerpt"].split())
    for c in r["claims"]:
        assert " ".join(c["quote"].split()) in src, c["claim_id"]
```

**Dedup at write time.** Claims hash on `(competitor, dimension, normalised text)`. On a hit the
new URL is appended to the existing claim's `linked_source_urls` instead of writing a second
claim. All concurrent extract tasks share one locked store, so IDs are allocated once and the
dedup index is global.

**Conflicts are preserved, never resolved.** Sentiment is per claim. When a dimension holds more
than one, the brief renders `⚠️ conflicting opinions preserved (negative, positive)` with both
sides listed. No code path collapses disagreement into a verdict.

**Pain points cite their members.** Complaint claims across *all* companies go to one clustering
call, reading the flat evidence log rather than per-company profiles — a complaint repeated
across three competitors is the domain signal. Returned IDs are validated against the known claim
set; hallucinated IDs are dropped and an empty cluster is discarded. Scope (`domain_wide` vs
`company_specific`) is computed from the evidence, not taken from the model.

**Gaps cite their evidence.** A gap citing no valid claim ID is discarded. The prompt requires
"no grounded claim found for X" rather than "X does not exist" — absence of evidence is not
evidence of absence.

**The report adds nothing.** `write_report` only narrates what earlier stages computed.

## How LinkedIn's access restrictions were handled

LinkedIn blocks automated scraping and has no open API for this, so it is **never scraped
directly**. Two indirect routes:

1. Search-engine results surfacing company pages — `site:linkedin.com/company "<name>"` via Jina.
2. Jina Reader on the resolved company URL, returning whatever renders publicly.

Accepting a page requires an **exact slug match** against the resolved URL or the company/alias
slug. A substring test is not enough: searching "Notion" surfaces `/company/notion-setup` and
`/company/revive-notion`, unrelated businesses. Regional subdomains
(`iq.linkedin.com/company/linearapp`) dedupe onto one entity.

**Limits of this approach:**

- LinkedIn yields company self-description only — positioning and marketing copy, never user
  opinion. It feeds the positioning dimension and contributes almost nothing to pain points.
- Coverage is inconsistent, depending on what LinkedIn renders to an unauthenticated request.
- When blocked or empty it is logged as a skip. No substitute is fabricated.

**G2 has the same problem** and deserves naming, since it was the intended primary review source.
It sits behind a CAPTCHA wall — the reader returns `"This page maybe requiring CAPTCHA"` with
empty content. The client detects the wall, records the marker, and falls through to third-party
review pages on reachable hosts, which is where the substantive review prose actually comes from.

### Reddit access

The assignment states Reddit's public JSON endpoints need no auth. That is no longer true —
tested directly, with fresh clients and no cookie carry-over, `www.reddit.com/search.json`,
`old.reddit.com/search.json`, subreddit search and subreddit listings all return **HTTP 403**,
with browser and script User-Agents alike. Only `oauth.reddit.com` with a bearer token returns
200.

So the client uses a browser session `token_v2`. The trade-off: it expires, so a reviewer needs
their own. On expiry the client returns a specific skip and stops early rather than burning every
query, and the run completes on remaining sources. A registered OAuth script app with a
client-credentials refresh flow is the correct production answer.

## One limitation

**Vendor marketing pages enter the corpus as evidence, and nothing distinguishes them from user
reports.**

When third-party review prose is thin, the fallback ladder widens until it finds something — and
a vendor's own domain always ranks. In the Kandji run, 25 of 223 claims came from `jumpcloud.com`
and `jamf.com`; *"Jamf Pro supports over 150 criteria"* is spec-sheet copy, not what real people
say. It is weighted identically to an r/macsysadmin complaint. Two consequences:

- **Sentiment skews positive** (138 positive vs 46 negative) — marketing copy is uniformly
  positive and outvotes thin community signal.
- **Coverage gaps fail silently.** Addigy drew 20 claims and zero negative ones. That does not
  mean Addigy has no weaknesses; too few people discuss it. The brief renders that identically to
  a genuine clean record. This is the more dangerous half, because it fails invisibly.

## One improvement with more time

**Source tiering, with pain points restricted to independent sources.**

Tag every document at fetch time — `community` (Reddit, HN), `third_party_review`,
`vendor_owned` — carry the tier onto each claim, then:

1. Never let a `vendor_owned` claim ground a pain point or weakness. Marketing copy can establish
   positioning and advertised features; it cannot establish that users are unhappy.
2. Show the source mix per company, so a reader sees Addigy's claims are mostly vendor copy while
   Hexnode's are community reports.
3. Emit a coverage warning when a company's community-source count is too low — "insufficient
   independent signal" is a finding, and more honest than an empty weaknesses section.

Roughly 40 lines, and it addresses the limitation, the positive skew and the silent coverage gaps
together.

---

## Tests

```bash
uv run pytest        # 3 tests, no network, ~0.2s
```

`tests/test_required_behaviors.py` — one test per required behavior:

| Behavior | Test asserts |
|---|---|
| Unreachable / blocked / empty source | A failing source is recorded as a `SourceSkip` with its real reason, the run does not crash, and the healthy source still returns documents |
| Dedup across sources | The second write returns `None`, one record exists, and the second URL is attached to the existing claim |
| Conflicting viewpoints | A positive and a negative claim on one dimension both survive |

No HTTP mocking. The failure case injects two stub clients into `ingest`; the other two run
against `EvidenceStore` on a temp path.

## Layout

```
src/compete/
  graph.py       LangGraph wiring and fan-out
  query.py       Query decomposition + relevance scoring
  models.py      Claims, evidence, profiles, clusters, gaps
  config.py      Env-driven settings
  sources/       reddit · hacker_news · jina · base
  nodes/         intake · resolve · ingest · extract · synthesize · compare · cluster · report
  store/         evidence.py (append-only JSONL) · validate.py (quote + hash)
  llm/client.py  Prompts and structured-output calls
tests/           test_required_behaviors.py
data/runs/       Committed run outputs
```

Stretch items implemented: configurable limits (`config.py`), a pluggable source layer
(implement `BaseSourceClient.fetch(FetchContext)`, append to `sources/__init__.py`), and skip
logging as a first-class artifact. Not implemented: competitor auto-discovery, per-claim
confidence scores, visualisations, cross-run caching.
