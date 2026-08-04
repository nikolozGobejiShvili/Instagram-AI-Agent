# Backend audit — 2026-08-04

Scope: `backend/` as it stands today, assessed against the goal of selling access to this
agent as a subscription product through an external website.

Method: static reading plus **executed** checks. Every claim below that carries a number was
measured on this machine, not inferred. Commands were run from the repository root with
`backend/.venv` (Python 3.10.11, pytest 9.0.3).

Severity key: **BLOCKER** (must be fixed before the product can be sold or built on) ·
**HIGH** · **MEDIUM** · **LOW** · **UNKNOWN** (could not be determined — stated, not guessed).

---

## Summary

The service works and the reels vertical is genuinely mature. It is nonetheless a prototype in
three specific respects that matter more than usual *because the thing being sold is metered
access*: there is no database, no authentication, and no version control.

The most serious finding is not a missing feature. It is that **the usage counter — the thing
customers pay against — loses 78% of concurrent increments and can leave the billing file
unreadable.** Everything else is downstream of that.

---

## 1. Usage accounting loses paid usage and can corrupt the billing file — **BLOCKER**

`BillingService.increment_generation_usage` (`backend/app/services/billing_service.py:340-352`)
is a read-modify-write over a whole JSON file with no lock:

```
items = self._load_items()      # read entire file
record = self._get_or_create_record(user_id)
record["monthly_generation_used"] += 1
items[user_id] = record
self._save_items(items)         # rewrite entire file
```

Measured, 50 concurrent increments against a throwaway copy of the store:

```
baseline count       : 1
concurrent increments: 50
expected final count : 51
actual final count   : 12
LOST INCREMENTS      : 39  (78% of paid usage unrecorded)
exceptions raised    : 28
    JSONDecodeError('Expecting value: line 1 column 1 (char 0)')
```

Two distinct defects:

- **Lost updates.** 39 of 50 generations were consumed but never charged.
- **Torn reads.** 28 readers observed a truncated file mid-write. In the running service each
  of those is a 500. A crash at the wrong moment leaves `billing_plans.json` permanently
  truncated — i.e. **loss of every customer's plan and usage state at once.** There is no
  backup and no version control to restore from.

This affects every entity, since all thirteen services share the same persistence shape, but
billing is the one that is money. `billing_plans.json` is rewritten on *every generation*.

**Fix:** transactional storage (SQLite at minimum) for billing and job state. A lock around the
existing JSON writes is not sufficient — it addresses lost updates but not the truncation
window, and it does not survive multiple processes.

## 2. Any caller can grant themselves the top plan — **BLOCKER**

`POST /api/v1/billing/plan/{user_id}/set` (`backend/app/api/routes/billing.py:17`) has no
authentication of any kind. Neither do the other two billing endpoints, nor the maintenance
cron endpoints. The entitlement layer is therefore advisory.

The only access control anywhere in the API is the internal admin key on the knowledge-pack
routes (`backend/app/api/routes/knowledge_packs.py:24`).

## 3. The test suite was red and non-hermetic — **BLOCKER — FIXED in this pass**

Baseline before any change:

```
9 failed, 40 passed in 21.36s
```

The nine failures were **not code defects**. They were environmental. Root cause, obtained by
calling the service directly because the API response hides it:

```
httpx.ConnectError: [WinError 10061] No connection could be made
because the target machine actively refused it
  -> langflow_service.py:1199  raise LangflowServiceError
  -> "AI generation failed. Please try again shortly."   (HTTP 502)
```

`load_dotenv()` runs at import time in `backend/app/main.py:24`,
`backend/app/services/langflow_service.py:13` and
`backend/app/services/deterministic_knowledge_retrieval_service.py:13`, so the test process
inherited `backend/.env` — real Langflow flow ids, provider flags, and whatever keys sat in the
developer's shell. `KnowledgePackService` uploads then call
`LangflowService.ingest_system_reels_knowledge`, which **no fixture stubs**, and it opened a
socket to a Langflow server that was not running. There was no `conftest.py` anywhere to
prevent any of this.

Consequences while this stood: test-driven development was impossible, "no previously passing
test may fail" was unenforceable, and a genuine regression was indistinguishable from a machine
difference.

**Fix applied:** added `backend/tests/conftest.py` (new file; no existing file modified). It

- neutralises `load_dotenv` before any application module is imported, and purges the ambient
  variables that steer runtime behaviour, so configuration is injected rather than inherited;
- supplies a small, obviously-fake deterministic test configuration in its place;
- blocks real outbound HTTP at `httpx.HTTPTransport`, so any unstubbed network call fails
  immediately with a named error. `TestClient` is unaffected by construction — it drives the
  app in-process through `starlette.testclient._TestClientTransport`, which is not an
  `httpx.HTTPTransport` subclass.

Result, verified:

```
49 passed in 3.67s
```

Same total of 49 tests, so nothing was skipped or lost. Re-running with a deliberately hostile
ambient environment (production-like flow id, base URL pointed at a live local port, provider
flags flipped, fake OpenAI key) still yields `49 passed` — the suite is now machine-independent.
Runtime fell from 21.4s to 3.7s because the network timeouts and retries are gone.

Note that two tests (`test_langflow_service_sends_only_sanitized_reels_runtime_payload`,
`test_agent_route_uses_safe_langflow_reels_without_local_chunk_injection`) were previously
passing *only* because `LANGFLOW_API_KEY` leaked in from `.env`. They now get an explicit test
value. **The honest baseline for this project is 49/49 — treat any failure as real.**

## 4. No version control, and live secrets sit in the working tree — **BLOCKER**

There is no `.git` and no ignore file. There is also no `Dockerfile`, `Procfile`, CI
configuration, `pyproject.toml` or `pytest.ini` — verified absent.

Meanwhile `backend/.env` holds a live Meta app secret, Langflow API key and internal admin key,
and `backend/app/data/instagram_connections.json` holds a real long-lived Meta access token in
cleartext. Any refactor from here has no undo, and the first careless `git init && git add -A`
would capture all of it permanently.

**Recommended (not performed):** rotate the Meta app secret, Langflow API key, internal admin
key and the stored long-lived token. They have been sitting in plaintext.

## 5. Every request handler is synchronous — **HIGH** (blocks the carousel feature)

Measured across `backend/app/api/routes/`:

```
route handlers -> async def: 1   sync def: 73
```

Each request occupies a threadpool worker for its whole duration, and services make blocking
`httpx.Client(timeout=30.0)` calls inside them.

Carousel generation as specified is one text call plus **one image generation per slide**. That
is plausibly 40–90 seconds. Two consequences: an external website will hit its own gateway
timeout, and a handful of concurrent carousels will exhaust the threadpool and stall **the whole
API**, not just carousels.

**Fix:** long-running generation must be a job — accept, return an id, complete asynchronously,
and let the caller poll. Retro-fitting this after the carousel feature is built means rebuilding
the carousel feature.

## 6. There is no authentication anywhere — **HIGH**

`user_id` arrives as a trusted path or body parameter on agent chat, every `instagram-*` route,
connected-accounts, generation-history and the maintenance cron routes. Fixing finding 2 alone
stops self-upgrade but still lets any caller transact as an existing paying customer.

This is a product decision, not only a technical one: either a shared site key on every
`/api/v1` route, or a deliberate, documented decision that the API is never publicly reachable
and the website is its only client.

## 7. Both wanted product models are uncredentialed; the direct-LLM path is currently dead — **HIGH**

Presence check of the process environment and `backend/.env` (names only, no values read):

| key | status |
|---|---|
| `OPENAI_API_KEY` | **absent** |
| `ANTHROPIC_API_KEY` | **absent** |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | **absent** |
| `GROQ_API_KEY` | present — **in the shell only, not in `.env`** |

Two implications. First, `LLMService` requires `OPENAI_API_KEY` and
`LLMService.run_agent` hard-raises for any provider that is not `openai`
(`backend/app/services/llm_service.py:1085-1091`) — so **the direct-LLM branch cannot currently
run at all**. Second, `GROQ_API_KEY` existing only in one developer's shell means the Langflow
reels component behaves differently on different machines, which is the same class of defect as
finding 3.

Adding Gemini and Anthropic requires a real provider abstraction; there is no interface to
extend today.

## 8. The error handler masks root causes — **HIGH**

`backend/app/error_handling.py:78-181` maps exceptions to API error codes by lowercase
substring matching over English prose, across roughly forty comparisons. Finding 3 is the
demonstration: a refused TCP connection was reported to the caller as
*"AI generation failed. Please try again shortly."* The true cause was invisible from the API
response and had to be recovered by invoking the service directly.

This will cost real debugging time in production, and it breaks silently whenever a message
string is edited.

## 9. Capability gaps against the product goal — **MEDIUM**

| Wanted capability | Status today |
|---|---|
| Content creation | **Exists** — 10 task types through `POST /api/v1/agent/chat` |
| Understand customer goals | **Partial** — `goal` / `niche` / `target_audience` are per-request fields on `AgentChatRequest`, never persisted |
| Tidy up the customer's own page | **Exists** — `profile_audit`, `performance_summary` over real Graph data |
| Analyse other Instagram pages | **Absent** — see finding 10 |
| Carousels | **Text only** — capped at 5 slides by `range(1, 6)` in `_parse_carousel` (`backend/app/services/agent_response_formatter_service.py:1340`); `CarouselSlideItem` is `{slide_number, headline, body}` with no image field |
| Image generation | **Absent entirely** — no Gemini SDK, no Pillow, no media endpoint, no output directory |

## 10. `tracked_accounts_limit` is billed for and enforced nowhere — **MEDIUM**

It is defined per tier in `PLAN_DEFAULTS` (0 / 3 / 10 / 50) and read by no code path. There is
no tracked-account entity at all. The only mechanism that touches a non-owned account is
`LinkContextService.extract_context`, which is an unrestricted redirect-following GET of a
user-supplied URL, reachable unauthenticated — an SSRF vector, and not a competitor-analysis
feature.

## 11. Current tier table, as it actually exists in code — **MEDIUM** (informational)

| plan | connected accts | tracked accts | generations/mo | carousel |
|---|---|---|---|---|
| `trial` | 1 | 0 | 15 | **no** |
| `creator` | 1 | 3 | 120 | **no** |
| `pro` | 2 | 10 | 400 | yes |
| `agency` | 10 | 50 | 2000 | yes |

Note the names are `trial / creator / pro / agency`, while `docs/instagram-agent-v1-scope.md`
claims `Basic / Pro / Agency`. Neither is authoritative; this needs an owner decision.

Latent defect, verified: `PLAN_DEFAULTS["agency"]["allowed_task_types"] is
BillingService.SUPPORTED_TASK_TYPES` returns `True` — the agency tier aliases the class
constant rather than copying it. Any in-place mutation of one silently rewrites the other.

## 12. The task-type list is duplicated across five modules — **MEDIUM**

`backend/app/schemas/agent.py`, `backend/app/services/billing_service.py`,
`backend/app/api/routes/agent.py`, `backend/app/services/deterministic_knowledge_retrieval_service.py`,
`backend/app/services/knowledge_pack_service.py`. Currently consistent in membership. Adding new
capabilities on top of five hand-synchronised lists will drift.

## 13. Documentation contradicts the code — **MEDIUM**

`docs/architecture-v1.md` describes a database, user authentication, and Langflow-owned
carousel/audit/link-analysis/content-plan flows. None of those four match reality.
`backend/docs/langflow_safe_reels_rag.md` documents flag values that are the **opposite** of what
`backend/.env` sets. Anyone onboarding from the docs will be misled.

## 14. No deployment path exists — **MEDIUM**

Verified absent: `Dockerfile`, `docker-compose.yml`, `Procfile`, `.github/`, `railway.json`,
`railway.toml`. The app is started by hand with `uvicorn app.main:app`. A product being sold
needs a defined way to reach production.

---

## Not determined

- **UNKNOWN — Meta `business_discovery` eligibility.** This gates the "analyse other Instagram
  pages" capability. `developers.facebook.com` is not reachable from this environment (DNS
  blocked), so the current permission set, the required caller account type, whether private and
  personal accounts are reachable at all, and whether **App Review** stands between the owner and
  shipping this could not be verified. It must be confirmed against live Meta documentation
  **before** the tier table treats tracked competitor accounts as a paid feature — App Review is
  a business process measured in weeks, not an engineering task.

- **BLOCKED — unit cost per carousel.** Cannot be measured without Gemini and Anthropic
  credentials (finding 7). The pro tier currently permits 400 generations/month; if a carousel
  is eight images, that is thousands of image generations per customer per month. The tier
  pricing cannot be set responsibly until this number exists.

---

## Recommended order

1. Persistence and job layer (findings 1, 5) — everything else is built on top of these.
2. Entitlement and authentication (findings 2, 6).
3. Version control and secret rotation (finding 4) — before, not after, the refactor.
4. Provider abstraction and credentials (finding 7).
5. Resolve the two open items above, then set the tier table (findings 9, 10, 11).
6. Feature work: goal intake, competitor analysis, carousel images.

Finding 3 is already resolved, which is what makes steps 1–6 verifiable at all.
