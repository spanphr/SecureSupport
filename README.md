# SecureSupport

## A Guardrailed Agentic Customer Support System

One reliable support agent: it retrieves approved policy, looks up the authenticated
customer's own records through deterministic tools, refuses or escalates anything it
cannot do safely, and traces every request. Python 3.11+.

Phase 1 is complete: a single orchestrating agent, end-to-end guardrails, structured
human escalation, per-request observability, and an offline evaluation harness. There
is no HTTP API, UI, multi-agent graph, or real customer system yet. See
[`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) for the full design and
[`DECISIONS.md`](DECISIONS.md) for the reasoning.

> The fictional policies in `data/synthetic_support.json` are demo content, not
> production commitments. Nothing here connects to real accounts or payments.

---

## Quick start

Run from the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env          # optional; only needed for live model runs
python -m scripts.init_db     # creates/seeds data/support.sqlite3
```

Offline demonstration (no API key, no network, no cost):

```sh
python -m scripts.demo              # scenarios A-D
python -m scripts.demo --scenario C # one scenario
```

Deterministic checks (also no network):

```sh
python -m unittest discover -s tests -v   # 131 tests
python -m evaluation.pipeline             # 15 pipeline cases + retrieval metrics
```

Live mode uses the configured OpenAI model and Chroma index and **spends tokens on
your key**:

```sh
python -m scripts.index_knowledge
python -m scripts.demo --live
python -m scripts.chat --user usr_002 "Why is my payment pending?"
```

Live **evaluation** is separate and refuse-by-default; it will not run without an
explicit spend acknowledgement:

```sh
python -m evaluation.pipeline --save                 # offline baseline (free)
python -m evaluation.live                            # exits: requires --confirm-live
python -m evaluation.live --confirm-live --save --compare
```

---

## What the pipeline does

```
validate → decide → retrieve and/or authorized tool → compose → output guard → answer
                 ↘ refuse / clarify / escalate ────────────────────────────────↗
```

| Stage | Behaviour |
|---|---|
| **Validate** | Length, control characters, instruction-override attempts, secret requests, cross-customer requests, restricted-operation flags. Refusals here happen **before** any model call. |
| **Decide** | One structured `AgentDecision`: category, action, retrieval/tool/clarification/escalation flags. Inconsistent decisions are rejected. |
| **Retrieve** | Up to `retrieval_top_k` articles with ranks and distances. An optional distance ceiling turns "nothing relevant" into a human handoff instead of a confident guess. |
| **Act** | Allowlisted deterministic tools only. Arguments validated, identity from the session, reads and writes separated, writes confirmation-gated. |
| **Compose** | At most one grounded-answer call, receiving policy articles and confirmed tool results. |
| **Guard output** | Citations must be retrieved articles. Completion claims require a confirming write. Foreign record ids suppress the answer. |
| **Escalate** | Structured handoff persisted as a real ticket; the response returns its actual `tkt_…` id. |

The path is a bounded LangGraph workflow, not an autonomous agent: no cycles, no
planner, at most one tool execution and two model calls per request.

---

## Guardrails

| Threat | Control |
|---|---|
| Prompt injection in the message | Refused before retrieval or model calls |
| Injection inside retrieved text | Articles are data; claims still need tool confirmation |
| Cross-customer access | Refused on phrasing, and every SQL query binds the session owner id regardless of ranking |
| Unauthorized writes | Only one write tool exists; it requires explicit confirmation |
| Unsupported policy claims | Citation allowlist; unanswerable questions escalate |
| False "I refunded you" | Completion-claim detection; blocked unless a write tool confirmed |
| Leaking another customer's ids | Identifiers must appear in an owned tool payload |
| Restricted operations | No refund/cancel/delete/MFA tool exists, so they are unrepresentable |
| Backend failure | Structured failure codes; request degrades to escalation or a safe error |

Deliberate limitation: claim detection is pattern-based, and citation validation
proves traceability rather than entailment. Residual risk is documented in
[`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md#8-guardrail-matrix).

---

## Tools

Synthetic, owner-scoped, and deterministic. No LLM involvement.

| Tool | Kind | Purpose |
|---|---|---|
| `get_account_status` | read | Own account state and MFA flag (no name or email) |
| `get_payment_status` | read | One payment owned by the session customer |
| `get_ticket_status` | read | One ticket owned by the session customer |
| `create_support_ticket` | write | Open a ticket; requires confirmation |

Existing complaint CRUD remains available to the developer CLI for the earlier
storage stage: `python -m scripts.support_tools --user usr_002 list-complaints`.

---

## Escalation

A handoff is created for an explicit human request, missing knowledge, unresolved
ambiguity, tool failure, or authorization-sensitive requests. It contains a summary,
detected issue, retrieved evidence with ranks, tools attempted with outcome codes,
the reason, and the trace id, and is stored as a JSON ticket body so a human can act
without replaying the conversation.

---

## Observability

One structured JSON line per request on the `securesupport.pipeline` logger:
trace id, decision, retrieval ranking, tool names and codes, retrieval/LLM/total
latency, token usage, cost from **explicitly configured** pricing, escalation reason,
and error class names.

Message text, answers, ticket bodies, amounts, emails, and exception messages are
never logged; the trace carries a SHA-256 fingerprint and character count instead.
Cost is `null` unless `SECURESUPPORT_INPUT_COST_PER_MILLION_USD` and
`SECURESUPPORT_OUTPUT_COST_PER_MILLION_USD` are both set.

---

## Evaluation

Deterministic and live evaluation are deliberately separate.

**Offline** (`python -m evaluation.pipeline`) runs 15 cases through the real
guardrails, real SQLite tools, and a deterministic lexical retriever with scripted
decisions. Scenarios: correct FAQ retrieval, wrong article ranked first, no relevant
knowledge, correct and incorrect tool selection, unauthorized payment access, missing
record, execution failure, ticket creation, human escalation, prompt injection in
input and in retrieved documentation, false action claims, and multi-turn
clarification.

Recorded baseline with this commit:

| Metric | Value |
|---|---|
| Deterministic tests | 131 passing |
| Pipeline cases | 15/15 |
| Authorization violations | 0 |
| Unsupported claims | 0 |
| Citation violations | 0 |
| Tool dispatch accuracy | 1.00 |
| Escalation accuracy | 1.00 |
| Retrieval (offline lexical, 12 labeled queries) | Hit@1 0.75 · Hit@3 1.00 · MRR@3 0.861 |
| Retrieval (earlier live embeddings run) | Hit@1 11/12 · Hit@3 12/12 · MRR@3 0.958 |

The pipeline numbers validate orchestration and guardrail behaviour under scripted
model output, not live model accuracy. The lexical retriever exists for reproducible
offline measurement and is not the production retriever.

**Live** (`python -m evaluation.live --confirm-live`) runs nine labeled cases against
the real chat model and embedding index. It refuses without `--confirm-live`, refuses
a production configuration, seeds a temporary database, and prints its spend
footprint before starting. Scoring asserts an allowed status set, required sources,
expected tool use, and absolute safety checks (no authorization leak, no unconfirmed
completion claim, no invented citation, no forbidden text) rather than exact wording.
With `--save --compare` it writes `evaluation/reports/pipeline_live.json` and prints a
side-by-side table against the offline report. Its scoring logic is unit-tested; the
network path never runs in the test suite.

---

## Configuration

`app/config.py` reads `SECURESUPPORT_` variables and an optional `.env`. The API key
is optional and stored as `SecretStr`; tests and demos need no key.

| Variable | Default | Purpose |
|---|---|---|
| `SECURESUPPORT_OPENAI_API_KEY` | unset | Required only for live model/embedding runs |
| `SECURESUPPORT_CHAT_MODEL` | `gpt-4.1-mini` | Decision and composition model |
| `SECURESUPPORT_EMBEDDING_MODEL` | `text-embedding-3-small` | Retrieval embeddings |
| `SECURESUPPORT_DATABASE_PATH` | `data/support.sqlite3` | Support records |
| `SECURESUPPORT_CHROMA_PATH` | `data/chroma` | Vector index |
| `SECURESUPPORT_RETRIEVAL_TOP_K` | `3` | Articles retrieved per query |
| `SECURESUPPORT_RETRIEVAL_MAX_DISTANCE` | unset | Relevance ceiling (uncalibrated; opt-in) |
| `SECURESUPPORT_MAX_HISTORY_TURNS` | `6` | Bounded conversation history |
| `SECURESUPPORT_INPUT_COST_PER_MILLION_USD` | unset | Explicit input price |
| `SECURESUPPORT_OUTPUT_COST_PER_MILLION_USD` | unset | Explicit output price |
| `SECURESUPPORT_TRACE_LOG_ENABLED` | `true` | Emit one JSON trace line per request |

---

## Folder structure

- `app/agent/` — pipeline facade, LangGraph wiring, nodes, model adapter, factory.
- `app/guardrails/` — input validation, tool authorization, output validation.
- `app/tools/` — deterministic tools and the allowlisted registry.
- `app/schemas/` — decision, pipeline, domain, retrieval, and tool contracts.
- `app/storage/` — SQLite schema, transactions, seeding, owner-scoped repository.
- `app/retrieval/` — OpenAI embeddings and persistent Chroma search.
- `app/observability.py` — trace assembly, structured logging, cost estimation.
- `data/` — synthetic fixtures; generated database and index are git-ignored.
- `docs/SYSTEM_DESIGN.md` — architecture, trust boundaries, failure modes.
- `evaluation/` — labeled retrieval set, offline retriever, scripted model, offline pipeline harness.
- `scripts/` — demo, chat, database init, indexing, fixture validation.
- `tests/` — deterministic unit and end-to-end tests.

---

## Earlier stages

**Synthetic data.** `data/synthetic_support.json` holds fictional users, payments,
tickets, knowledge articles, complaints, and six deliberate failure prompts. Emails
use the reserved `.invalid` domain; amounts are integer AUD cents. Validate with
`python -m scripts.validate_data`.

**Retrieval.** OpenAI embeddings with explicit vectors in Chroma; collections are
fingerprinted by corpus and model so changed policies never reuse a stale index. Only
knowledge articles are indexed, never customer records. Search returns source ids,
titles, policy text, and cosine distances.

**Storage.** Standard-library `sqlite3` with schema version 1, composite
complaint/ticket ownership, bound parameters everywhere, atomic seeding with a
fixture fingerprint, and version-checked updates. Missing and foreign records return
the same error.

---

## Known limitations

- `SessionContext.customer_id` is a demo identity contract, not authentication.
- No durable conversation state: the caller passes bounded history each turn.
- One clarification round is supported; there is no resumable approval workflow.
- Relevance cutoffs are uncalibrated and disabled by default.
- Output claim detection is regex-based, so recall is finite.
- Escalation writes a ticket without a separate confirmation turn (documented).
- Single agent by design; multi-agent decomposition is a Phase 2 question.
- Live evaluation results depend on the model version and are reported, not asserted;
  only the offline suite is a gate.

## Next

Authenticated sessions, an HTTP transport, durable checkpointing with a human
approval queue for refunds and account changes, and calibrated relevance thresholds.
