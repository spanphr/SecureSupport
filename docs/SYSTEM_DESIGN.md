# SecureSupport — System Design (Phase 1)

One guardrailed support agent that answers policy questions with cited evidence,
performs a small set of authorized account actions, and hands off to a human when
it cannot proceed safely.

This document describes what is built, why it is shaped this way, and where the
remaining risk sits. It intentionally excludes Phase 2 ambitions from the design
itself so the current system stays small and reviewable.

---

## 1. Goals and non-goals

**Goals**

- A single, reliable, testable path from customer message to final response.
- Answers grounded in retrieved policy, with source attribution.
- A small set of deterministic tools that enforce authorization server-side.
- No claim of a completed action without a confirming tool result.
- A structured human handoff that a support agent can act on directly.
- Every request traceable: decision, ranking, tools, latency, tokens, cost.
- Behaviour reproducible offline, with no paid API calls in the test suite.

**Non-goals for Phase 1**

- Multi-agent collaboration, planners, or autonomous loops.
- Real payments, refunds, or customer data.
- An authentication/identity provider, queues, or horizontal scale.
- Learned relevance thresholds, fine-tuning, or an LLM-as-judge evaluation.

---

## 2. Design principles

The structure follows the workflow-first position in Sinan Ozdemir's *Building
Agentic AI* (Pearson, 2025): establish a deterministic workflow with explicit
state before introducing autonomy, treat tools as typed contracts, evaluate beyond
accuracy, and optimize cost, latency, and privacy deliberately.

| Principle | How it is enforced here |
|---|---|
| Workflow before autonomy | A finite LangGraph `StateGraph`; each branching node sets `next`; terminal nodes have a plain edge to `END`, so no cycle is representable. |
| The model proposes, code disposes | `AgentDecision` is a validated contract, but identity, authorization, tool existence, confirmation, and claim checking are code (`app/guardrails`, `app/tools/registry.py`). |
| Tools are contracts, not prompts | Each tool declares a Pydantic argument model, read/write kind, confirmation requirement, and handler. Arguments are validated before the handler runs. |
| Explicit memory | No hidden state. The caller passes bounded history; each request starts from an empty state object. |
| Retrieval is evidence, not instruction | Retrieved articles are untrusted data. Ranking never grants permission; a cross-customer request is refused before retrieval. |
| Failures are values | Tools return `ToolOutcome` with a code. Model and retrieval exceptions become a safe `error` status, never a traceback in an answer. |
| Evaluation beyond accuracy | Offline suite reports Hit@1/Hit@3/MRR@3 plus tool dispatch, authorization violations, escalation correctness, unsupported claims, and latency. |
| Cost and latency are design constraints | Greetings bypass the LLM entirely; at most two model calls per request; token counts and explicit-price cost are traced. |
| Human-in-the-loop for irreversible work | Writes require confirmation; every unsafe or unanswerable path produces a persisted handoff ticket. |

---

## 3. Component architecture

Dependencies point inward: transports depend on the pipeline, the pipeline depends
on contracts and guardrails, and only the tool registry touches storage.

```
CLI (scripts/chat.py, scripts/demo.py)   HTTP + UI (app/api, frontend/)
                │                                      │
                └──────────────┬───────────────────────┘
                               ▼
                    app/agent/pipeline.py
                    SupportPipeline.run(request, session)
                               │
                      app/agent/graph.py  (LangGraph wiring)
                               │
    ┌──────────────┬───────────┼────────────┬───────────────┐
    ▼              ▼           ▼            ▼               ▼
 validate       decide       tool        retrieve        compose
 (guardrails/  (agent/     (tools/      (retrieval/     (agent/
  input.py)     model.py)   registry.py)  retriever.py)  model.py)
                               │                              │
                               ▼                              ▼
                     storage/repository.py            guardrails/output.py
                       (owner-scoped SQL)                      │
                                                               ▼
                                                        check_output
                                                               │
                                                     escalate (nodes.py)
                                                               │
                                              create_support_ticket → tickets
```

`app/observability.py` is orthogonal: the pipeline owns one `TraceBuilder` per
request and emits one `TraceRecord` at the end.

| Layer | Responsibility | Must not |
|---|---|---|
| `app/schemas/` | Validated contracts | Contain I/O or policy logic |
| `app/guardrails/` | Input, authorization, output rules | Call models or the database |
| `app/tools/registry.py` | Allowlist, argument validation, confirmation, error mapping | Interpret natural language |
| `app/storage/` | Schema, transactions, owner-scoped queries | Decide product policy |
| `app/agent/` | Orchestration, prompts, response assembly | Enforce authorization by prompt |
| `app/observability.py` | Trace assembly and redaction | Serialize message or record contents |

---

## 4. Trust boundaries

| Input | Trust | Control |
|---|---|---|
| Customer message | Untrusted | `guardrails/input.py`: length, control characters, instruction-override and secret patterns, cross-customer phrasing, restricted-operation flags |
| Retrieved article text | Untrusted | Confined to the composition payload; never executed or treated as instructions; citations restricted to retrieved ids |
| Model decision | Untrusted | `AgentDecision` consistency validator; tool name must equal the action's mapped tool; only allowlisted tools exist |
| Model answer | Untrusted | `guardrails/output.py`: citation allowlist, completion-claim detection, foreign record-id detection |
| Session customer id | Trusted | Supplied by the application; sole source of `ToolContext`; flows one way into SQL owner scope |
| Tool payload | Trusted but scoped | Produced by owner-scoped queries; ids in the answer are only allowed if they appear in an owned payload |

The system never asks the model whether a customer is authorized, and never reads
identity from message text.

---

## 5. Request lifecycle

```
validate ──ok──► decide ──respond──► respond ─────────────────────────────► END
   │                │
   │                ├──clarify──► clarify ───────────────────────────────► END
   │                ├──refuse───► refuse ────────────────────────────────► END
   │                ├──escalate─► escalate ──► check_output ─────────────► END
   │                ├──policy───► retrieve ──┬─ hits ──► compose ──► check_output ─► END
   │                │                       └─ none ──► escalate
   │                └──tool─────► tool ──┬─ ok ──► retrieve? ──► compose
   │                                     ├─ confirmation_required ──► check_output
   │                                     ├─ not_found ──► check_output
   │                                     ├─ not_authorized ──► refuse
   │                                     └─ failure ──► escalate ──► check_output
   └──invalid/injection/cross-customer──► refuse (before any model call)
```

Bounded by construction: at most one decide, one tool execution, one retrieval,
one composition, and one output check. The graph has no cycle.

`compose` is skipped when the answer is deterministic (greeting, refusal,
clarification, not-found, confirmation prompt), which is both cheaper and safer.

---

## 6. Decision contract

`AgentDecision` carries `category`, `action`, four capability flags, an optional
tool name/arguments, and an optional clarification question. A model self-reported
confidence would be untrustworthy, so no confidence field exists.

Actions: `respond`, `answer_policy`, `lookup_account`, `lookup_payment`,
`lookup_ticket`, `create_ticket`, `clarify`, `escalate`, `refuse`.

Enforced invariants (rejected at construction time):

- Tool actions must set `needs_tool` and name exactly their mapped tool with arguments.
- `answer_policy` requires retrieval.
- `needs_retrieval` is only valid for policy or tool-backed answers.
- `clarify` requires a question; other actions may not carry one.
- `needs_escalation` is true if and only if the action is `escalate`.
- `respond` is limited to greetings and capability questions.

---

## 7. Tools and authorization

| Tool | Kind | Arguments | Confirmation |
|---|---|---|---|
| `get_account_status` | read | none | no |
| `get_payment_status` | read | `record_id` | no |
| `get_ticket_status` | read | `record_id` | no |
| `create_support_ticket` | write | `subject`, `message` | yes |

Read and write sets are disjoint and asserted at import. Execution order is
allowlist lookup → Pydantic argument validation → confirmation policy → handler →
structured outcome. Every query binds `ToolContext.user_id`; missing and
foreign records return the identical message so record existence is not leaked.

There is deliberately no refund, cancellation, deletion, or account-update tool,
so those operations are unrepresentable rather than merely discouraged.

Outcome codes: `ok`, `unknown_tool`, `not_authorized`, `confirmation_required`,
`invalid_arguments`, `not_found`, `execution_failed`.

---

## 8. Guardrail matrix

| Threat | Control | Residual risk |
|---|---|---|
| Prompt injection in the message | Pattern refusal before any model call | Novel phrasing can pass; structural controls still hold |
| Injection inside retrieved text | Text is data; citations restricted; completion claims blocked | A subtle non-claim distortion is not detected |
| Cross-customer access | Refused on phrasing; owner-scoped SQL regardless of ranking | An unprefixed request is still limited by SQL scope |
| Unauthorized action | No write tool exists for refunds/deletes; writes need confirmation | Escalation writes bypass confirmation by design (see §9) |
| False completion claim | Regex completion detection + write-tool confirmation requirement | Recall is regex-limited; unusual phrasing may slip through |
| Leaking another customer's ids | Answer ids must appear in an owned payload | Numbers without an id prefix are not checked |
| Hallucinated policy | Citation allowlist; unsupported answers escalate | Citation validity proves traceability, not entailment |
| Secret disclosure | Secret-request refusal; no secret is ever placed in a prompt | Indirect inference is not modelled |
| Tool abuse / bad arguments | Allowlist, typed args, owner scope, confirmation | A valid-but-wrong tool choice can still return `not_found` |
| Crash on backend error | Exceptions mapped to codes; request degrades to escalation/error | Persistent outages rely on the escalation write succeeding |

---

## 9. Escalation

Triggers: explicit human request, missing knowledge (no evidence above the cutoff,
or a negative `supported` verdict), tool failure, authorization-sensitive
requests, and unresolved ambiguity. The handoff contains reason, category,
summary, detected issue, retrieved evidence with ranks, tools attempted and their
codes, the trace id, and a timestamp.

The handoff is persisted as a real ticket in the existing `tickets` table
(subject `Escalation: <reason> (<category>)`, JSON body). No schema migration was
required, and the response returns the actual `tkt_…` identifier.

**Deliberate exception:** escalation writes without a separate confirmation turn.
The customer either explicitly asked for a human or reached a path where the system
cannot proceed safely, and the write is a support record rather than a financial
or account change. This is recorded in `DECISIONS.md` and covered by tests.

---

## 10. Observability

One JSON line per request on the `securesupport.pipeline` logger:

- `trace_id`, `started_at`, `status`, `category`, `action`
- `retrieval_hit[]` — article id, title, rank, distance
- `retrieval_latency_ms`, `llm_latency_ms`, `llm_calls`, `total_latency_ms`
- `input_tokens`, `output_tokens`, `estimated_cost_usd`
- `tool_calls[]` — name and outcome code only
- `escalation_reason`, `errors[]` (exception class names), `flags[]`
- `message_fingerprint` (SHA-256 prefix) and `message_chars`

Never logged: message text, answers, ticket bodies, payment amounts, email
addresses, API keys, or exception messages. Cost is `null` unless both input and
output prices are explicitly configured.

---

## 11. Evaluation

Two layers, kept separate:

**Deterministic (default, free).** `python -m evaluation.pipeline` drives 15 cases
through the real guardrails, real SQLite tools, and a deterministic lexical
retriever with scripted decisions. It covers correct retrieval, a wrong article
ranked first, no relevant knowledge, correct and incorrect tool selection,
unauthorized access, missing records, execution failure, ticket creation, human
escalation, injection in input and in retrieved documents, false action claims, and
multi-turn clarification.

Metrics: Hit@1, Hit@3, MRR@3 (over the 12 labeled retrieval queries), tool dispatch
accuracy, authorization violations, escalation accuracy, unsupported claims,
citation violations, and latency.

**Live (opt-in, paid).** `python -m evaluation.live --confirm-live` runs nine labeled
cases through the real chat model and embedding index, scores them, and can save a
report and print a comparison against the offline baseline. Guards: the confirm flag
is mandatory, production configuration is refused, the database is a seeded temporary
copy, and the spend footprint is printed before the first call. Scoring asserts an
allowed status set plus absolute safety invariants rather than exact wording, so model
variation cannot be mistaken for a safety pass. `python -m scripts.demo --live`
exercises the same pipeline interactively. No live run is part of the test suite.

Recorded baseline in this repository: 138 deterministic tests passing; 15/15
pipeline cases; authorization violations 0; unsupported claims 0; citation
violations 0. Offline lexical retrieval scores Hit@1 0.75, Hit@3 1.00, MRR@3 0.861
on the 12 labeled queries. The earlier live embedding run scored Hit@1 11/12,
Hit@3 12/12, MRR@3 0.958 and remains the reference for semantic quality.

A live run of the harness (gpt-4.1-mini, 9 cases) passed 8 with zero authorization,
claim, or citation violations: tool accuracy 1.00, live retrieval Hit@1 0.5, mean
latency 1.26 s. Its first pass passed only 5, which surfaced a real defect: models fill
`tool_name`/`tool_arguments` even for tool-free actions because structured outputs
require every field, and strict validation turned that harmless artifact into an
error. `coerce_decision` now derives flags and the tool name from the chosen action
before validation, so those fields are not part of the trust surface. The remaining
failure is a routing preference, not a safety issue: a customer-specific pending-payment
question is answered with a request for the payment reference instead of general policy.

The numbers measure orchestration and guardrail behaviour under scripted model
output. They are not a claim about live model accuracy.

---

## 12. Failure modes

| Failure | Behaviour | User sees |
|---|---|---|
| Model call fails | `error`; trace records the exception class | Generic retry message, no account action claimed |
| Retrieval backend fails | `error`; no composition attempted | Same |
| No evidence above cutoff | `escalated` with `missing_knowledge` | Ticket id, honest limitation |
| Model declines to answer | `escalated` with `missing_knowledge` | Ticket id |
| Record absent or foreign | `insufficient_context` | Neutral "could not find" message |
| Tool backend error | `escalated` with `tool_failure` | Ticket id; attempt history recorded |
| Write without confirmation | `clarification_required` | Explicit confirmation prompt |
| Escalation write fails | `error` | "could not record a handoff"; no false claim |
| Answer claims unconfirmed action | `unverified_action` | Replaced with an explicit non-confirmation |
| Answer names a foreign record | `refused`; citations cleared | Neutral refusal |

---

## 13. Deliberate omissions and next steps

Not built in Phase 1, in rough order of value:

1. Verified authentication to replace the `SessionContext` demo contract; the HTTP
   transport already resolves identity server-side from a signed cookie, so only the
   issuer changes.
2. A streaming transport (server-sent events) over the same pipeline, and a deployed
   demo that rate-limits the paid model.
3. LangGraph checkpointing for durable multi-turn state and resumable approvals.
4. A human approval queue for refunds and account changes (the missing write path).
5. Calibrated relevance thresholds derived from live retrieval data.
6. An injection classifier to complement the current pattern rules.
7. Audit log retention, rate limiting, and per-tenant knowledge bases.
8. Live evaluation in CI with recorded fixtures and per-version score tracking.
