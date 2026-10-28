# Foundation decisions

1. Use Python 3.11 or newer.
2. Keep this stage limited to project structure and configuration.
3. Use Pydantic Settings v2 for validated environment-based configuration.
   Its dependencies include Pydantic and dotenv support, so only
   `pydantic-settings` is listed as a direct dependency.
4. Prefix configuration variables with `SECURESUPPORT_`.
5. Make API credentials optional and mask their normal representation with
   `SecretStr`. Commit only a blank key placeholder in `.env.example`.
6. Define the settings class without a global instance; importing the module
   does not load configuration or contact external services.
7. Use `__init__.py` in Python package directories and `.gitkeep` in reserved
   non-package directories so the initial structure survives a Git checkout.
8. Defer business logic, LangGraph, Chroma, SQLite/SQLAlchemy integrations,
   FastAPI, Streamlit, evaluation logic, tests, and Docker setup to later stages.
9. Initialize a local Git repository without creating an initial commit.

## Synthetic domain data

10. Keep fixtures deterministic and fictional, with `.invalid` email addresses.
11. Store payment amounts as integer cents to avoid floating-point money errors.
12. Validate record schemas, unique IDs, user references, and complaint ownership.
13. Treat knowledge articles as demo policies and failure prompts as future
    evaluation cases. No agent, CRUD actions, or runtime guardrails are added yet.
14. Use standard-library unittest for data checks; no test dependency is needed.

## Knowledge retrieval

15. Use explicit OpenAI vectors and Chroma cosine distance; no implicit embedding
    function or model download. Default model: text-embedding-3-small.
16. Index knowledge articles only; return source IDs and text for later grounding.
17. Version collections by article content and embedding provider/model identity
    so changed policies and models cannot silently reuse an old index.
18. Report Hit@1, Hit@3, and MRR@3 on a 12-query demo set. Validate plumbing offline;
    report live semantic quality only after an actual API-backed evaluation.
19. Keep generated vectors and environments out of Git. Do not add answer
    generation, routing, or sensitive actions during this retrieval step.

## First agent workflow

20. Use a finite LangGraph StateGraph: classify, conditionally route, and terminate.
    Keep dependency injection for model and searcher so node behavior is testable offline.
21. Use structured OpenAI Responses outputs for routing and grounded policy answers;
    default chat model is gpt-4.1-mini, configurable independently of embeddings.
22. Implement direct, retrieval, tool-request, and out-of-scope routes. The tool
    branch is an explicit unavailable response until database tools are implemented.
23. Treat each request as independent; checkpointing and human approval are deferred.
24. Require policy source IDs to come from the retrieved set. This is source
    validation, not a complete entailment check or a replacement for guardrails.
25. Keep API errors out of user-facing responses. Store no Responses API objects
    remotely via these chat calls; local smoke reports contain demo requests only.
26. Keep the prior live retrieval report as historical evidence with its corpus
    hashes rather than claiming that older scores apply to the modified dataset.

## Deterministic storage and tools

27. Use standard-library sqlite3 for the small local dataset; no ORM dependency
    is required yet. Explicit initialization assigns schema version 1.
28. Keep knowledge search and customer-record storage separate. Only users,
    payments, tickets, and complaints are seeded into the support database.
29. Bind all caller values to SQL placeholders. Enable foreign keys on each
    connection and use a composite complaint/ticket owner reference.
30. Seed in one transaction; a saved fixture fingerprint makes reruns preserve
    edits and deletions. Refuse changed fixtures and unrelated existing schemas.
31. Require owner context on each tool call, and give missing and unauthorized
    records the same error. Context is supplied by the caller; real authentication
    is deferred, and the CLI user selector is for demo testing only.
32. Generate complaint IDs server-side. Track versions to reject stale updates
    and deletes, while preserving created_at. Deletion is hard deletion in this
    demo; audit history and request deduplication remain future work.
33. Expose CRUD through local tools and a developer CLI. Keep the LangGraph
    action branch unavailable until human approval and verified context exist.

## Phase 1 pipeline and guardrails

34. Supersede 22 and 33: the tool branch now executes allowlisted tools. The
    earlier unavailability note reflected a missing approval/identity boundary;
    that boundary is now session identity plus confirmation-gated writes.
35. Keep the orchestrator a bounded LangGraph workflow, not an autonomous agent.
    Every branching node sets `next`; terminal nodes have a plain edge to END, so
    a cycle is not representable. Terminal nodes must never consult stale state.
36. Represent the decision as one validated `AgentDecision`. No confidence field:
    a self-reported score cannot authorize anything. Consistency between action,
    tool name, retrieval, clarification, and escalation flags is enforced by a
    Pydantic validator, not by prompt wording.
37. Code owns identity, authorization, tool existence, and confirmation. Identity
    flows `SessionContext` -> `ToolContext` -> bound SQL owner parameter and is
    never read from the message or from model output.
38. Name the tools after the operations they perform (`get_account_status`,
    `get_payment_status`, `get_ticket_status`, `create_support_ticket`) and keep
    `get_payment`/`get_ticket` as thin CLI aliases so the earlier storage surface
    and its tests stay valid.
39. Separate read and write tools, asserted disjoint at import. Only writes declare
    a confirmation requirement, and a denied write returns `confirmation_required`
    instead of creating a record.
40. Treat tool failures as values (`ToolOutcome` codes), never exceptions. A
    failing handler must not leak diagnostics or abort the request.
41. Persist escalation as a ticket in the existing `tickets` table rather than
    adding a schema version. It reuses ownership, foreign keys, and transactions,
    needs no migration, and returns a real identifier.
42. Escalation writes without a separate confirmation turn. The customer either
    asked for a human or hit a path where the system cannot proceed safely, and a
    support ticket is not a financial or account change. This exception is tested
    and documented rather than left implicit.
43. Treat input heuristics as signals, not permissions. Structural controls
    (allowlist, owner-scoped SQL, output checks) enforce policy; patterns only decide
    when to refuse early or hand off.
44. Refuse cross-customer requests before retrieval, so ranking can never unlock
    access. Ownership is still enforced independently in SQL.
45. Validate output three ways: citations must come from the retrieved set;
    completion claims require a confirming write tool; record identifiers must
    appear in an owned tool payload. A blocked answer is replaced, not annotated.
46. Make the relevance ceiling opt-in (`retrieval_max_distance`). Cosine distance is
    not a calibrated probability, so a default threshold would be a fabricated
    guarantee. When configured, an empty filtered set escalates as missing knowledge.
47. Cap the workflow at two model calls. Greetings and capability questions are
    answered deterministically, and retrieval is skipped when it cannot help.
48. Report cost only from explicitly configured per-million-token prices, and token
    usage only from the provider's usage object. Unpriced models report `null`
    rather than a hard-coded price table that silently goes stale.
49. Trace one JSON record per request containing ids, codes, timings, usage, cost,
    and a message fingerprint. Never log message text, answers, ticket bodies,
    amounts, emails, or exception messages.
50. Split evaluation into deterministic offline and opt-in live runs. The lexical
    retriever exists only for reproducible offline measurement; production retrieval
    remains OpenAI embeddings in Chroma. Offline scores are not presented as model
    quality.
51. Add no new dependencies. The pipeline, guardrails, registry, tracing, and
    evaluation use Pydantic, the standard library, and libraries already installed.
52. Remove `app/schemas/support.py`. Its `RoutingDecision`, `SupportRequest`, and
    `SupportResponse` described the superseded four-route graph, including
    `action_performed: Literal[False]`, which contradicts Phase 1. Keeping it would
    have left two competing response contracts in the codebase.

## Live evaluation

53. Keep live evaluation in a separate module that refuses to run without
    `--confirm-live` and refuses a production configuration. Its scoring functions
    are unit-tested offline; the network path is never exercised by the test suite.
54. Score live cases on an allowed status set plus absolute safety invariants
    (no authorization leak, no unconfirmed completion claim, no invented citation,
    no forbidden text) rather than exact answer text, because wording varies across
    model versions while safety properties must not.
55. Run live cases against a seeded temporary database and treat the saved report as
    a review artifact: it includes answers, while structured traces never do. The
    offline report stays the gate; live numbers are reported, not asserted.

## Findings from the first live run

56. Normalize model decisions before validating them. Structured outputs require
    every field, so the model populated `tool_name`/`tool_arguments` for tool-free
    actions and strict validation rejected the whole request as an error; four of
    nine live cases failed on that artifact alone. `coerce_decision` now derives the
    flags and the tool name from the chosen action and discards fields that action
    does not use. This narrows the model's influence rather than widening it: the
    strict `AgentDecision` contract still runs, so a lookup can only ever map to
    `get_payment_status`, never to a tool the model names.
57. Leave `live_02_pending_payment` failing rather than changing it to pass. For
    "Why is my payment pending?" the model asks for the payment reference instead of
    explaining the general policy. That is a product judgment about helpfulness, not
    a safety defect, so it is recorded as an open question. The honest resolutions are
    a two-turn live case (clarify, then look up the supplied reference) or an explicit
    routing preference in the prompt. Editing the expectation to accept the current
    behaviour would hide the ambiguity instead of resolving it.
