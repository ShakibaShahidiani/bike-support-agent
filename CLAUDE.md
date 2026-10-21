# Bike Support Agent (demo)

A small customer-support agent for a direct-to-consumer bike brand, built as an interview demo.
All data is MOCK data written by hand. No real brand logos, no scraping, no real customer data.
Fictional brand: "Ferro Cycles". English only.
Keep it small: this must be finished in one evening and fully explainable by its author.

## Goal
Show, end to end: tool calling, agentic RAG with model-year version handling, a human confirmation
gate before actions, guardrails, an evaluation set, latency/token logging, prompt versioning,
a REST API, and a simple chat UI with user feedback.

## Stack
- Python 3.11 (uv-managed `.venv`), LiteLLM for all chat model calls (default `openai/gpt-6-luna`,
  read from env var `MODEL`). Switching to Azure OpenAI must only require changing `MODEL` and credentials.
  Cost: use LiteLLM's cost if known, else a small fallback price table in `agent/config.py`.
- LangChain only for (a) retrieval: `langchain-chroma` + `OpenAIEmbeddings` (`EMBEDDING_MODEL`,
  default `text-embedding-3-small`) + text splitter, and (b) tool definitions via `@tool` (JSON schema
  from the function signature). The agent loop, confirmation gate, guardrails and logging are hand-written.
- ChromaDB (local) for support documents, with metadata `model`, `year`, `doc_type`, `valid_from`, `valid_to`.
- FastAPI: `POST /chat` (session_id, message) and `POST /feedback` (session_id, turn_id, rating).
- Streamlit chat UI calling the API, with thumbs up / down per answer.
- pytest for unit tests of tools. A separate eval runner for agent behaviour.

## Mock data (`data/`)
- `bikes.json`: Strada GX (gravel), Aero R1 (road), Ridge E (e-MTB), Metro C (city), each 2023 and 2025.
  Each entry: frame, fork, brake type, max rotor size (front/rear), tyre clearance, battery (if e-bike).
  Differences: Strada GX rotor 160→180mm and tyre clearance 42→45mm; Ridge E battery 625→750Wh.
- `orders.json`: 5 orders (`CN-1042` …): delivered inside the 30-day window, delivered outside it,
  shipped, processing, cancelled. No real names or emails.
- `docs/`: 4 to 6 short markdown support docs, all labelled "fictional policy for demo".
  Returns policy in TWO versions: old (14 days, buyer pays shipping, valid 2023-01-01 to 2025-12-31)
  and current (30 days, free return shipping, valid_from 2026-01-01, no valid_to).

## Tools (`agent/tools.py`), LangChain `@tool`s
Read-only:
- `get_bike_spec(model, year)`, `check_compatibility(part_type, part_value, model, year)`,
  `get_order_status(order_id)`, `search_support_docs(query, model=None, year=None)` (drops superseded docs).
- Arguments are validated against the catalogue; unknown model/year returns a structured error listing
  known values (no fuzzy matching). Order IDs must match `CN-\d{4}`.
Action (requires confirmation):
- `start_return(order_id, reason)` never executes on the first call: it stores a pending action in the
  session. The next user message is checked in code against fixed confirm phrases; match → execute,
  anything else → cancel pending action and handle the message normally.
  Checks: order exists, status delivered, within `RETURN_WINDOW_DAYS` (config, must match current policy doc;
  real date, injectable `today` for tests). Idempotent: a second return on the same order returns the same id.

## Guardrails (code)
- Input length cap 2000 chars. PII (emails/phones) redacted before logging.
- Safety escalation (crash, injury, frame damage, battery): keyword check → model called without tools and
  with a forced handover instruction → post-check (mentions human handover, no spec numbers, no repair verbs)
  → otherwise fixed handover text.
- Grounding post-check: numbers with mm/Wh/kg/V/km/days/€ in the answer must appear in that turn's tool
  results, else safe fallback offering a human.
- Prompt rules (checked by eval): off-topic refusal, ignore prompt injection, never promise refunds,
  discounts or dates.

## Agent loop (`agent/agent.py`)
- System prompt loaded from `prompts/system_v1.md` (later `system_v2.md`); the prompt version is logged with every turn.
- Loop: call model with tools, execute requested tools, feed results back, max 5 tool iterations, then answer.
- Rules: if the model year is unknown and the answer depends on it, ask for it instead of guessing.
  Answer only from tool results. If a tool returns nothing, say so and offer a human agent. Never invent specs, prices or order data.
- Session memory: keep the conversation per `session_id` in memory (dict) — demo only.

## Logging
- `logs/turns.jsonl`, one line per turn: timestamp, session_id, turn_id, prompt_version, model, tools called
  (name + args), latency_ms (total and per model call), input/output tokens, estimated cost, answer length,
  guardrail flags.
- `logs/feedback.jsonl`: timestamp, session_id, turn_id, rating.

## Evaluation (`eval/`)
- `cases.yaml`: ~18 cases with expectations `expected_tools`, `must_ask_year`, `must_not_execute_return`,
  `must_abstain`, `must_escalate`, `answer_contains`, `answer_not_contains`.
  Include: rotor without year, rotor 2023 vs 2025, order status, return without / with confirmation,
  unknown model (abstain), returns policy (current only), off-topic, prompt injection, safety escalation,
  no promises of refunds/discounts/dates, unknown year.
- Deterministic scoring only (no LLM judge).
- `run_eval.py`: table + summary (tool selection accuracy, ask-year rate, confirmation gate pass rate,
  abstention accuracy, average and p95 latency), saves `eval/results_<prompt_version>.json`.
- Run v1, look at failures, write `system_v2.md`, run again, compare. Keep both result files.

## Tests
- `tests/test_tools.py`: unit tests for every tool (no LLM calls), including the superseded-doc filter,
  `start_return` gate, return window inside/outside, idempotency, argument validation.

## Repo hygiene
- `.env.example` (MODEL, OPENAI_API_KEY, EMBEDDING_MODEL), `.gitignore` (.env, logs, chroma db), `requirements.txt`.
- README: what it is (demo, mock data), architecture diagram (Mermaid), how to run, eval results v1 vs v2,
  design decisions, and "what I would add for production": 1 Azure OpenAI, 2 real order/CRM APIs behind auth,
  3 webhooks for order events, 4 monitoring dashboards + CSAT from real users + PII handling,
  5 customer verification (second detail or logged-in session; today anyone guessing an order id sees status),
  6 idempotency keys in a database, 7 timeouts/retries/fallback message on model API failure,
  8 rate limiting per session, 9 handover to a human on frustration or repeated failures.
- GitHub Actions workflow running `pytest tests/` on push.

## Working rules for Claude Code
- Build in small steps; after each step run it and show me the output. Commit locally after each step.
- Stop for the user after the agent loop + confirmation gate work, and after the v1 eval results.
- Explain each file briefly after writing it, in plain language, so I can explain it in an interview.
- Do not add frameworks beyond the stack above (no LangGraph, no LangChain agents/AgentExecutor).
