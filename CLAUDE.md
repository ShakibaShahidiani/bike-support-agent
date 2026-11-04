# Bike Support Agent (demo)

A small customer-support agent for a direct-to-consumer bike brand, built as an interview demo.
All data is MOCK data written by hand. No real brand logos, no scraping, no real customer data.
Fictional brand: "Ferro Cycles". English only.
Keep it small: this must be finished in one evening and fully explainable by its author.

## Goal
Show, end to end: a LangGraph workflow where the model labels and extracts and code decides, RAG with
model-year and policy-version handling, a human confirmation pause before actions, guardrails, an evaluation set,
latency/token logging, Langfuse tracing, prompt versioning, a REST API, and a simple chat UI with user feedback.

## Stack
- Python 3.11 (uv-managed `.venv`). LangGraph for the workflow. All chat model calls go through LiteLLM via
  `ChatLiteLLM` (default `openai/gpt-6-luna`, read from env var `MODEL`). Switching to Azure OpenAI must only require
  changing `MODEL` and credentials. Cost: use LiteLLM's cost if known, else a small fallback price table in `agent/config.py`.
- LangChain for retrieval (`langchain-chroma` + `OpenAIEmbeddings` (`EMBEDDING_MODEL`, default
  `text-embedding-3-small`) + text splitter) and `@tool` for catalogue operations called by nodes.
- Langfuse for tracing via its LangChain callback; PII masked; thumbs up/down become scores. Optional: no keys, no tracing.
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

## Tools (`agent/tools.py`), LangChain `@tool`s called by graph nodes, never chosen by the model
- `get_bike_spec(model, year)`, `check_compatibility(part_type, part_value, model, year)`,
  `get_order_status(order_id)`, `search_support_docs(query, model=None, year=None)` (drops superseded docs).
- Arguments are validated against the catalogue; unknown model/year returns a structured error listing
  known values (no fuzzy matching). Order IDs must match `CN-\d{4}`.
- `check_return_eligibility(order_id, today=None)`: exists, delivered, within `RETURN_WINDOW_DAYS` (config, must match
  current policy doc), not already returned. Changes nothing.
- `create_return(order_id, reason, today=None)`: re-checks eligibility; idempotent (same return id for the same order).

## Graph (`agent/graph.py`), one `handle_turn()` per message in `agent/agent.py`
- `intake`: length check (2000 chars), then one model call for the intent: continue, docs, parts, order, return, human, other.
  A new task clears the old task's slots.
- `fixed_reply`: templates for too long, other, human. No tickets: handover is a message only.
- `docs`: rewrite follow-ups to a standalone question, retrieve currently valid chunks, answer with chunk IDs validated
  in code and rendered as citations. No valid ID means "I don't know". Grounding check: numbers with mm/Wh/kg/V/km/days/€
  must appear in the chunks, else a fallback offering a human.
- `parts_collect`: extract model, one year, attribute, optional part value; ask for one missing or invalid field and end
  the turn. `parts_answer`: fill the answer from the catalogue; clear attribute and part value, keep the bike.
- `order_status`: order ID by regex, template from the data.
- `return_check`: order ID, eligibility in code, then a required reason. Nothing changes here.
- `confirm`: LangGraph interrupt. On resume a model call labels the reply confirm, cancel or other. Only confirm reaches
  `execute_return`. Other cancels the return, says so, and goes through `intake`. Model failure never executes.
- `execute_return`: `create_return`, template reply, clear state. The only node that changes anything.
- State is checkpointed per session with `InMemorySaver` (demo only).
- Safety: no separate check. The router sends damage, injury and battery hazards to `human`. PII is redacted before logging.
- Prompts: one file per model-calling node in `prompts/<version>/` (default `graph_v2`); the version is logged every turn.
  `prompts/system_v*.md` belong to the earlier agent loop and are kept with their eval results.
- Prompt rules (checked by eval): off-topic refusal, ignore prompt injection, never promise refunds, discounts or dates.

## Logging
- `logs/turns.jsonl`, one line per turn: timestamp, session_id, turn_id, prompt_version, model, tools called
  (name + args), latency_ms (total and per model call), input/output tokens, estimated cost, answer length,
  guardrail flags.
- `logs/feedback.jsonl`: timestamp, session_id, turn_id, rating.

## Evaluation (`eval/`)
- `cases.yaml`: regression cases that have been seen. `holdout.yaml`: blind cases, written without access to the code
  or prompts. Once a holdout case is read to fix a failure, it moves to regression and gets a blind replacement.
- Deterministic scoring only (no LLM judge).
- `run_eval.py`: table + summary (tool selection accuracy, ask-year rate, confirmation gate pass rate,
  abstention accuracy, average and p95 latency), saves `eval/results_<prompt_version>.json`.
- The headline score is the holdout: old agent loop (`results_holdout_loop_v3.json`) vs graph.

## Tests
- `tests/test_tools.py`: unit tests for every tool (no LLM calls), including the superseded-doc filter,
  return window inside/outside, idempotency, argument validation.
- `tests/test_graph.py`: the real graph with a scripted fake model: routing, slots, the confirmation pause, templates.

## Repo hygiene
- `.env.example` (MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, LANGFUSE_*), `.gitignore` (.env, logs, chroma db), `requirements.txt`.
- README: what it is (demo, mock data), architecture diagram (Mermaid), how to run, eval results v1 vs v2,
  design decisions, and "what I would add for production": 1 Azure OpenAI, 2 real order/CRM APIs behind auth,
  3 webhooks for order events, 4 monitoring dashboards + CSAT from real users + PII handling,
  5 customer verification (second detail or logged-in session; today anyone guessing an order id sees status),
  6 idempotency keys in a database, 7 timeouts/retries/fallback message on model API failure,
  8 rate limiting per session, 9 real handover to a human (tickets), 10 Postgres checkpointer so paused returns survive restarts.
- GitHub Actions workflow running `pytest tests/` on push.

## Working rules for Claude Code
- Build in small steps; after each step run it and show me the output. Commit locally after each step.
- Explain each file briefly after writing it, in plain language, so I can explain it in an interview.
- Do not add frameworks beyond the stack above (no LangChain agents/AgentExecutor).
