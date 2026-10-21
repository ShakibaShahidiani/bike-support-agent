# Bike Support Agent (demo)

A small customer-support agent for a direct-to-consumer bike brand, built as an interview demo.
All data is MOCK data written by hand. No real brand logos, no scraping, no real customer data.
Keep it small: this must be finished in one evening and fully explainable by its author.

## Goal
Show, end to end: tool calling, agentic RAG with model-year version handling, a human confirmation
gate before actions, guardrails, an evaluation set, latency/token logging, prompt versioning,
a REST API, and a simple chat UI with user feedback.

## Stack
- Python 3.11, LiteLLM for all model calls (default model: a Groq model that supports tool use,
  e.g. `groq/llama-3.3-70b-versatile`; model name read from env var `MODEL`).
  Switching to Azure OpenAI must only require changing `MODEL` and credentials.
- ChromaDB (local) for support documents, with metadata `model`, `year`, `doc_type`, `valid_from`, `valid_to`.
- FastAPI: `POST /chat` (session_id, message) and `POST /feedback` (session_id, turn_id, rating).
- Streamlit chat UI calling the API, with thumbs up / down per answer.
- pytest for unit tests of tools. A separate eval runner for agent behaviour.

## Mock data (`data/`)
- `bikes.json`: 4 models x 2 model years (e.g. a gravel bike 2023 and 2025, a road bike, an e-MTB, a city bike).
  Each entry: frame, fork, brake type, max rotor size (front/rear), tyre clearance, battery (if e-bike).
  Make at least two specs differ between years of the same model (e.g. max rotor 160mm in 2023, 180mm in 2025).
- `orders.json`: 5 orders with id (e.g. `CN-1042`), status, items, order date, delivery date. No real names or emails.
- `docs/`: 4 to 6 short markdown support docs (returns policy, warranty, sizing guide, care).
  The returns policy exists in TWO versions with different terms and `valid_from`/`valid_to` metadata;
  only the current one may be used. Label all docs "fictional policy for demo".

## Tools (`agent/tools.py`), each with a JSON schema
Read-only:
- `get_bike_spec(model: str, year: int)` returns the spec or a not-found error.
- `check_compatibility(part_type: str, part_value: str, model: str, year: int)` returns compatible true/false + reason from the spec.
- `get_order_status(order_id: str)` returns status and dates only.
- `search_support_docs(query: str, model: str | None, year: int | None)` vector search with a metadata filter
  that drops superseded documents (only `valid_to` empty or in the future).
Action (requires confirmation):
- `start_return(order_id: str, reason: str)` must NOT execute on the first call. The agent returns a
  confirmation request to the user; it executes only after the user's next message explicitly confirms.
  Enforce this in code (a pending-action state in the session), not only in the prompt.

## Agent loop (`agent/agent.py`)
- System prompt loaded from `prompts/system_v1.md` (later `system_v2.md`); the prompt version is logged with every turn.
- Loop: call model with tools, execute requested tools, feed results back, max 5 tool iterations, then answer.
- Rules: if the model year is unknown and the answer depends on it, ask for it instead of guessing.
  Answer only from tool results. If a tool returns nothing, say so and offer a human agent. Never invent specs, prices or order data.
- Session memory: keep the conversation per `session_id` in memory (dict) — demo only.

## Logging (`logs/turns.jsonl`)
One line per turn: timestamp, session_id, turn_id, prompt_version, model, tools called (name + args),
latency_ms (total and per model call), input/output tokens, estimated cost if LiteLLM provides it, final answer length.

## Evaluation (`eval/`)
- `cases.yaml`: 12 to 15 cases. Each has user message(s) and expectations:
  `expected_tools` (names), `must_ask_year` (bool), `must_not_execute_return` (bool), `must_abstain` (bool),
  `answer_contains` (optional keywords).
  Include: rotor question without year (must ask), with year 2023 vs 2025 (different answers),
  order status, return without confirmation (must not execute), return with confirmation,
  a spec for a model that does not exist (must abstain), returns question (must use current policy only),
  an off-topic question (polite refusal).
- `run_eval.py`: runs all cases, prints a table and a summary (tool selection accuracy, ask-year rate,
  confirmation gate pass rate, abstention accuracy, average and p95 latency), saves results to `eval/results_<prompt_version>.json`.
- Run v1, look at failures, write `system_v2.md`, run again, compare. Keep both result files.

## Tests
- `tests/test_tools.py`: unit tests for every tool (no LLM calls), including the superseded-doc filter
  and that `start_return` cannot run without a pending confirmation.

## Repo hygiene
- `.env.example` (MODEL, GROQ_API_KEY), `.gitignore` (.env, logs, chroma db), `requirements.txt`.
- README: what it is (demo, mock data), architecture diagram (Mermaid), how to run, eval results v1 vs v2,
  design decisions, and "what I would add for production" (Azure OpenAI, real order/CRM APIs behind auth,
  webhooks for order events, monitoring dashboards, CSAT from real users, PII handling).
- Optional if time: GitHub Actions workflow running `pytest tests/` on push.

## Working rules for Codex
- Build in small steps; after each step run it and show me the output.
- Explain each file briefly after writing it, in plain language, so I can explain it in an interview.
- Do not add frameworks beyond the stack above (no LangGraph, no LangChain agents).
