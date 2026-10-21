"""Central settings. Everything that may change between environments comes from env vars."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
PROMPTS_DIR = ROOT / "prompts"
LOGS_DIR = ROOT / "logs"
CHROMA_DIR = ROOT / "chroma_db"

MODEL = os.getenv("MODEL", "openai/gpt-6-luna")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3")

MAX_TOOL_ITERATIONS = 5
MAX_INPUT_CHARS = 2000
MODEL_TIMEOUT_S = 30
PENDING_MAX_TURNS = 2  # a pending return survives at most 2 follow-up turns

# Must match the current returns policy doc (data/docs/returns_policy_2026.md, "30 days").
RETURN_WINDOW_DAYS = 30

# USD per token, used only when LiteLLM does not know the model (e.g. an Azure deployment name).
FALLBACK_PRICES = {
    "gpt-6-luna": {"input": 0.10 / 1_000_000, "output": 0.50 / 1_000_000},
}
