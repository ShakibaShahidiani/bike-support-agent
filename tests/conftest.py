"""Tests never send traces to Langfuse, even when .env has keys."""
import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
