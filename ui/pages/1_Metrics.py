"""Monitoring page: reads GET /metrics from the API (the UI never reads log files directly)."""
import os

import pandas as pd
import requests
import streamlit as st

from auth import require_password

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Metrics (demo)", page_icon="📊", layout="wide")
st.title("Metrics")
require_password()
st.caption("From logs/turns.jsonl and logs/feedback.jsonl, grouped by prompt version. "
           "Thumbs-up rate is the CSAT proxy.")

include_eval = st.toggle("Include eval runs", value=False)
try:
    data = requests.get(f"{API_URL}/metrics", params={"include_eval": include_eval}, timeout=10).json()
except requests.RequestException:
    st.error("The API is unreachable. Start it with: uvicorn api.main:app")
    st.stop()

rows = data["by_version"]
if not rows:
    st.info("No turns logged yet. Chat with the agent first.")
    st.stop()

version = st.selectbox("Prompt version", [r["prompt_version"] for r in rows], index=len(rows) - 1)
row = next(r for r in rows if r["prompt_version"] == version)


def fmt(value, suffix=""):
    return "–" if value is None else f"{value}{suffix}"


cols = st.columns(6)
cols[0].metric("Turns", row["turns"], help=f"{row['sessions']} sessions")
cols[1].metric("Thumbs-up rate", fmt(row["thumbs_up_rate"], "%"), help=f"{row['rated_turns']} rated turns")
cols[2].metric("Latency p50 / p95", f"{fmt(row['latency_p50_ms'])} / {fmt(row['latency_p95_ms'])} ms")
cols[3].metric("Avg cost / turn", fmt(row["avg_cost_usd"] and f"${row['avg_cost_usd']:.5f}"))
cols[4].metric("Escalation rate", fmt(row["escalation_rate"], "%"))
cols[5].metric("Fallback rate", fmt(row["fallback_rate"], "%"),
               help="Grounding fallback, escalation fallback, model error or max tool iterations")

st.subheader("By prompt version")
st.dataframe(pd.DataFrame(rows).set_index("prompt_version"), width="stretch")

st.subheader("Tool calls")
if data["tool_usage"]:
    st.bar_chart(pd.Series(data["tool_usage"], name="calls"), horizontal=True)
else:
    st.caption("No tool calls yet.")

st.subheader("Thumbs-down turns")
st.caption("Read these first when deciding the next prompt version.")
if not data["thumbs_down"]:
    st.caption("None yet.")
for t in data["thumbs_down"]:
    with st.expander(f"{t['prompt_version']} · {t['user_message'][:80]}"):
        st.markdown(f"**User:** {t['user_message']}\n\n**Agent:** {t['answer']}")
        st.caption(f"session {t['session_id'][:8]} · turn {t['turn_id']} · flags {t['flags']}")
