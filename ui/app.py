"""Streamlit chat UI. Talks to the API only (no agent imports). Run: streamlit run ui/app.py"""
import os
import uuid

import requests
import streamlit as st

from auth import require_password

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ferro Cycles support (demo)", page_icon="🚲")
st.title("Ferro Cycles support")
st.caption("Demo with fictional brand, bikes, orders and policies.")
require_password()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.turns = []  # list of {"role", "content", "turn_id", "meta"}


def send_feedback(turn_id: int):
    value = st.session_state[f"fb_{turn_id}"]  # 1 = thumbs up, 0 = thumbs down
    requests.post(f"{API_URL}/feedback", timeout=10, json={
        "session_id": st.session_state.session_id, "turn_id": turn_id, "rating": "up" if value == 1 else "down"})


def render(turn: dict):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("turn_id"):
            st.feedback("thumbs", key=f"fb_{turn['turn_id']}", on_change=send_feedback, args=(turn["turn_id"],))
            meta = turn["meta"]
            with st.expander("details"):
                st.write({"tools": [t["name"] for t in meta["tools_called"]], "flags": meta["flags"],
                          "latency_ms": meta["latency_ms"]})


for turn in st.session_state.turns:
    render(turn)

if prompt := st.chat_input("Ask about your bike, an order or a return"):
    user_turn = {"role": "user", "content": prompt}
    st.session_state.turns.append(user_turn)
    render(user_turn)
    try:
        r = requests.post(f"{API_URL}/chat", timeout=60,
                          json={"session_id": st.session_state.session_id, "message": prompt})
        r.raise_for_status()
        data = r.json()
        reply = {"role": "assistant", "content": data["answer"], "turn_id": data["turn_id"], "meta": data}
    except requests.RequestException:
        reply = {"role": "assistant", "content": "The support service is unreachable right now. Please try again later."}
    st.session_state.turns.append(reply)
    render(reply)
