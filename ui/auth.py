"""Shared demo password for the deployed UI. Off when DEMO_PASSWORD is unset (local runs)."""
import hmac
import os

import streamlit as st


def require_password() -> None:
    expected = os.getenv("DEMO_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return
    password = st.text_input("Demo password", type="password")
    if password and hmac.compare_digest(password, expected):
        st.session_state.authed = True
        st.rerun()
    if password:
        st.error("Wrong password.")
    st.stop()
