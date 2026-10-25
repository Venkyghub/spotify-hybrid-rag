import os

import requests
import streamlit as st

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

st.set_page_config(
    page_title="Spotify Hybrid RAG",
    page_icon="🎧",
    layout="wide",
)

st.title("Spotify Architecture — Hybrid RAG")
st.write(
    "Ask questions about the Spotify architecture PDF "
    "using vector search and knowledge graph retrieval."
)

if st.button("Check Backend Connection"):
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        st.success(f"Backend connected: {response.json()}")
    except requests.RequestException as exc:
        st.error(f"Backend connection failed: {exc}")