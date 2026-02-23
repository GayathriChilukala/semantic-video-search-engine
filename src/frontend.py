import streamlit as st
import requests

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

st.title("Semantic Video Search — Prototype")
q = st.text_input("Search query")
if st.button("Search") and q:
    resp = requests.post(f"{API_URL}/search", json={"q": q, "k": 10})
    if resp.status_code == 200:
        data = resp.json()
        for r in data.get("results", []):
            meta = r.get("meta", {})
            st.write(f"**Score:** {r.get('score'):.4f}")
            st.write(meta.get("text", ""))
            st.write(f"Source: {meta.get('source')}, chunk: {meta.get('chunk_id')}")
            st.markdown("---")
    else:
        st.error(f"Search error: {resp.status_code} - {resp.text}")
