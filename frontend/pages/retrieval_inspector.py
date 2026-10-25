import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Retrieval Inspector",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Hybrid RAG Retrieval Inspector")
st.write(
    "Inspect the evidence retrieved from FAISS vector search "
    "and the Neo4j knowledge graph."
)

with st.form("retrieval_form"):
    question = st.text_area(
        "Enter a question",
        placeholder="Example: What technologies are used in the Spotify web app architecture?",
        max_chars=2000,
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Vector / final result limit", 1, 20, 5)
    with col2:
        graph_limit = st.slider("Graph retrieval limit", 1, 20, 10)

    submitted = st.form_submit_button("Inspect Retrieval", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Retrieving evidence from FAISS and Neo4j..."):
                response = requests.post(
                    f"{BACKEND_URL.rstrip('/')}/api/retrieval/inspect",
                    json={
                        "question": question.strip(),
                        "top_k": top_k,
                        "graph_limit": graph_limit,
                    },
                    timeout=120,
                )

            if not response.ok:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Retrieval failed ({response.status_code}): {detail}")
            else:
                payload = response.json()
                results = payload.get("results", [])

                st.success(
                    f"Retrieved {payload.get('result_count', len(results))} result(s)."
                )

                if not results:
                    st.warning("No evidence was retrieved for this question.")

                for index, item in enumerate(results, start=1):
                    source = item.get("source", "Unknown source")
                    page = item.get("page", "Unknown")
                    chunk_id = item.get("chunk_id", "Unknown")
                    method = item.get("retrieval_source", "unknown")
                    similarity = item.get("similarity")

                    with st.expander(
                        f"Result {index} · {method.upper()} · {source} · Page {page}",
                        expanded=(index == 1),
                    ):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Retrieval method", str(method).upper())
                        col_b.metric("Page", str(page))
                        col_c.metric(
                            "Similarity",
                            f"{similarity:.4f}"
                            if isinstance(similarity, (int, float))
                            else "N/A",
                        )

                        st.caption(f"Chunk ID: {chunk_id}")

                        text_content = item.get("text") or item.get("evidence")
                        if text_content:
                            st.markdown("**Retrieved text / evidence**")
                            st.write(text_content)
                        else:
                            st.info("This result has no text field.")

                        relationships = item.get("graph_relationships") or []
                        if relationships:
                            st.markdown("**Neo4j graph relationships**")
                            for rel in relationships:
                                st.write(
                                    f"- {rel.get('entity', 'Unknown')} "
                                    f"—[{rel.get('relationship', 'RELATED_TO')}]→ "
                                    f"{rel.get('related_entity', 'Unknown')}"
                                )
                                if rel.get("evidence"):
                                    st.caption(f"Evidence: {rel['evidence']}")

                        if item.get("entity") or item.get("relationship"):
                            st.markdown("**Graph result details**")
                            st.json({
                                "entity": item.get("entity"),
                                "relationship": item.get("relationship"),
                                "related_entity": item.get("related_entity"),
                                "entity_labels": item.get("entity_labels"),
                            })

                        with st.expander("View complete result JSON"):
                            st.json(item)

        except requests.RequestException as exc:
            st.error(
                "Could not contact the backend. Check that the backend "
                "container is healthy."
            )
            st.caption(str(exc))
