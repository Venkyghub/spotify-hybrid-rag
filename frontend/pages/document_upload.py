import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Document Upload",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Upload")
st.write(
    "Upload a PDF to extract its text and generate chunks "
    "for the Spotify Hybrid RAG system."
)

st.info(
    "Current stage: PDF ingestion only. The uploaded document "
    "is not yet added to FAISS or Neo4j."
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    st.write(f"**Selected file:** {uploaded_file.name}")
    st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")

    if uploaded_file.size > 20 * 1024 * 1024:
        st.error("The file exceeds the 20 MB upload limit.")
    elif uploaded_file.size == 0:
        st.error("The selected file is empty.")
    elif st.button("Process PDF", type="primary"):
        try:
            with st.spinner("Uploading and processing PDF..."):
                response = requests.post(
                    f"{BACKEND_URL.rstrip('/')}/api/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=180,
                )

            if response.ok:
                result = response.json()
                st.success("PDF processed successfully!")
                st.metric(
                    "Chunks generated",
                    result.get("chunks_generated", 0),
                )
                st.write(
                    "**Original filename:**",
                    result.get("original_filename", uploaded_file.name),
                )
                st.write(result.get("message", "Processing completed."))
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text

                st.error(f"Upload failed ({response.status_code}): {detail}")

        except requests.RequestException as exc:
            st.error(
                "Could not contact the backend. "
                "Please check the backend container logs."
            )
            st.caption(str(exc))
