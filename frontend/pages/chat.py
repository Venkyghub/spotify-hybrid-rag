
import streamlit as st


from frontend.services.backend_client import ask_question

st.set_page_config(
    page_title="Spotify Hybrid RAG",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Spotify Architecture Assistant")
st.caption(
    "Ask questions about the Spotify Web App Architecture "
    "documentation using Hybrid RAG."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(
                        f"File: {source.get('source', 'Unknown')} | "
                        f"Page: {source.get('page', 'Unknown')} | "
                        f"Chunk: {source.get('chunk_id', 'Unknown')}"
                    )

        if message.get("status"):
            st.caption(f"Status: {message['status']}")

# Accept a new question.
question = st.chat_input(
    "Ask about the Spotify Web App architecture..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and generating an answer..."):
            try:
                result = ask_question(question)

                answer = result.get(
                    "answer",
                    "The backend returned no answer.",
                )
                sources = result.get("sources", [])
                status = result.get("status", "unknown")
                grounding = result.get("grounding", {})

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(
                                f"File: {source.get('source', 'Unknown')} | "
                                f"Page: {source.get('page', 'Unknown')} | "
                                f"Chunk: {source.get('chunk_id', 'Unknown')}"
                            )

                st.caption(f"Status: {status}")

                if grounding:
                    st.caption(
                        "Grounding check: "
                        f"{grounding.get('label', 'not reported')}"
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "status": status,
                    }
                )

            except Exception:
                st.error(
                    "Unable to get an answer from the backend. "
                    "Check that FastAPI is running at "
                    "http://127.0.0.1:8000 and review its terminal "
                    "for errors."
                )
