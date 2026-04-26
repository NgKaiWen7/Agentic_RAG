import streamlit as st
from orchrastrator import Orchestrator


@st.cache_resource(show_spinner=True)
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


st.set_page_config(page_title="Agentic RAG Demo", page_icon=":mag:")
st.title("Agentic RAG Demo")
st.caption("Ask questions with Agentic RAG (tool-using retrieval + answering).")

try:
    orchestrator = get_orchestrator()
except Exception as exc:
    st.error(f"Failed to initialize orchestrator: {exc}")
    st.stop()

query = st.text_input("Enter your query", "")

if st.button("Ask (Agentic RAG)", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Running agentic retrieval and generating answer..."):
            result = orchestrator.agentic_rag_query(query)
            references = orchestrator.get_references(query)

        st.subheader("Model Response")
        st.write(result)

        st.subheader("References")
        if references:
            for idx, ref in enumerate(references, start=1):
                title = ref.get("title", "Untitled")
                source = ref.get("source", "")
                if source:
                    st.markdown(f"{idx}. [{title}]({source})")
                else:
                    st.markdown(f"{idx}. {title}")
        else:
            st.write("No references found.")
