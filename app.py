import streamlit as st
from orchrastrator import agentic_rag_query

st.title("Agentic RAG Demo")

st.markdown("""
This is a demonstration of an Agentic RAG system that can retrieve information from local documents and web sources.
The agent uses reasoning to decide whether to search local docs, web, or both.
""")

query = st.text_input("Enter your query:", "Latest trends in badminton analytics")

if st.button("Ask"):
    with st.spinner("Processing..."):
        answer = agentic_rag_query(query)
    st.write("**Answer:**")
    st.write(answer)

st.markdown("""
### How it works:
1. The agent analyzes the query
2. Decides which tools to use (LocalSearch, WebSearch)
3. Retrieves relevant information
4. Generates a coherent answer

### Differences from Traditional RAG:
- **Traditional RAG**: Direct retrieval based on similarity
- **Agentic RAG**: Agent reasons about what to retrieve, can use multiple sources, refine queries
""")