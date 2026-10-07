import streamlit as st
from google import genai
from rag import load_pdf, chunk, build_index, search, answer

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A with Citations")
st.caption("Upload a PDF, ask a question, and get an answer with page references.")

access_code = st.secrets.get("ACCESS_CODE", "")
if access_code:
    entered = st.sidebar.text_input("Access code", type="password")
    if entered != access_code:
        st.info("Enter the access code in the sidebar to use the app.")
        st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

file = st.file_uploader("Upload a PDF", type="pdf")

if file:
    if st.session_state.get("file_name") != file.name:
        with st.spinner("Reading and indexing your document..."):
            chunks = chunk(load_pdf(file), size=400, overlap=80)
            if not chunks:
                st.error("No text found. This may be a scanned PDF.")
                st.stop()
            st.session_state.update(
                file_name=file.name, chunks=chunks, index=build_index(chunks)
            )
        st.success(f"Indexed {len(st.session_state.chunks)} chunks.")

    question = st.text_input("Ask a question about the document")
    if question:
        with st.spinner("Thinking..."):
            hits = search(st.session_state.index, st.session_state.chunks, question, k=6)
            st.markdown(answer(client, question, hits))
        with st.expander("Sources used"):
            for h in hits:
                st.caption(f"Page {h['page']}")
                st.write(h["text"])