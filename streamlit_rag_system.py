# app.py
import streamlit as st
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize LLM and embeddings (only once)
llm = ChatOpenAI(model="gpt-4o")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Define Functions ---
def load_and_store_papers(arxiv_ids):
    """
    Loads multiple research papers from arXiv, splits them into chunks,
    and stores them in a FAISS vectorstore.
    """
    all_documents = []
    for arxiv_id in arxiv_ids:
        st.write(f"📥 Fetching paper {arxiv_id} from arXiv...")
        # Use spinner to show progress
        with st.spinner(f"Loading {arxiv_id}..."):
            docs = ArxivLoader(query=arxiv_id, load_max_docs=1).load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(docs)
            all_documents.extend(split_docs)
    st.success(f"✅ Loaded and split {len(all_documents)} chunks from {len(arxiv_ids)} papers.")

    # Create FAISS vectorstore from the split documents
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

def summarize_paper(retriever, query):
    """
    Queries the retrieval-augmented generation (RAG) system for a given query.
    """
    with st.spinner("Generating summary..."):
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa_chain.run(query)
    return response

# --- Streamlit UI ---
st.set_page_config(page_title="Research Paper Summarizer", layout="wide")
st.title("📚 Research Paper Summarizer (RAG System)")

# Layout: Use columns to separate input sections
col1, col2 = st.columns(2)

with col1:
    st.header("Paper Loader")
    arxiv_input = st.text_input("Enter arXiv paper IDs (comma-separated):", value="1706.03762")
    if st.button("Load Papers"):
        # Process input into a list of IDs
        arxiv_ids = [x.strip() for x in arxiv_input.split(",") if x.strip()]
        # Store retriever in session state
        st.session_state["retriever"] = load_and_store_papers(arxiv_ids)
        st.success("Papers processed and stored!")

with col2:
    st.header("Query")
    if "retriever" not in st.session_state or st.session_state["retriever"] is None:
        st.info("Please load the papers first.")
    else:
        user_query = st.text_input("Ask a question about the papers:")
        if st.button("Get Results") and user_query:
            summary = summarize_paper(st.session_state["retriever"], user_query)
            st.subheader("Summary:")
            st.write(summary)

# Optional: Expandable section to show debug info (e.g., sample chunks)
with st.expander("Show Debug Info"):
    if "retriever" in st.session_state and st.session_state["retriever"] is not None:
        st.write("Retriever is loaded and ready.")
    else:
        st.write("No retriever loaded yet.")
