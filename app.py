import os
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from llama_index.llms.google_genai import GoogleGenAI
import chromadb

load_dotenv()
PERSIST_DIR = "./data/chroma"

def load_query_engine() -> RetrieverQueryEngine:
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_store  = ChromaVectorStore(chroma_collection=chroma_client.get_collection("knowledge") )

    storage_context = StorageContext.from_defaults(persist_dir = PERSIST_DIR, vector_store  = chroma_store,)
    index = load_index_from_storage(storage_context = storage_context, embed_model = embed_model,)

    llm = GoogleGenAI(model="gemini-1.5-flash", temperature=0.1 )

    response_synthesizer = get_response_synthesizer(response_mode="compact", llm = llm,)

    retriever     = index.as_retriever(similarity_top_k=5)
    query_engine  = RetrieverQueryEngine(retriever = retriever, response_synthesizer = response_synthesizer,)
    return query_engine

def main():
    st.title("📄 KnowledgeBase Assistant")
    st.markdown(
        "Ask any question about your documents.<br>",
        unsafe_allow_html=True,
    )

    query_engine = load_query_engine()
    user_query   = st.text_input("Your question:")

    if st.button("Ask") and user_query:
        with st.spinner("Gemini is thinking…"):
            response = query_engine.query(user_query)
            st.markdown("💬 Answer")
            st.write(str(response))

if __name__ == "__main__":
    main()
