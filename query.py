import os
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.google_genai import GoogleGenAI
import chromadb

load_dotenv()
PERSIST_DIR = "./data/chroma"

def load_index():
    embed_model   = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_store  = ChromaVectorStore(
        chroma_collection=chroma_client.get_collection("knowledge")
    )
    storage_ctx = StorageContext.from_defaults(
        persist_dir  = PERSIST_DIR,
        vector_store = chroma_store,
    )
    return load_index_from_storage(storage_context=storage_ctx,
                                   embed_model=embed_model)

def build_query_engine(index):
    llm = GoogleGenAI(model="gemini-1.5-flash", temperature=0.1)
    synth = get_response_synthesizer(response_mode="compact", llm=llm)
    return index.as_query_engine(similarity_top_k=5,response_synthesizer=synth)

def main():
    index        = load_index()
    query_engine = build_query_engine(index)

    while True:
        q = input("Ask a question (type 'exit' to quit): ")
        if q.lower() in {"exit", "quit"}:
            break
        resp = query_engine.query(q)
        print("\nAnswer:\n", resp, "\n")

if __name__ == "__main__":
    main()
