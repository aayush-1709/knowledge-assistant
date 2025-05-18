import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()
PERSIST_DIR = "./data/chroma"

def load_index():
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_store = ChromaVectorStore(chroma_collection=chroma_client.get_collection("knowledge"))
    storage_context = StorageContext.from_defaults(vector_store=chroma_store)
    return load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

def main():
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=5)
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break
        response = query_engine.query(query)
        print("\nAnswer:\n", response, "\n")

if __name__ == "__main__":
    main()
