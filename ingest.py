import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.graph_stores import SimpleGraphStore

import chromadb

load_dotenv()

PERSIST_DIR = "./data/chroma"
DOCS_DIR = "./docs"

os.makedirs(DOCS_DIR, exist_ok=True)

def load_documents():
    return SimpleDirectoryReader(DOCS_DIR).load_data()

def build_index():
    documents = load_documents()

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    # Initialize Chroma client and collection
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_store = ChromaVectorStore(
        chroma_collection=chroma_client.get_or_create_collection("knowledge")
    )

    # Create empty stores for docstore, index_store, graph_store
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()
    graph_store = SimpleGraphStore()

    # Create StorageContext with those stores plus chroma vector store
    storage_context = StorageContext(
        docstore=docstore,
        index_store=index_store,
        graph_store=graph_store,
        vector_stores={"default" : chroma_store}
    )

    # Create the index from documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # Persist the storage context metadata (docstore.json etc.)
    storage_context.persist(persist_dir=PERSIST_DIR)

    print("✅ Index created and persisted!")

if __name__ == "__main__":
    build_index()
