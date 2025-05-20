# 📄 Knowledge‑Base Assistant
# Insight AI

A **local‑first Retrieval‑Augmented Generation (RAG) chatbot** that lets you ask natural‑language questions about any set of documents.
It combines fast **MiniLM sentence embeddings**, a **persistent Chroma vector store**, and **Google Gemini** to craft concise answers, all wrapped in a Streamlit UI and a simple CLI.

---

## 🏗️ Tech Stack

* **Python 3.9 +**
* **LlamaIndex** - indexing, retrieval, RAG utilities
* **ChromaDB** – local vector database
* **Sentence‑Transformers** (`all‑MiniLM‑L6‑v2`) – embeddings
* **Google Gemini** via `llama-index-llms-google-genai`
* **Streamlit** – web UI

---

## 📂 Project Layout

```
.
├── app.py            # Streamlit UI
├── query.py          # Command‑line Q&A
├── ingest.py         # One command doc ingestion
├── docs/             # ← put your source documents here
├── data/
│   └── chroma/       # auto‑generated Chroma files (vectors + metadata)
├── .env              # holds GOOGLE_API_KEY
└── requirements.txt  # pip dependencies
```

---

## 🚀 Quick Start

1. **Clone & set up env**

   ```
   git clone https://github.com/aayush-1709/knowledge-assistant
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add your Gemini key**

   ```bash
    GEMINI_API_KEY="your-key" (Create through https://aistudio.google.com/)
   ```

3. **Ingest documents**

   ```bash
   # copy your PDFs, TXTs, etc. into ./docs first
   python ingest.py
   ```

4. **Ask questions**

   * **CLI**

     ```bash
     python query.py
     ```
   * **Streamlit UI**

     ```bash
     streamlit run app.py
     ```

---

## 🙏 Acknowledgements
* [LlamaIndex](https://github.com/run-llama/llama_index) community for the RAG framework.
* [Google AI Studio](https://ai.google.dev/) for access to Gemini.
