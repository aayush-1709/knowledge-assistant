# 📚 Knowledge-Base Assistant

An interactive Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents (PDF, CSV, TXT, etc.) using powerful Large Language Models (LLMs) like OpenAI GPT or Hugging Face Transformers.

## 🚀 What It Does

1. **Upload Documents**  
   Place your PDFs, CSVs, or TXT files in the `docs/` folder.

2. **Generate Embeddings**  
   Converts your documents into vector embeddings using OpenAI’s `text-embedding-3-small` (or Hugging Face models).

3. **Store in a Vector DB**  
   Uses ChromaDB for efficient storage and fast semantic retrieval of document chunks.

4. **Index & Query**  
   LlamaIndex builds a searchable index and queries the most relevant parts of your files.

5. **Answer with Citations**  
   A Streamlit chatbot answers your questions and provides source snippets for transparency.

---

## 🛠 Tech Stack

| Component           | Tool / Library                     | Purpose                              |
|---------------------|------------------------------------|--------------------------------------|
| LLM                 | OpenAI GPT / Hugging Face          | Answers your questions               |
| Embedding Model     | OpenAI / Sentence-Transformers     | Converts docs into vectors           |
| RAG Framework       | LlamaIndex                         | Ingestion, indexing, and querying    |
| Vector Store        | ChromaDB                           | Stores & retrieves vector embeddings |
| Interface           | Streamlit                          | Web-based chatbot UI                 |
| Environment Config  | `dotenv` (`.env`)                  | API key management                   |

---

## 📁 Folder Structure


💡 Example Use Case
Suppose you upload your college syllabus or insurance policy. You can ask:

"What are the core AI subjects in the syllabus?"

"What does the policy say about medical emergencies?"

The assistant finds and explains the answer, showing which page and document it came from.