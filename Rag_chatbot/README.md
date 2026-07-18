# LALLI RAG CHATBOT

A local PDF-based RAG project that ingests PDF documents, extracts text, stores processed page-level JSON, creates embeddings for semantic search, and answers questions through either a Streamlit web app or a CLI chatbot.

## Project Overview

This project is a small end-to-end Retrieval-Augmented Generation (RAG) pipeline with these stages:

1. Ingestion
   - Reads PDF files from the data/pdfs folder
   - Extracts text directly from the PDF
   - Falls back to OCR if the extracted text is too short
   - Saves processed page data as JSON in data/processed

2. Chunking and Embedding
   - Splits document text into chunks
   - Creates embeddings with a sentence-transformers model
   - Stores chunks and metadata in a Qdrant collection

3. Retrieval and Answering
   - Searches the indexed chunks for the most relevant context
   - Uses a local Hugging Face model for answer generation when available
   - Falls back to local text matching if vector search or the model is unavailable

4. Web UI
   - Runs a Streamlit web app for interactive question answering

## Project Structure

- ingestion.py
  - Extracts text from PDFs and creates JSON output files
- chunk_embedding.py
  - Splits processed text into chunks and stores embeddings in Qdrant
- vector_retrieval.py
  - Performs retrieval from processed data or Qdrant
- chatbot.py
  - CLI-based local chatbot using the processed data and local model
- streamlit_app.py
  - Streamlit web UI for the chatbot experience
- data/
  - pdfs/ : input PDF files
  - processed/ : output JSON files from ingestion
  - vector_db/ : local Qdrant storage
- tests/
  - basic project tests

## Models Used

### Embedding model
- Model: sentence-transformers/all-MiniLM-L6-v2
- Purpose: Converts text into vector embeddings for retrieval

### Local generation model
- Model: Qwen/Qwen2.5-0.5B-Instruct
- Purpose: Generates answers from the retrieved context
- This is optional; if it cannot be loaded, the app falls back to a simple context-based answer

### OCR engine
- Tool: Tesseract OCR via pytesseract
- Purpose: Used when the PDF text extraction returns too little content

## Installation

### 1. Create a virtual environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

On Windows, install Tesseract and add it to PATH.
If needed, update this line in ingestion.py:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## How to Run the Project

### Step 1: Put your PDF files in the input folder

Place your PDFs in:
```text
data/pdfs/
```

### Step 2: Run ingestion

```bash
python ingestion.py
```

This will create JSON files in:
```text
data/processed/
```

### Step 3: Create embeddings and populate Qdrant

```bash
python chunk_embedding.py
```

This will:
- read the processed JSON files
- split them into chunks
- create embeddings
- store them in the local Qdrant collection named pdf_collection

### Step 4: Run the CLI chatbot (optional)

```bash
python chatbot.py "what is rag"
```

Or run interactively:
```bash
python chatbot.py
```

### Step 5: Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

If Streamlit is not found, use:
```bash
python -m streamlit run streamlit_app.py
```

## Architecture

This project follows a local, file-first RAG pipeline:

1. Ingestion layer
   - ingestion.py reads PDFs from data/pdfs
   - it extracts text from each page
   - if the page text is too short, it uses OCR with Tesseract
   - each page is saved as a JSON object with text, language, filename, and page number in data/processed

2. Indexing layer
   - chunk_embedding.py loads the processed JSON files
   - it splits the text into smaller chunks
   - each chunk is embedded using sentence-transformers/all-MiniLM-L6-v2
   - the embeddings are stored in a local Qdrant collection named pdf_collection in data/vector_db

3. Retrieval layer
   - chatbot.py and streamlit_app.py both use the same retrieval flow
   - they first try semantic search in Qdrant
   - if the vector index is missing or empty, they fall back to searching the processed JSON files directly
   - vector_retrieval.py is a standalone retrieval helper for testing the same logic

4. Answer generation layer
   - the retrieved chunks are combined into context
   - a local Hugging Face model is used when available for answer generation
   - if the model cannot be loaded, the system falls back to a simple context-based answer

5. User interfaces
   - chatbot.py provides a terminal/CLI experience
   - streamlit_app.py provides a browser-based chat experience

```text
User PDF Files (data/pdfs)
        |
        v
ingestion.py
        |
        v
Processed page JSON (data/processed)
        |
        v
chunk_embedding.py
        |
        v
Qdrant vector store (data/vector_db)
        |
        +----------------------------+
        |                            |
        v                            v
chatbot.py                streamlit_app.py
        |                            |
        +------------> Retrieval + Context Build + LLM Answering <------------+
                                      |
                                      v
                              Final answer + citations
```

## Flow Diagram

```text
1. User uploads or selects PDF files
2. ingestion.py extracts and cleans page text
3. The processed page data is saved as JSON
4. chunk_embedding.py creates embeddings and stores them in Qdrant
5. The user asks a question through chatbot.py or streamlit_app.py
6. The app retrieves relevant chunks from Qdrant or from processed JSON fallback
7. The retrieved context is passed to the local LLM
8. The answer is returned with source citations
```

## Notes

- The project is designed to run locally without needing a paid API key.
- If the local LLM cannot be loaded, the app still works by using the retrieved text directly.
- The retriever always falls back to the processed JSON files if vector search is not available.

## Troubleshooting

### Streamlit not found
```bash
pip install streamlit
```

### OCR not working
- Make sure Tesseract is installed and available in PATH.

### No answer returned
- Make sure your PDFs have been processed with ingestion.py
- Check that data/processed contains JSON files
- If the vector DB is empty, the app will still use the fallback text search
