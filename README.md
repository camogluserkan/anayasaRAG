# Turkish Constitution RAG System

A Retrieval-Augmented Generation (RAG) application specialized in Turkish Constitutional Law. This system allows users to ask natural language questions about the Turkish Constitution and receive accurate, cited answers based on the legal text.

It uses **Ollama** (Llama 3) for the LLM backend and **ChromaDB** for vector storage, ensuring high performance and data privacy by running locally.

## Features

- **Semantic Search:** Uses a fine-tuned Turkish embedding model (`TR-MTEB`) to find relevant legal articles.
- **Local Inference:** Runs entirely on your machine using Ollama.
- **Accurate Citations:** Provides the exact Article Number and Page Number for every answer.
- **Interactive UI:** Modern, responsive web interface built with Flask and Vanilla JS.
- **Hierarchical Chunking:** Preserves the structural integrity of legal texts (Articles > Clauses).

## Architecture

1.  **Preprocessing:** PDF documents are split into semantic chunks (Articles).
2.  **Indexing:** Chunks are embedded and stored in ChromaDB.
3.  **Retrieval:** User queries are matched against the vector index.
4.  **Generation:** Relevant context + Query is sent to Llama 3 (via Ollama) to generate the response.

## Prerequisites

- **Python 3.10+**
- **Ollama:** Must be installed and running. ([Download Ollama](https://ollama.com/))
- **GPU (Optional):** Recommended for faster inference.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/anayasaRAG.git
    cd anayasaRAG
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Ollama Model:**
    Ensure Ollama is running, then pull the required model (or create from Modelfile):
    ```bash
    # Option 1: Pull directly (if available)
    ollama pull llama3
    
    # Option 2: Create custom model (Recommended)
    ollama create Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M -f Modelfile
    ```

## Usage

### 1. Indexing the Data
Before starting the web server, you must index the legal documents.
```bash
python src/indexing.py
```
This process reads PDFs from `legal_data/`, chunks them, and builds the ChromaDB index.

### 2. Running the Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

## Project Structure

```
anayasaRAG/
├── app.py                  # Main Flask application entry point
├── config.py               # Central configuration settings
├── requirements.txt        # Python dependencies
├── Modelfile               # Ollama model definition
├── legal_data/             # Folder for PDF source files
├── models/                 # Folder for local models (if not using Ollama)
├── vector_db/              # Persistent ChromaDB storage
├── src/
│   ├── chunking.py         # Logic for splitting PDFs into chunks
│   ├── indexing.py         # Logic for creating vector embeddings
│   └── query_engine_ollama.py # RAG pipeline implementation
└── frontend/
    ├── static/             # CSS, JS, and Assets
    └── templates/          # HTML files
```

## Contributing

Contributions are welcome! Please ensure your code follows the project's coding standards (English comments, type hinting, and modular design).

## License

MIT License
