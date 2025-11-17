# Turkish Constitutional RAG System

A local RAG (Retrieval-Augmented Generation) system built for the Turkish Constitution.

## Features

- 🇹🇷 **Turkish-Specialized Models**: High-accuracy retrieval with TR-MTEB embedding model
- 🏛️ **Legal Text Chunking**: Article and clause-aware chunking strategy
- 💾 **Fully Local**: Entire system runs locally, no internet required after setup
- 🎯 **Source Citations**: Every answer comes with source article references
- ⚡ **CPU/GPU Hybrid**: Optimized for RTX 3050 (4GB VRAM)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Embedding and Indexing

```bash
python src/indexing.py
```

This command:
- Chunks the Constitution PDF
- Computes embeddings using TR-MTEB model
- Saves to ChromaDB vector database

### 3. Download LLM Model

```bash
python download_model.py
```

Downloads Mistral-7B-Instruct GGUF model (4.4GB).

### 4. Run the Application

```bash
python app.py
```

## Usage

```
💬 Question: Milletvekili seçilme yaşı kaçtır?
(What is the minimum age for MP election?)

📝 ANSWER:
The minimum age for election to the Turkish Grand National Assembly is 25.

📚 SOURCE ARTICLES:
[1] anayasa.pdf
    Article: 76
    Page: 45
```

## Project Structure

```
anayasaRAG/
├── src/
│   ├── chunking.py      # Legal text chunking
│   ├── indexing.py      # Vector DB indexing
│   └── query_engine.py  # RAG query engine
├── legal_data/          # PDF files
├── models/              # GGUF models
├── vector_db/           # ChromaDB database
├── config.py            # Configuration
├── app.py               # Main application
├── test_queries.py      # Test suite
└── README.md
```

## Technical Details

### Models

- **Embedding**: `trmteb/turkish-embedding-model-fine-tuned` (TR-MTEB SOTA)
- **LLM**: `Mistral-7B-Instruct-v0.2` Q4_K_M quantized

### Parameters

- **Chunk Size**: 1200 characters
- **Overlap**: 200 characters
- **Top-K Retrieval**: 5 chunks
- **Temperature**: 0.1 (for legal precision)
- **GPU Layers**: 15 (for RTX 3050)

## Testing

Run the test suite to verify system accuracy:

```bash
python test_queries.py
```

This tests the system with known constitutional questions and validates:
- Answer accuracy
- Source article correctness
- Retrieval quality

## Development

### Adding New Documents

1. Place PDF in `legal_data/` directory
2. Run `python src/indexing.py`

### Performance Tuning

Adjust parameters in `config.py`:
- `N_GPU_LAYERS`: Increase/decrease based on VRAM
- `CHUNK_SIZE`: Adjust for document structure
- `SIMILARITY_TOP_K`: More context vs. precision

## Requirements

- Python 3.8+
- 4GB+ VRAM (recommended) or CPU-only mode
- 10GB+ disk space (for models and vector DB)

## License

This project is for educational purposes.

## Contact

Feel free to open an issue for questions or suggestions.
