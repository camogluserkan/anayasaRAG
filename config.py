"""
Turkish Constitutional RAG System - Central Configuration File

This file serves as the centralized configuration hub for the entire application.
It contains:
- Project directory paths (data, models, vector db).
- Model parameters (Embedding and LLM settings).
- Text processing and chunking configurations.
- Vector database (ChromaDB) connection settings.
- Helper functions to validate the environment.
"""

import os
from pathlib import Path


# ==================== PROJECT PATHS ====================
BASE_DIR = Path(__file__).parent.absolute()
LEGAL_DATA_DIR = BASE_DIR / "legal_data"  # Directory for PDF source files
MODELS_DIR = BASE_DIR / "models"          # Directory for local models (if any)
VECTOR_DB_DIR = BASE_DIR / "vector_db"    # Directory for ChromaDB persistence
SRC_DIR = BASE_DIR / "src"                # Source code directory


# ==================== EMBEDDING MODEL ====================
# Using a fine-tuned Turkish embedding model that performs well on TR-MTEB benchmark.
EMBEDDING_MODEL_NAME = "trmteb/turkish-embedding-model-fine-tuned"


# ==================== OLLAMA MODEL CONFIGURATION ====================
# The system now uses Ollama for local LLM inference.
# Ensure Ollama is running and the model is created via `ollama create ...`
OLLAMA_MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434"


# ==================== GENERATION PARAMETERS ====================
# Parameters controlling the LLM output generation.

# Temperature: Lower values (0.1) result in more deterministic/factual answers.
TEMPERATURE = 0.1

# Max New Tokens: Maximum number of tokens the model can generate.
# Increased to allow for comprehensive legal explanations.
MAX_NEW_TOKENS = 768

# Top P / Top K: Sampling parameters for diversity (standard values).
TOP_P = 0.9
TOP_K = 40

# Context Window: Maximum input tokens (model specific).
# Mistral/Llama usually supports larger windows, set to 4096 for balance.
N_CTX = 4096


# ==================== CHUNKING PARAMETERS ====================
# Configuration for splitting PDF text into manageable chunks for embedding.

# Chunk Size: Characters per chunk.
# 1200 chars is roughly one substantial legal article.
CHUNK_SIZE = 1200

# Chunk Overlap: Characters to overlap between chunks to preserve context.
CHUNK_OVERLAP = 200

# Legal Hierarchy Separators (Regex)
# These patterns help split text exactly at article boundaries.
LEGAL_SEPARATORS = [
    # Priority 1: Article Headers (e.g., "MADDE 1-", "Madde 5")
    r"(\nMADDE \d+[-–—])",
    r"(\nMadde \d+[-–—])",
    r"(\nMADDE \d+\s)",
    r"(\nMadde \d+\s)",
    r"(\nGEÇİCİ MADDE \d+)",  # Provisional Articles

    # Priority 2: Sub-structures (e.g., "A. Title", Clause numbers "(1)")
    r"(\n[A-Z]\.\s+)",
    r"(\n\([1-9][0-9]*\)\s)",

    # Priority 3: General Text Separators
    "\n\n",  # Paragraphs
    "\n",    # Lines
    ". ",    # Sentences
    " "      # Words
]


# ==================== CHROMADB SETTINGS ====================
CHROMA_COLLECTION_NAME = "turkish_constitution"
CHROMA_PERSIST_DIRECTORY = str(VECTOR_DB_DIR)


# ==================== RETRIEVAL PARAMETERS ====================
# Settings for the semantic search engine.

# Top K: Number of relevant chunks to retrieve for context.
SIMILARITY_TOP_K = 5

# Retrieval Mode: Strategy for fetching chunks ("similarity" or "mmr").
RETRIEVAL_MODE = "similarity"


# ==================== LOGGING SETTINGS ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ==================== HELPER FUNCTIONS ====================

def check_data_exists() -> bool:
    """
    Check if the Vector Database has been initialized and contains data.

    Returns:
        bool: True if database exists and has files, False otherwise.
    """
    db_path = Path(CHROMA_PERSIST_DIRECTORY)
    return db_path.exists() and any(db_path.iterdir())


def ensure_directories():
    """
    Ensure all required project directories exist.
    Creates them if they are missing.
    """
    LEGAL_DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    SRC_DIR.mkdir(exist_ok=True)
    print(f"✓ Directory structure verified at: {BASE_DIR}")


def check_source_files_exist() -> bool:
    """
    Check if there are any PDF files in the legal_data directory to index.

    Returns:
        bool: True if PDF files are found, False otherwise.
    """
    pdf_files = list(LEGAL_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠ No PDF files found in {LEGAL_DATA_DIR}!")
        return False
    print(f"✓ Found {len(pdf_files)} PDF file(s) ready for indexing.")
    return True


if __name__ == "__main__":
    print("=== Turkish Constitutional RAG System - Configuration Check ===")
    ensure_directories()
    check_source_files_exist()
    if check_data_exists():
        print("✓ Vector Database found.")
    else:
        print("⚠ Vector Database NOT found. Please run 'python src/indexing.py' to initialize.")
