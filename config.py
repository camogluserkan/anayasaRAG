"""
Turkish Constitutional RAG System - Central Configuration File

This file contains all model parameters, paths, and settings.
"""

import os
from pathlib import Path


def check_data_exists() -> bool:
    """Check if ChromaDB database exists"""
    db_path = Path(CHROMA_PERSIST_DIRECTORY)
    return db_path.exists() and any(db_path.iterdir())


# ==================== PROJECT PATHS ====================
BASE_DIR = Path(__file__).parent.absolute()
LEGAL_DATA_DIR = BASE_DIR / "legal_data"
MODELS_DIR = BASE_DIR / "models"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
SRC_DIR = BASE_DIR / "src"

# ==================== EMBEDDING MODEL ====================
# Turkish embedding model with SOTA performance on TR-MTEB Benchmark
EMBEDDING_MODEL_NAME = "trmteb/turkish-embedding-model-fine-tuned"

# Alternative models (as backup):
# EMBEDDING_MODEL_NAME = "trmteb/bert-base-turkish-uncased-cachedmnrl-contrastive-loss"
# EMBEDDING_MODEL_NAME = "multilingual-e5-large-instruct"

# ==================== GGUF LLM MODEL ====================
# Model to download from Hugging Face
# Qwen2.5-3B was tested but has issues with ChatML format (repeating outputs)
# Mistral 7B Instruct v0.2 works more stable
# NOTE: Commented out because using Ollama model instead
#GGUF_MODEL_HF_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
#GGUF_MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
#GGUF_MODEL_PATH = MODELS_DIR / GGUF_MODEL_FILENAME

# Note: Mistral 7B supports Turkish and provides more reliable results
# Qwen2.5-3B alternative: bartowski/Qwen2.5-3B-GGUF / Qwen2.5-3B-Q4_K_M.gguf

# ==================== OLLAMA MODEL ====================
# Alternative LLM backend using Ollama (faster, easier setup)
# Model imported from: hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M
# To create: ollama create Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M -f Modelfile
OLLAMA_MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434"

# ==================== LLM PARAMETERS ====================
# GPU/CPU Hybrid Execution Parameters
# Starting value for RTX 3050 (4GB VRAM)
# If you get CUDA out of memory errors, reduce: 15 -> 10 -> 5
N_GPU_LAYERS = 15  # Optimal for Mistral 7B (Qwen2.5 tested: 20)

# Context Window (Mistral 7B supports 32K, but limiting for memory)
N_CTX = 4096  # For stable performance (tested: 8192)

# Generation Parameters
TEMPERATURE = 0.1  # Low for legal precision (deterministic answers)
MAX_NEW_TOKENS = 768  # Increased for longer answers (old: 512)
TOP_P = 0.9
TOP_K = 40

# ==================== CHUNKING PARAMETERS ====================
# Text splitting settings (IMPROVED)
# Reduced to ensure each article becomes a separate chunk
CHUNK_SIZE = 1200  # Medium size: most articles fit but not too small
CHUNK_OVERLAP = 200  # Context preservation at article boundaries

# Special Regex separators for legal hierarchy
# ARTICLE regex patterns have HIGHEST priority - each article should be a separate chunk
LEGAL_SEPARATORS = [
    # Priority 1: ARTICLE headers (MOST IMPORTANT)
    r"(\nMADDE \d+[-–—])",         # "MADDE 76-" or "MADDE 76 -"
    r"(\nMadde \d+[-–—])",         # "Madde 76-" lowercase
    r"(\nMADDE \d+\s)",            # "MADDE 76 " (with space)
    r"(\nMadde \d+\s)",            # "Madde 76 " (with space)
    r"(\nGEÇİCİ MADDE \d+)",       # "GEÇİCİ MADDE 3" (Temporary Article)
    
    # Priority 2: Substructures
    r"(\n[A-Z]\.\s+)",             # Subheadings like "A. Foundation"
    r"(\n\([1-9][0-9]*\)\s)",      # Clause numbers: "(1) ", "(12) "
    
    # Priority 3: General separators
    "\n\n",                        # Paragraph endings
    "\n",                          # Line endings
    ". ",                          # Sentence endings
    " "                            # Space (last resort)
]

# ==================== CHROMADB SETTINGS ====================
CHROMA_COLLECTION_NAME = "turkish_constitution"
CHROMA_PERSIST_DIRECTORY = str(VECTOR_DB_DIR)

# ==================== RETRIEVAL PARAMETERS ====================
# Number of most similar chunks to retrieve during query
SIMILARITY_TOP_K = 5  # Increased for more context

# Retrieval strategy
RETRIEVAL_MODE = "similarity"  # Alternatives: "mmr", "similarity_score_threshold"

# MMR (Maximal Marginal Relevance) parameters
MMR_DIVERSITY_BIAS = 0.3  # 0=similarity only, 1=diversity only

# ==================== RAGAS EVALUATION SETTINGS ====================
# Test dataset path (to be created)
EVAL_DATASET_PATH = BASE_DIR / "eval_dataset.json"

# Evaluation metrics
EVAL_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy"
]

# ==================== LLAMA-CPP-PYTHON INSTALLATION ====================
"""
CUDA-enabled llama-cpp-python Installation:

Windows PowerShell:
```powershell
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
$env:FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

Linux/Mac:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

Note: CUDA Toolkit must be installed on your system.
Check NVIDIA Driver and CUDA version compatibility.
"""

# ==================== LOGGING SETTINGS ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== HELPER FUNCTIONS ====================
def ensure_directories():
    """Check existence of required directories, create if missing."""
    LEGAL_DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    SRC_DIR.mkdir(exist_ok=True)
    print(f"✓ Directory structure ready: {BASE_DIR}")

def check_model_exists():
    """Check if GGUF model has been downloaded (not used when using Ollama)."""
    # When using Ollama, this check is not needed
    print("✓ Using Ollama model (GGUF check skipped)")
    return True

def check_data_exists():
    """Check existence of legal document files."""
    pdf_files = list(LEGAL_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠ No PDF files found in {LEGAL_DATA_DIR}!")
        return False
    print(f"✓ Found {len(pdf_files)} PDF file(s)")
    return True

if __name__ == "__main__":
    print("=== Turkish Constitutional RAG System - Configuration Check ===")
    ensure_directories()
    check_data_exists()
    check_model_exists()
