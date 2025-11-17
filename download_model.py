"""
GGUF Model Download Script

Downloads Turkish GGUF model from Hugging Face.
"""

from huggingface_hub import hf_hub_download
from pathlib import Path
from config import GGUF_MODEL_HF_REPO, GGUF_MODEL_FILENAME, MODELS_DIR

def download_gguf_model():
    """Download GGUF model from Hugging Face"""
    
    print("="*70)
    print("GGUF MODEL DOWNLOAD")
    print("="*70)
    print(f"\nRepo: {GGUF_MODEL_HF_REPO}")
    print(f"File: {GGUF_MODEL_FILENAME}")
    print(f"Destination: {MODELS_DIR}")
    print(f"\n⚠ Model is approximately 4.4GB, download may take time!")
    
    try:
        # Download model
        print("\nDownload starting...")
        model_path = hf_hub_download(
            repo_id=GGUF_MODEL_HF_REPO,
            filename=GGUF_MODEL_FILENAME,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"Location: {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Download error: {e}")
        return None

if __name__ == "__main__":
    download_gguf_model()
