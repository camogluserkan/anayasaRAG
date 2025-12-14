"""
Legal RAG System - Flask Web Application

This is the main entry point for the web interface.
It initializes the Flask application and provides API endpoints for the frontend.

Key Responsibilities:
- Serving the frontend (HTML/JS).
- API for handling user queries (Chat).
- System health checks and status reporting.
"""

import os
import sys
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Configure environment for PyTorch
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.query_engine_ollama import LegalRAGEngineOllama
from config import (
    check_data_exists,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    LEGAL_DATA_DIR
)


# Initialize Flask Application
app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')

# Enable CORS for development flexibility
CORS(app)


# Global variable to hold the RAG engine instance (Singleton pattern)
_rag_engine = None


def get_rag_engine():
    """
    Get or initialize the RAG engine instance.
    Uses lazy loading to avoid long startup times if not needed immediately.
    
    Returns:
        LegalRAGEngineOllama: The active RAG engine instance.
    """
    global _rag_engine
    if _rag_engine is None:
        try:
            print("Initializing RAG Engine...")
            _rag_engine = LegalRAGEngineOllama()
            # Pre-load models to ensure readiness
            _ = _rag_engine.collection
            _ = _rag_engine.embedding_model
            print("RAG Engine ready.")
        except Exception as e:
            print(f"Error initializing RAG engine: {e}")
            raise
    return _rag_engine


# ==================== WEB ROUTES ====================

@app.route('/')
def index():
    """
    Serve the main landing page (Single Page Application).
    """
    return render_template('index.html')


@app.route('/health')
def health():
    """
    Health check endpoint for system monitoring.
    Checks if the RAG system is ready (data indexed, models loaded).
    """
    try:
        rag_ready = check_data_exists()
        if rag_ready:
            # Try to access engine if data exists
            try:
                engine = get_rag_engine()
                rag_ready = engine is not None
            except:
                rag_ready = False
        
        return jsonify({
            'status': 'healthy',
            'rag_ready': rag_ready
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'rag_ready': False,
            'error': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Return information about the active AI models.
    Used by the frontend to display system status.
    """
    try:
        return jsonify({
            'embedding_model': {
                'name': EMBEDDING_MODEL_NAME,
                'display_name': 'TR-MTEB Fine-tuned',
                'model_id': EMBEDDING_MODEL_NAME
            },
            'llm_model': {
                'name': OLLAMA_MODEL_NAME,
                'display_name': 'Meta-Llama-3.1-8B-Instruct',
                'backend': 'Ollama',
                'quantization': 'Q4_K_M'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdfs', methods=['GET'])
def get_pdfs():
    """
    List indexed PDF documents and total chunk count.
    """
    try:
        import chromadb
        
        # List physical PDF files
        pdf_files = list(LEGAL_DATA_DIR.glob("*.pdf"))
        pdf_names = [pdf.name for pdf in pdf_files]
        
        # Get statistics from Vector DB
        total_chunks = 0
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
            total_chunks = collection.count()
        except:
            # Collection might not exist yet
            pass
        
        return jsonify({
            'pdfs': pdf_names,
            'total_chunks': total_chunks
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main Chat Endpoint.
    Receives a user message, processes it through the RAG pipeline, 
    and returns the answer with sources.
    """
    try:
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get the RAG engine
        engine = get_rag_engine()
        
        # Execute Query
        response = engine.query(message)
        
        # Format sources for the frontend
        formatted_sources = []
        for source in response.get('sources', []):
            # Calculate similarity score from distance
            # ChromaDB uses Cosine Distance (0 to 2)
            similarity = source.get('similarity')
            if similarity is None:
                distance = source.get('distance', 0.5)
                # Normalize: 0 distance -> 1.0 similarity
                similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            
            formatted_sources.append({
                'source_file': source.get('source', 'anayasa.pdf'),
                'article': source.get('article_no', ''),
                'page': source.get('page', ''),
                'page_number': source.get('page', ''),
                'preview': source.get('text_preview', ''),
                'score': similarity,
                'similarity_score': similarity
            })
        
        # Calculate overall confidence score
        confidence = None
        if formatted_sources:
            avg_score = sum(s['score'] for s in formatted_sources) / len(formatted_sources)
            confidence = int(avg_score * 100)
        
        # Return structured response
        return jsonify({
            'response': response.get('answer', ''),
            'answer': response.get('answer', ''),
            'sources': formatted_sources,
            'confidence': confidence,
            'has_sources': len(formatted_sources) > 0,
            'low_confidence': confidence is not None and confidence < 50,
            'warning': 'Low confidence score. Try asking a more specific question.' if (confidence is not None and confidence < 50) else None
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'response': f'Error processing request: {str(e)}'
        }), 500


if __name__ == "__main__":
    print("=== Starting Legal RAG Web Server ===")
    
    # Pre-flight check
    if not check_data_exists():
        print("⚠ WARNING: Vector database not found!")
        print("Please run 'python src/indexing.py' to index your documents first.")
    
    # Start Flask
    # Debug mode should be False in production
    app.run(host='0.0.0.0', port=5000, debug=True)
