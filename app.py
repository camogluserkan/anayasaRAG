"""
Legal RAG System - Flask Web Application

Web interface for Q&A on the Turkish Constitution using RAG.
"""

import os
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Import local modules
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

# Initialize Flask app
app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')
CORS(app)

# Global RAG engine instance (lazy loaded)
_rag_engine = None


def get_rag_engine():
    """Get or initialize RAG engine (singleton pattern)"""
    global _rag_engine
    if _rag_engine is None:
        try:
            _rag_engine = LegalRAGEngineOllama()
            # Pre-load models
            _ = _rag_engine.collection
            _ = _rag_engine.embedding_model
        except Exception as e:
            print(f"Error initializing RAG engine: {e}")
            raise
    return _rag_engine


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        rag_ready = check_data_exists()
        if rag_ready:
            # Try to access engine
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
    """Get model information"""
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
    """Get list of indexed PDFs and chunk count"""
    try:
        import chromadb
        
        # Get PDF files
        pdf_files = list(LEGAL_DATA_DIR.glob("*.pdf"))
        pdf_names = [pdf.name for pdf in pdf_files]
        
        # Get chunk count from ChromaDB
        total_chunks = 0
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
            total_chunks = collection.count()
        except:
            pass
        
        return jsonify({
            'pdfs': pdf_names,
            'total_chunks': total_chunks
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint - process user query"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get RAG engine
        engine = get_rag_engine()
        
        # Process query
        response = engine.query(message)
        
        # Format response for frontend
        sources = []
        for source in response.get('sources', []):
            # Use similarity if available, otherwise calculate from distance
            similarity = source.get('similarity')
            if similarity is None:
                distance = source.get('distance', 0.5)
                # Convert distance to similarity (ChromaDB cosine distance: 0=identical, 2=opposite)
                similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            
            sources.append({
                'source_file': source.get('source', 'anayasa.pdf'),
                'article': source.get('article_no', ''),
                'page': source.get('page', ''),
                'page_number': source.get('page', ''),
                'preview': source.get('text_preview', ''),
                'score': similarity,  # Already 0-1 range
                'similarity_score': similarity
            })
        
        # Calculate confidence (average similarity of top chunks, as percentage)
        confidence = None
        if sources:
            avg_score = sum(s['score'] for s in sources) / len(sources)
            confidence = int(avg_score * 100)  # Convert to percentage (0-100)
        
        return jsonify({
            'response': response.get('answer', ''),
            'answer': response.get('answer', ''),
            'sources': sources,
            'confidence': confidence,
            'has_sources': len(sources) > 0,
            'low_confidence': confidence is not None and confidence < 50,
            'warning': 'Düşük güven skoru. Daha spesifik bir soru deneyebilirsiniz.' if (confidence is not None and confidence < 50) else None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'response': f'Hata: {str(e)}'
        }), 500


if __name__ == "__main__":
    # Check if data exists
    if not check_data_exists():
        print("⚠️  Warning: Vector database not found!")
        print("Please run 'python src/indexing.py' first.")
        print("Starting server anyway...")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
