#!/bin/bash
# Quick start script for the RAG system

echo "🚀 Starting Turkish Constitutional RAG System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check if vector database exists
if [ ! -d "vector_db" ] || [ -z "$(ls -A vector_db 2>/dev/null)" ]; then
    echo "⚠️  Vector database not found!"
    echo "📚 Running indexing..."
    python src/indexing.py
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running!"
    echo "Please start Ollama: ollama serve"
    echo ""
fi

# Start Flask application
echo "🌐 Starting web server..."
echo "📍 Open http://localhost:5000 in your browser"
echo ""
python app.py


