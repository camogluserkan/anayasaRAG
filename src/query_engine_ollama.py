"""
RAG Query Engine Module (Ollama Version)

Retrieves from ChromaDB and generates answers using Ollama LLM.
"""

import os
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
import ollama

# Config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL,
    TEMPERATURE,
    MAX_NEW_TOKENS,
    SIMILARITY_TOP_K
)


class LegalRAGEngineOllama:
    """
    Legal RAG System - Query Engine (Ollama Version)
    
    Retrieves relevant chunks from ChromaDB and generates answers using Ollama LLM.
    """
    
    def __init__(
        self,
        db_path: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
        ollama_model_name: str = OLLAMA_MODEL_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME
    ):
        """
        Args:
            db_path: ChromaDB path
            collection_name: Collection name
            ollama_model_name: Ollama model name
            embedding_model_name: Embedding model name
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.ollama_model_name = ollama_model_name
        self.embedding_model_name = embedding_model_name
        
        # Lazy loading
        self._embedding_model = None
        self._collection = None
    
    @property
    def embedding_model(self):
        """Load embedding model (lazy)"""
        if self._embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✓ Embedding model ready")
        return self._embedding_model
    
    @property
    def collection(self):
        """Load ChromaDB collection (lazy)"""
        if self._collection is None:
            print(f"Loading ChromaDB: {self.db_path}")
            client = chromadb.PersistentClient(path=self.db_path)
            self._collection = client.get_collection(name=self.collection_name)
            print(f"✓ Collection loaded: {self.collection_name}")
        return self._collection
    
    def retrieve(
        self,
        query: str,
        top_k: int = SIMILARITY_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from ChromaDB
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        # Query expansion for specific questions
        expanded_queries = [query]
        
        if "milletvekili" in query.lower() and "yaş" in query.lower():
            expanded_queries.append("MADDE 76 seçilme yeterliliği")
        elif "cumhurbaşkanı" in query.lower() and "seçim" in query.lower():
            expanded_queries.append("MADDE 101 cumhurbaşkanı seçim süresi")
        
        all_results = []
        seen_ids = set()
        
        for q in expanded_queries:
            # Create embedding
            query_embedding = self.embedding_model.encode(q).tolist()
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Parse results
            if results and results['documents'] and results['documents'][0]:
                docs = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for doc, metadata, distance in zip(docs, metadatas, distances):
                    chunk_id = metadata.get('chunk_id', '')
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        all_results.append({
                            'text': doc,
                            'metadata': metadata,
                            'distance': distance
                        })
        
        # Sort by distance (lower is better)
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:top_k]
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using Ollama LLM
        
        Args:
            query: User query
            context_chunks: Retrieved chunks
            
        Returns:
            Generated answer
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk['text']
            metadata = chunk['metadata']
            article_no = metadata.get('article_no', 'Unknown')
            
            context_parts.append(f"[{i}] ARTICLE {article_no}:\n{text}")
        
        context_str = "\n\n".join(context_parts)
        
        # Llama 3.1 Chat Template Prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sen bir Türk Anayasa Hukuku uzmanısın. Aşağıdaki anayasa maddelerine dayanarak soruyu Türkçe cevapla. Kesin ve net ol, ilgili madde numaralarını belirt.<|eot_id|><|start_header_id|>user<|end_header_id|>

İLGİLİ ANAYASA MADDELERİ:
{context_str}

SORU: {query}

Cevabını sadece verilen maddelerle sınırlı tut. Madde numaralarını belirt (örn: "MADDE 76").<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Generate with Ollama
        print(f"\nGenerating answer with Ollama ({self.ollama_model_name})...")
        
        try:
            response = ollama.generate(
                model=self.ollama_model_name,
                prompt=prompt,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_NEW_TOKENS
                }
            )
            
            answer = response['response'].strip()
            return answer
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(
        self,
        user_query: str,
        top_k: int = SIMILARITY_TOP_K
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: Retrieve + Generate
        
        Args:
            user_query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        print(f"\n{'='*60}")
        print(f"QUERY: {user_query}")
        print(f"{'='*60}")
        
        # 1. Retrieval
        print(f"\n[1/2] Retrieving relevant chunks (top-{top_k})...")
        chunks = self.retrieve(user_query, top_k=top_k)
        
        if not chunks:
            return {
                'answer': 'No relevant information found in the database.',
                'sources': [],
                'query': user_query
            }
        
        print(f"✓ Retrieved {len(chunks)} chunks")
        
        # 2. Generation
        print(f"\n[2/2] Generating answer...")
        answer = self.generate(user_query, chunks)
        print(f"✓ Answer generated")
        
        # Prepare sources
        sources = []
        for chunk in chunks:
            metadata = chunk['metadata']
            distance = chunk.get('distance', 0.5)  # Default distance if not present
            # Convert distance to similarity score (0-1 range)
            # ChromaDB uses cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: similarity = 1 - (distance / 2)
            similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            sources.append({
                'article_no': metadata.get('article_no', 'Unknown'),
                'page': metadata.get('page', 'Unknown'),
                'chunk_id': metadata.get('chunk_id', 'Unknown'),
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'distance': distance,
                'similarity': similarity
            })
        
        return {
            'answer': answer,
            'sources': sources,
            'query': user_query
        }
    
    def print_response(self, response: Dict[str, Any]):
        """
        Print formatted response
        
        Args:
            response: Query response dictionary
        """
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(response['answer'])
        
        print(f"\n{'='*60}")
        print("SOURCES:")
        print(f"{'='*60}")
        for i, source in enumerate(response['sources'], 1):
            print(f"\n[{i}] ARTICLE {source['article_no']} (Page {source['page']})")
            print(f"Chunk ID: {source['chunk_id']}")
            print(f"Preview: {source['text_preview']}")
        
        print(f"\n{'='*60}\n")


def main():
    """Test the query engine"""
    # Initialize engine
    engine = LegalRAGEngineOllama()
    
    # Test query
    test_query = "Türkiye'nin başkenti neresidir?"
    response = engine.query(test_query)
    engine.print_response(response)


if __name__ == "__main__":
    main()

