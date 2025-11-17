"""
RAG Query Engine Module

Retrieves from ChromaDB and generates answers using LLM.
"""

import os
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from llama_cpp import Llama

# Config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GGUF_MODEL_PATH,
    N_GPU_LAYERS,
    N_CTX,
    TEMPERATURE,
    MAX_NEW_TOKENS,
    SIMILARITY_TOP_K,
    TOP_P,
    TOP_K
)


class LegalRAGEngine:
    """
    Legal RAG System - Query Engine
    
    Retrieves relevant chunks from ChromaDB and generates answers using LLM.
    """
    
    def __init__(
        self,
        db_path: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
        model_path: Path = GGUF_MODEL_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME
    ):
        """
        Args:
            db_path: ChromaDB path
            collection_name: Collection name
            model_path: GGUF model path
            embedding_model_name: Embedding model name
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name
        
        # Lazy loading
        self._llm = None
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
            self._collection = client.get_collection(self.collection_name)
            count = self._collection.count()
            print(f"✓ ChromaDB ready ({count} chunks)")
        return self._collection
    
    @property
    def llm(self):
        """Load LLM (lazy)"""
        if self._llm is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            print(f"Loading LLM: {self.model_path.name}")
            print(f"(GPU layers: {N_GPU_LAYERS}, Context: {N_CTX})")
            
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                verbose=False
            )
            print("✓ LLM ready")
        return self._llm
    
    def retrieve(self, query: str, top_k: int = SIMILARITY_TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve closest chunks to query (improved)
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of chunks (text, metadata, score)
        """
        # Query expansion: Express question in different ways
        expanded_queries = [query]
        
        # Keyword extraction - important terms in question
        if "yaş" in query.lower() and "seçil" in query.lower():  # Turkish: age and elect
            expanded_queries.append("milletvekili seçilme şartları")  # MP election requirements
            expanded_queries.append("yaş şartı seçim")  # age requirement election
        
        if "cumhurbaşkanı" in query.lower() and "seçil" in query.lower():  # Turkish: president and elect
            expanded_queries.append("cumhurbaşkanı seçim usulü")  # presidential election procedure
            expanded_queries.append("cumhurbaşkanı nasıl belirlenir")  # how president is determined
        
        # Get embeddings for each query and combine results
        all_chunks = {}
        
        for q in expanded_queries:
            query_embedding = self.embedding_model.encode([q])[0].tolist()
            
            # Search in ChromaDB (get more results)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2  # Get more results, then filter
            )
            
            # Format and combine results
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 1.0
                
                # If this chunk was found before, keep best score
                if chunk_id not in all_chunks or distance < all_chunks[chunk_id]['distance']:
                    all_chunks[chunk_id] = {
                        'id': chunk_id,
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': distance,
                        'similarity': 1 - distance  # Similarity score
                    }
        
        # Sort by score and take top_k
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x['distance'])
        return sorted_chunks[:top_k]
    
    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using context and query (improved prompt)
        
        Args:
            query: User question
            context_chunks: Retrieved chunks
            
        Returns:
            Response dict (answer, sources)
        """
        # Combine context - highlight article numbers
        context_texts = []
        for i, chunk in enumerate(context_chunks):
            source = chunk['metadata'].get('source', 'Unknown')
            article = chunk['metadata'].get('article_no', '')
            page = chunk['metadata'].get('page', '')
            similarity = chunk.get('similarity', 0)
            
            if article:
                header = f"[ARTICLE {article}] (Similarity: {similarity:.2f})"
            else:
                header = f"[SOURCE {i+1}] (Similarity: {similarity:.2f})"
            
            context_texts.append(f"{header}\n{chunk['text']}")
        
        context_str = "\n\n" + "="*70 + "\n\n".join([""] + context_texts)
        
        # Improved prompt - Mistral Instruct format
        prompt = f"""<s>[INST] You are a Turkish Constitutional Law expert. Answer the question in Turkish based on the constitutional articles below. Be precise and clear, cite relevant article numbers.

RELEVANT CONSTITUTIONAL ARTICLES:
{context_str}

QUESTION: {query}

Limit your answer to the provided articles only. Cite article numbers (e.g., "MADDE 76"). [/INST]"""
        
        # Get answer from LLM
        response = self.llm(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            stop=["</s>", "[INST]", "\n\nQUESTION:"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        
        return {
            'answer': answer,
            'sources': context_chunks,
            'prompt_tokens': response['usage']['prompt_tokens'],
            'completion_tokens': response['usage']['completion_tokens']
        }
    
    def query(self, question: str, top_k: int = SIMILARITY_TOP_K) -> Dict[str, Any]:
        """
        End-to-end RAG query
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            Response dict
        """
        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print(f"{'='*70}")
        
        # 1. Retrieve
        print(f"\n[1/2] Searching for relevant articles...")
        chunks = self.retrieve(question, top_k=top_k)
        print(f"✓ {len(chunks)} articles found")
        
        # 2. Generate
        print(f"\n[2/2] Generating answer...")
        response = self.generate(question, chunks)
        print(f"✓ Answer ready ({response['completion_tokens']} tokens)")
        
        return response
    
    def print_response(self, response: Dict[str, Any]):
        """Print response in nice format"""
        print(f"\n{'='*70}")
        print("ANSWER:")
        print(f"{'='*70}")
        print(response['answer'])
        
        print(f"\n{'='*70}")
        print("SOURCES:")
        print(f"{'='*70}")
        
        for i, chunk in enumerate(response['sources']):
            source = chunk['metadata'].get('source', 'Unknown')
            article = chunk['metadata'].get('article_no', 'None')
            page = chunk['metadata'].get('page', 'N/A')
            distance = chunk.get('distance')
            
            print(f"\n[Source {i+1}]")
            print(f"  File: {source}")
            print(f"  Article: {article}")
            print(f"  Page: {page}")
            if distance is not None:
                print(f"  Similarity: {1 - distance:.3f}")
            print(f"  Text (first 150 chars): {chunk['text'][:150]}...")


def main():
    """Example usage for testing"""
    print("="*70)
    print("LEGAL RAG SYSTEM - QUERY ENGINE TEST")
    print("="*70)
    
    # Start engine
    engine = LegalRAGEngine()
    
    # Test questions
    test_questions = [
        "Milletvekili seçilme yaşı kaçtır?",  # What is the age for MP election?
        "Cumhurbaşkanı nasıl seçilir?",  # How is the president elected?
        "Türkiye Cumhuriyeti'nin başkenti neresidir?"  # What is the capital of Turkey?
    ]
    
    for question in test_questions:
        response = engine.query(question)
        engine.print_response(response)
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
