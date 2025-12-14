"""
RAG Query Engine Module (Ollama Integration)

This module implements the Retrieval-Augmented Generation (RAG) pipeline.
It orchestrates the flow between the Vector Database (ChromaDB) and the Local LLM (Ollama).

Pipeline Stages:
1. Retrieval: Semantic search in ChromaDB to find relevant document chunks.
2. Augmentation: Constructing a prompt with retrieved context.
3. Generation: Using Ollama (Llama 3) to generate a grounded answer.
"""

import os
# Suppress TensorFlow warnings
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import sys
import ollama
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Set

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    TEMPERATURE,
    MAX_NEW_TOKENS,
    SIMILARITY_TOP_K
)


class LegalRAGEngineOllama:
    """
    Core RAG Engine class that handles user queries.
    Uses ChromaDB for retrieval and Ollama for generation.
    """
    
    def __init__(
        self,
        db_path: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
        ollama_model_name: str = OLLAMA_MODEL_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            db_path (str): Path to ChromaDB persistence directory.
            collection_name (str): Name of the ChromaDB collection.
            ollama_model_name (str): Name of the Ollama model to use.
            embedding_model_name (str): HuggingFace embedding model name.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.ollama_model_name = ollama_model_name
        self.embedding_model_name = embedding_model_name
        
        # Lazy loading properties
        self._embedding_model = None
        self._collection = None
    
    @property
    def embedding_model(self):
        """
        Lazy load the embedding model to save resources on startup.
        """
        if self._embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✓ Embedding model ready.")
        return self._embedding_model
    
    @property
    def collection(self):
        """
        Lazy load the ChromaDB collection connection.
        """
        if self._collection is None:
            print(f"Connecting to ChromaDB at: {self.db_path}")
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
        Retrieve relevant document chunks from the vector database.
        
        Args:
            query (str): The user's question.
            top_k (int): Number of chunks to retrieve.
            
        Returns:
            List[Dict]: A list of chunks with their metadata and distance scores.
        """
        # Basic Query Expansion logic for specific legal terms
        # This helps in retrieving more relevant chunks if the user query is vague
        expanded_queries = [query]
        
        lower_query = query.lower()
        if "milletvekili" in lower_query and "yaş" in lower_query:
            expanded_queries.append("MADDE 76 seçilme yeterliliği")
        elif "cumhurbaşkanı" in lower_query and "seçim" in lower_query:
            expanded_queries.append("MADDE 101 cumhurbaşkanı seçim süresi")
        
        all_results = []
        seen_ids: Set[str] = set()
        
        # Execute query for each expanded variation
        for q in expanded_queries:
            # 1. Compute embedding for the query
            query_embedding = self.embedding_model.encode(q).tolist()
            
            # 2. Query the vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 3. Parse and deduplicate results
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
        
        # Sort combined results by distance (lower is better) and slice to top_k
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:top_k]
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate an answer using the LLM based on the retrieved context.
        
        Args:
            query (str): The user's original question.
            context_chunks (List[Dict]): The retrieved legal text chunks.
            
        Returns:
            str: The generated response from the LLM.
        """
        # 1. Format Context
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk['text']
            metadata = chunk['metadata']
            article_no = metadata.get('article_no', 'Unknown')
            
            context_parts.append(f"[{i}] ARTICLE {article_no}:\n{text}")
        
        context_str = "\n\n".join(context_parts)
        
        # 2. Construct Prompt (Llama 3 Chat Template)
        # Note: Keeps the system prompt in Turkish for the model to behave correctly in Turkish context
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sen bir Türk Anayasa Hukuku uzmanısın. Aşağıdaki anayasa maddelerine dayanarak soruyu Türkçe cevapla. Kesin ve net ol, ilgili madde numaralarını belirt. Hukuki olmayan yorumlardan kaçın.<|eot_id|><|start_header_id|>user<|end_header_id|>

İLGİLİ ANAYASA MADDELERİ:
{context_str}

SORU: {query}

Cevabını sadece verilen maddelerle sınırlı tut. Madde numaralarını belirt (örn: "MADDE 76").<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        print(f"\nGenerating answer with Ollama ({self.ollama_model_name})...")
        
        # 3. Call Ollama API
        try:
            response = ollama.generate(
                model=self.ollama_model_name,
                prompt=prompt,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_NEW_TOKENS,
                    "top_p": 0.9
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
        Execute the full RAG pipeline (Retrieve -> Generate).
        
        Args:
            user_query (str): The user's question.
            top_k (int): Number of chunks to use for context.
            
        Returns:
            Dict: Result containing answer, sources, and metadata.
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING QUERY: {user_query}")
        print(f"{'='*60}")
        
        # Step 1: Retrieval
        print(f"\n[1/2] Retrieving relevant context (Top-{top_k})...")
        chunks = self.retrieve(user_query, top_k=top_k)
        
        if not chunks:
            return {
                'answer': 'Üzgünüm, veritabanında bu konuyla ilgili bilgi bulunamadı.',
                'sources': [],
                'query': user_query
            }
        
        print(f"✓ Retrieved {len(chunks)} relevant chunks.")
        
        # Step 2: Generation
        print(f"\n[2/2] Generating answer with LLM...")
        answer = self.generate(user_query, chunks)
        print(f"✓ Answer generated.")
        
        # Step 3: Format Sources
        sources = []
        for chunk in chunks:
            metadata = chunk['metadata']
            distance = chunk.get('distance', 0.5)
            
            # Normalize distance to similarity score (0.0 to 1.0)
            # Chroma uses cosine distance (0=same, 1=orthogonal, 2=opposite)
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
        Helper method to print a formatted response to the console.
        """
        print(f"\n{'='*60}")
        print("GENERATED ANSWER:")
        print(f"{'='*60}")
        print(response['answer'])
        
        print(f"\n{'='*60}")
        print("RETRIEVED SOURCES:")
        print(f"{'='*60}")
        for i, source in enumerate(response['sources'], 1):
            print(f"\n[{i}] ARTICLE {source['article_no']} (Page {source['page']})")
            print(f"Confidence: {source['similarity']:.2%}")
            print(f"Preview: {source['text_preview']}")
        
        print(f"\n{'='*60}\n")


def main():
    """
    Test function to run a sample query directly.
    """
    engine = LegalRAGEngineOllama()
    
    test_query = "Türkiye'nin başkenti neresidir?"
    response = engine.query(test_query)
    engine.print_response(response)


if __name__ == "__main__":
    main()
