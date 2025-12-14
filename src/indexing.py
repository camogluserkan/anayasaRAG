"""
Vector Database Indexing Module

This module is responsible for the vectorization and storage of document chunks.
It bridges the gap between raw text chunks and the retrieval system.

Features:
- Embedding Generation: Converts text to vector embeddings using HuggingFace models.
- Persistence: Stores vectors and metadata in ChromaDB for fast retrieval.
- Batch Processing: Handles large datasets efficiently.
"""

import os
# Suppress TensorFlow warnings (as we use PyTorch)
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import sys
import time
import chromadb
from pathlib import Path
from typing import List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    LEGAL_DATA_DIR
)
from src.chunking import LegalChunker, Document


class VectorDBIndexer:
    """
    Manages the lifecycle of the Vector Database (ChromaDB).
    Handles embedding computation and document insertion.
    """
    
    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME
    ):
        """
        Initialize the Indexer.
        
        Args:
            persist_directory (str): Path to store the database files.
            collection_name (str): Name of the ChromaDB collection.
            embedding_model_name (str): HuggingFace model ID for embeddings.
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        print(f"Initializing ChromaDB Client at: {persist_directory}")
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Lazy loading for the embedding model
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """
        Property to access the embedding model. Loads it on first access.
        """
        if self._embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            print("(First time load might verify/download model files...)")
            
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            
            print("✓ Embedding model loaded successfully.")
        
        return self._embedding_model
    
    def collection_exists(self) -> bool:
        """
        Check if the target collection already exists in the DB.
        """
        collections = self.client.list_collections()
        return any(col.name == self.collection_name for col in collections)
    
    def get_or_create_collection(self, overwrite: bool = False):
        """
        Retrieve existing collection or create a new one.
        
        Args:
            overwrite (bool): If True, deletes existing collection before creating.
            
        Returns:
            Collection: The ChromaDB collection object.
        """
        if overwrite and self.collection_exists():
            print(f"⚠ Overwrite mode: Deleting existing collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
        
        # Create or get collection with Cosine Similarity space
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        count = collection.count()
        if count > 0:
            print(f"ℹ Collection loaded with {count} existing documents.")
        else:
            print(f"✓ New collection created: {self.collection_name}")
        
        return collection
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Compute embeddings for a list of texts using batch processing.
        
        Args:
            texts (List[str]): Input texts.
            batch_size (int): Number of texts to process at once.
            
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        print(f"Computing embeddings for {len(texts)} chunks...")
        
        embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            # Encode batch
            # show_progress_bar=True helps visualize progress for large datasets
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=(i == 0), 
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
            
            if (i + batch_size) % 100 == 0:
                print(f"  Processed: {min(i + batch_size, total)}/{total}")
        
        print(f"✓ Computed {len(embeddings)} vectors.")
        return embeddings
    
    def index_chunks(
        self,
        chunks: List[Document],
        overwrite: bool = False,
        batch_size: int = 32
    ):
        """
        Main method to index document chunks into ChromaDB.
        
        Args:
            chunks (List[Document]): Processed document chunks.
            overwrite (bool): Whether to rebuild the index.
            batch_size (int): Batch size for embedding and insertion.
        """
        start_time = time.time()
        
        # 1. Prepare Collection
        collection = self.get_or_create_collection(overwrite=overwrite)
        existing_count = collection.count()
        
        # 2. Extract Texts
        texts = [chunk.page_content for chunk in chunks]
        
        # 3. Compute Embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        # 4. Prepare Metadata and IDs
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = f"chunk_{existing_count + i}"
            ids.append(chunk_id)
            
            # Prepare metadata (must be primitives for ChromaDB)
            meta = {
                "source": str(chunk.metadata.get("source", "")),
                "page": str(chunk.metadata.get("page", "")),
                "article_no": str(chunk.metadata.get("article_no", "")),
                "original_chunk_id": str(chunk.metadata.get("chunk_id", i))
            }
            metadatas.append(meta)
        
        # 5. Insert into Database
        print(f"\nInserting {len(chunks)} documents into ChromaDB...")
        
        for i in range(0, len(chunks), batch_size):
            end_idx = i + batch_size
            collection.add(
                ids=ids[i:end_idx],
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            if end_idx % 100 == 0:
                print(f"  Inserted: {min(end_idx, len(chunks))}/{len(chunks)}")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("INDEXING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Chunks Indexed: {len(chunks)}")
        print(f"Database Path: {self.persist_directory}")
        print(f"Collection: {self.collection_name}")
        print(f"Time Taken: {elapsed:.2f} seconds")
        print(f"{'='*70}\n")


def main():
    """
    Main entry point for indexing script.
    """
    print("="*70)
    print("RAG INDEXING PIPELINE")
    print("="*70)
    
    # Step 1: Chunking
    print("\n[STEP 1/3] Loading and Chunking Documents...")
    chunker = LegalChunker()
    try:
        docs = chunker.load_documents_from_directory(LEGAL_DATA_DIR)
        chunks = chunker.chunk_documents(docs)
        print(f"✓ {len(chunks)} chunks prepared.")
    except Exception as e:
        print(f"❌ Error during chunking: {e}")
        return
    
    # Step 2: Init Indexer
    print(f"\n[STEP 2/3] Initializing Vector Database...")
    indexer = VectorDBIndexer()
    
    # Step 3: Indexing
    print(f"\n[STEP 3/3] Vectorizing and Indexing...")
    try:
        # Default behavior: Overwrite on run to ensure fresh index
        indexer.index_chunks(chunks, overwrite=True)
        print("\n✅ Indexing complete! The system is ready for queries.")
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
