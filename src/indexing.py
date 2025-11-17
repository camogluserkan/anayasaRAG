"""
Vector Database Indexing Module

This module indexes chunked legal texts into ChromaDB vector database.

Features:
- CREATE mode: Create new database from scratch
- UPDATE mode: Add new documents to existing database
- Embedding computation (using TR-MTEB model)
- Persistent storage (stored on disk)
"""

import os
# Disable TensorFlow (for Keras 3 incompatibility)
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import sys
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import time

# Import config and chunking modules
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
    Vector database indexing manager using ChromaDB.
    
    Loads embedding model, vectorizes chunks, and
    saves them persistently to ChromaDB.
    """
    
    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME
    ):
        """
        Args:
            persist_directory: Directory where ChromaDB database will be saved
            collection_name: Database collection name
            embedding_model_name: Embedding model to use
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize ChromaDB client
        print(f"Initializing ChromaDB ({persist_directory})...")
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Prepare embedding model (lazy loading)
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Load embedding model with lazy loading"""
        if self._embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            print("(First load will download the model, may take some time...)")
            
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            
            print("✓ Embedding model loaded")
        
        return self._embedding_model
    
    def collection_exists(self) -> bool:
        """Check if collection exists"""
        collections = self.client.list_collections()
        return any(col.name == self.collection_name for col in collections)
    
    def get_or_create_collection(self, overwrite: bool = False):
        """
        Get or create collection.
        
        Args:
            overwrite: If True, existing collection is deleted and recreated
            
        Returns:
            ChromaDB collection object
        """
        if overwrite and self.collection_exists():
            print(f"⚠ Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
        
        # Create or get collection
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        # Show existing document count
        count = collection.count()
        if count > 0:
            print(f"ℹ Existing collection: {count} documents present")
        else:
            print(f"✓ New collection created: {self.collection_name}")
        
        return collection
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Convert texts to vectors.
        
        Args:
            texts: List of texts to vectorize
            batch_size: Batch processing size
            
        Returns:
            List of embedding vectors
        """
        print(f"Computing embeddings ({len(texts)} chunks)...")
        
        # Compute embeddings with batch processing
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=(i == 0),  # Show progress bar only for first batch
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
            
            if (i + batch_size) % 100 == 0:
                print(f"  Processed: {i + batch_size}/{len(texts)}")
        
        print(f"✓ {len(embeddings)} embeddings computed")
        return embeddings
    
    def index_chunks(
        self,
        chunks: List[Document],
        overwrite: bool = False,
        batch_size: int = 32
    ):
        """
        Index chunks into vector database.
        
        Args:
            chunks: List of chunks to index
            overwrite: Delete existing database and create new one
            batch_size: Batch processing size
        """
        start_time = time.time()
        
        # Prepare collection
        collection = self.get_or_create_collection(overwrite=overwrite)
        
        # Get existing document count (for ID offset)
        existing_count = collection.count()
        
        # Compute embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        # Prepare metadata and IDs
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{existing_count + i}"
            ids.append(chunk_id)
            
            # Convert metadata to strings (ChromaDB requirement)
            # Note: Keep field name as "article_no" even though it was "madde_no" in Turkish
            # This maintains consistency across the codebase
            metadata = {
                "source": str(chunk.metadata.get("source", "")),
                "page": str(chunk.metadata.get("page", "")),
                "article_no": str(chunk.metadata.get("article_no", "")),
                "chunk_id": str(chunk.metadata.get("chunk_id", i))
            }
            metadatas.append(metadata)
        
        # Load into ChromaDB
        print(f"\nLoading into ChromaDB ({len(chunks)} chunks)...")
        
        # Add in batches
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            
            if (i + batch_size) % 100 == 0:
                print(f"  Loaded: {i + batch_size}/{len(chunks)}")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("✓ INDEXING COMPLETED")
        print(f"{'='*70}")
        print(f"Total Chunks: {len(chunks)}")
        print(f"Database Path: {self.persist_directory}")
        print(f"Collection Name: {self.collection_name}")
        print(f"Total Time: {elapsed_time:.2f} seconds")
        print(f"{'='*70}\n")


def main():
    """
    Main indexing process - CREATE mode
    """
    print("="*70)
    print("LEGAL TEXT VECTORIZATION AND INDEXING")
    print("="*70)
    
    # 1. Create chunks using chunking module
    print("\n[STEP 1/3] Loading and chunking documents...")
    chunker = LegalChunker()
    
    try:
        documents = chunker.load_documents_from_directory(LEGAL_DATA_DIR)
        chunks = chunker.chunk_documents(documents)
        
        print(f"\n✓ {len(chunks)} chunks ready")
        
    except Exception as e:
        print(f"\n❌ Chunking error: {e}")
        return
    
    # 2. Initialize indexer
    print(f"\n[STEP 2/3] Preparing vector database...")
    indexer = VectorDBIndexer()
    
    # 3. Indexing
    print(f"\n[STEP 3/3] Vectorizing and indexing chunks...")
    print("⚠ This process may take time depending on embedding model and chunk count!")
    
    try:
        # Offer option to user
        print("\nDo you want to delete existing database and create new one?")
        print("  [1] Yes - Create new database from scratch (OVERWRITE)")
        print("  [2] No - Add to existing database (UPDATE)")
        
        # Use overwrite=True for demo
        overwrite = True  # Always create new on first run
        
        indexer.index_chunks(chunks, overwrite=overwrite)
        
        print("\n✅ All operations completed successfully!")
        print("RAG system can now use this database to answer questions.")
        
    except Exception as e:
        print(f"\n❌ Indexing error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
