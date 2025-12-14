"""
Legal Text Chunking Module

This module handles the preprocessing and segmentation of legal PDF documents into semantic chunks.
It is designed to preserve the hierarchical structure of legal texts (Articles, Clauses) 
to ensure that the retrieval system finds relevant legal contexts.

Key Features:
- Recursive text splitting with regex support for custom separators.
- Metadata enrichment (extracting article numbers, page numbers).
- Handling of Turkish legal text specific patterns (e.g., "Madde 1", "GEÇİCİ MADDE").
"""

import re
import sys
import pypdf
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LEGAL_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LEGAL_SEPARATORS
)


class Document:
    """
    A unified data class to represent a document or a text chunk.
    Similar to LangChain's Document class but lightweight.
    
    Attributes:
        page_content (str): The actual text content.
        metadata (Dict[str, Any]): Dictionary containing metadata like page number, source, article no.
    """
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


class SimpleRecursiveSplitter:
    """
    A custom implementation of recursive text splitting logic.
    It splits text based on a list of separators, trying to keep chunks under a specific size.
    It prioritizes high-level separators (like Article headers) over low-level ones (newlines).
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str]):
        """
        Initialize the splitter.

        Args:
            chunk_size (int): Maximum size of a single chunk (in characters).
            chunk_overlap (int): Number of characters to overlap between chunks.
            separators (List[str]): List of regex strings or characters to split by.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """
        Public method to split a long string into chunks.
        
        Args:
            text (str): The input text to split.
            
        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        self._recursive_split(text, 0, chunks)
        return chunks
    
    def _recursive_split(self, text: str, sep_index: int, chunks: List[str]):
        """
        Internal recursive method to process the splitting logic.
        
        Args:
            text (str): Text segment to process.
            sep_index (int): Index of the current separator in the separators list.
            chunks (List[str]): Accumulator list for results.
        """
        # Base case: If text fits in chunk_size or we ran out of separators
        if len(text) <= self.chunk_size or sep_index >= len(self.separators):
            if text.strip():
                chunks.append(text)
            return
        
        separator = self.separators[sep_index]
        
        # Split using Regex or String split
        try:
            # Check if it is a regex pattern (basic check)
            parts = re.split(separator, text)
        except re.error:
            # Fallback to simple string split if regex fails
            parts = text.split(separator)
        
        # If the separator didn't work (no split happened), move to the next separator
        if len(parts) == 1:
            self._recursive_split(text, sep_index + 1, chunks)
            return
        
        # Reconstruct chunks from parts
        current_chunk = ""
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            
            # Try to re-attach the separator to the part to preserve context
            # This is complex because re.split consumes the separator.
            # Simplified approach: Append matched separator if possible.
            part_with_sep = part
            
            # Simple heuristic: If regex split, we might have lost the separator.
            # For exact reconstruction, more complex logic is needed, but this suffices for RAG.
            if i < len(parts) - 1 and separator.strip():
                 # Try to find what actually matched to append it back
                try:
                    match = re.search(separator, text)
                    if match:
                        part_with_sep = part + match.group(0)
                    else:
                         part_with_sep = part + separator # Fallback
                except:
                    part_with_sep = part + separator
                
            # If adding this part exceeds chunk size, save current_chunk and start a new one
            if len(current_chunk) + len(part_with_sep) <= self.chunk_size:
                current_chunk += part_with_sep
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If the single part itself is too big, recurse on it
                if len(part) > self.chunk_size:
                    self._recursive_split(part, sep_index + 1, chunks)
                    current_chunk = "" # Reset
                else:
                    current_chunk = part_with_sep
        
        # Append any remaining text
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of Documents and split them.
        
        Args:
            documents (List[Document]): Original full-page documents.
            
        Returns:
            List[Document]: The resulting chunked documents.
        """
        chunked_docs = []
        
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            
            for chunk_text in text_chunks:
                new_doc = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                chunked_docs.append(new_doc)
        
        return chunked_docs


class LegalChunker:
    """
    High-level class to manage the PDF loading and chunking workflow.
    Specialized for Turkish Legal Texts.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the LegalChunker.
        
        Args:
            chunk_size (int): Size of chunks. Defaults to config value.
            chunk_overlap (int): Overlap size. Defaults to config value.
            separators (List[str], optional): Custom separators. Defaults to config value.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else LEGAL_SEPARATORS
        
        self.text_splitter = SimpleRecursiveSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def load_documents_from_directory(self, directory_path: Optional[Path] = None) -> List[Document]:
        """
        Load PDF files from the given directory using pypdf.
        
        Args:
            directory_path (Path, optional): Path to directory. Defaults to LEGAL_DATA_DIR.
            
        Returns:
            List[Document]: A list of Document objects (one per page usually).
            
        Raises:
            FileNotFoundError: If directory or PDFs are missing.
        """
        if directory_path is None:
            directory_path = LEGAL_DATA_DIR
            
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        pdf_files = list(directory_path.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in directory: {directory_path}")
        
        print(f"Loading PDF files from {directory_path}...")
        for pdf_path in pdf_files:
            try:
                reader = pypdf.PdfReader(str(pdf_path))
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():  # Skip empty pages
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_num + 1
                            }
                        )
                        documents.append(doc)
                print(f"  ✓ {pdf_path.name}: {len(reader.pages)} pages loaded.")
            except Exception as e:
                print(f"  ❌ Error processing {pdf_path.name}: {e}")
        
        print(f"✓ Total {len(documents)} pages loaded.")
        return documents
    
    def extract_article_numbers(self, text: str) -> List[str]:
        """
        Extract article numbers from a text block using Regex.
        
        Args:
            text (str): The text content.
            
        Returns:
            List[str]: Found article numbers (e.g. ['5', '76']).
        """
        article_numbers = []
        seen = set()  # To avoid duplicates within the same chunk
        
        # Patterns to catch various Turkish article headers
        patterns = [
            r'[Mm][Aa][Dd][Dd][Ee]\s+(\d+)',   # "Madde 5"
            r'Madde\s*:\s*(\d+)',               # "Madde: 5"
            r'^\s*(\d+)\s*[-–—]\s*',            # "76 - " (Start of line)
            r'GEÇİCİ\s+MADDE\s+(\d+)',          # "GEÇİCİ MADDE 1"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                article_no = match.group(1)
                if article_no not in seen:
                    article_numbers.append(article_no)
                    seen.add(article_no)
        
        return article_numbers
    
    def enrich_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Add extra information to chunk metadata.
        
        Args:
            chunks (List[Document]): The raw chunks.
            
        Returns:
            List[Document]: Chunks with added 'article_no', 'summary', etc.
        """
        for i, chunk in enumerate(chunks):
            # 1. Clean filename
            if "source" in chunk.metadata:
                source_path = Path(chunk.metadata["source"])
                chunk.metadata["source"] = source_path.name
            
            # 2. Extract Article Numbers
            article_numbers = self.extract_article_numbers(chunk.page_content)
            
            if article_numbers:
                # Primary article number (for display)
                chunk.metadata["article_no"] = article_numbers[0]
                
                # If multiple articles in one chunk, store all
                if len(article_numbers) > 1:
                    chunk.metadata["article_numbers"] = ",".join(article_numbers)
                    chunk.metadata["multi_article"] = True
            
            # 3. Add Chunk ID and Summary
            chunk.metadata["chunk_id"] = i
            chunk.metadata["summary"] = chunk.page_content[:100].replace("\n", " ")
        
        return chunks
    
    def chunk_documents(self, documents: List[Document], enrich: bool = True) -> List[Document]:
        """
        Main pipeline execution: Split documents -> Enrich Metadata.
        
        Args:
            documents (List[Document]): Raw documents.
            enrich (bool): Whether to run metadata enrichment.
            
        Returns:
            List[Document]: Final processed chunks.
        """
        print(f"Splitting documents (Size={self.chunk_size}, Overlap={self.chunk_overlap})...")
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ {len(chunks)} raw chunks created.")
        
        if enrich:
            print("Enriching metadata (detecting article numbers)...")
            chunks = self.enrich_metadata(chunks)
            print("✓ Metadata enrichment complete.")
        
        return chunks
    
    def analyze_chunks(self, chunks: List[Document], num_examples: int = 3):
        """
        Print statistics and examples of generated chunks for verification.
        """
        print("\n" + "="*70)
        print("CHUNK ANALYSIS REPORT")
        print("="*70)
        
        total_chunks = len(chunks)
        if total_chunks == 0:
            print("No chunks created.")
            return

        avg_length = sum(len(c.page_content) for c in chunks) / total_chunks
        chunks_with_article = sum(1 for c in chunks if "article_no" in c.metadata)
        
        print(f"\nTotal Chunks: {total_chunks}")
        print(f"Avg Length: {avg_length:.0f} chars")
        print(f"Chunks with Article IDs: {chunks_with_article} ({chunks_with_article/total_chunks*100:.1f}%)")
        
        print(f"\n{'='*70}")
        print(f"EXAMPLES (First {num_examples})")
        print("="*70)
        
        for i, chunk in enumerate(chunks[:num_examples]):
            print(f"\n--- Chunk #{i+1} ---")
            print(f"Source: {chunk.metadata.get('source', 'N/A')}")
            print(f"Page: {chunk.metadata.get('page', 'N/A')}")
            print(f"Article: {chunk.metadata.get('article_no', 'None')}")
            print(f"Content Preview:\n{chunk.page_content[:200]}...")
            print("-" * 30)


def main():
    """
    Entry point for running this module directly (e.g. for testing).
    """
    print("Running Chunking Module Test...")
    chunker = LegalChunker()
    
    try:
        docs = chunker.load_documents_from_directory()
        chunks = chunker.chunk_documents(docs)
        chunker.analyze_chunks(chunks)
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
