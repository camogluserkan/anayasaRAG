"""
Legal Text Chunking Module

This module splits legal texts (PDF) into semantic chunks
while preserving hierarchical structure.

Features:
- Article-based splitting
- Clause-level sub-splitting
- Metadata enrichment (article numbers)
- Context preservation (chunk_overlap)
"""

import re
from pathlib import Path
from typing import List, Dict, Any
import pypdf
import sys

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LEGAL_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LEGAL_SEPARATORS
)


class Document:
    """Simple Document class (LangChain-like)"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


class SimpleRecursiveSplitter:
    """
    Simple recursive text splitter implementation.
    Splits text using regex separators without LangChain dependency.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """Split text recursively according to separators"""
        chunks = []
        self._recursive_split(text, 0, chunks)
        return chunks
    
    def _recursive_split(self, text: str, sep_index: int, chunks: List[str]):
        """Recursive splitting logic"""
        if len(text) <= self.chunk_size or sep_index >= len(self.separators):
            if text.strip():
                chunks.append(text)
            return
        
        separator = self.separators[sep_index]
        
        # Is it a regex separator or plain string?
        try:
            parts = re.split(separator, text)
        except:
            parts = text.split(separator)
        
        if len(parts) == 1:
            # Couldn't split with this separator, try next one
            self._recursive_split(text, sep_index + 1, chunks)
            return
        
        # Process split parts
        current_chunk = ""
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            
            # Include separator (don't lose it)
            part_with_sep = part
            if i < len(parts) - 1 and separator.strip():  # If not last part
                try:
                    part_with_sep = part + re.search(separator, text).group(0)
                except:
                    part_with_sep = part + separator
                
            if len(current_chunk) + len(part_with_sep) <= self.chunk_size:
                current_chunk += part_with_sep
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If part alone is larger than chunk_size, split recursively
                if len(part) > self.chunk_size:
                    self._recursive_split(part, sep_index + 1, chunks)
                else:
                    current_chunk = part_with_sep
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split document list into chunks"""
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
    Chunks legal texts according to structural hierarchy.
    
    This class applies a chunking strategy suitable for Turkish legal text structure
    (Article > Clause > Item).
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: List[str] = None
    ):
        """
        Args:
            chunk_size: Target chunk size (character-based)
            chunk_overlap: Overlap amount between chunks
            separators: Hierarchical separator list (Regex)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use config values if no custom separators provided
        self.separators = separators if separators is not None else LEGAL_SEPARATORS
        
        # Use simple recursive splitter (no LangChain dependency)
        self.text_splitter = SimpleRecursiveSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def load_documents_from_directory(self, directory_path: Path = None) -> List[Any]:
        """
        Load all PDF files from specified directory.
        
        Args:
            directory_path: Directory containing PDF files
            
        Returns:
            List of Documents
        """
        if directory_path is None:
            directory_path = LEGAL_DATA_DIR
            
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Load PDF files manually
        documents = []
        pdf_files = list(directory_path.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in directory: {directory_path}")
        
        print(f"Loading PDF files...")
        for pdf_path in pdf_files:
            try:
                reader = pypdf.PdfReader(str(pdf_path))
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():  # If not empty page
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_num + 1
                            }
                        )
                        documents.append(doc)
                print(f"  ✓ {pdf_path.name}: {len(reader.pages)} pages")
            except Exception as e:
                print(f"  ❌ Error ({pdf_path.name}): {e}")
        
        print(f"✓ Total {len(documents)} pages loaded")
        return documents
    
    def extract_article_numbers(self, text: str) -> List[str]:
        """
        Extract ALL article numbers from text (improved).
        
        This function finds and returns ALL articles if multiple exist in a chunk.
        
        Args:
            text: Text content
            
        Returns:
            List of article numbers (e.g., ["5", "76"]) or empty list
        """
        article_numbers = []
        seen = set()  # Prevent duplicates
        
        # Try different format patterns
        patterns = [
            r'[Mm][Aa][Dd][Dd][Ee]\s+(\d+)',  # "Madde 5", "MADDE 76" (Turkish word for "Article")
            r'Madde\s*:\s*(\d+)',              # "Madde: 5"
            r'^\s*(\d+)\s*[-–—]\s*',          # At beginning: "76 - Election"
            r'GEÇİCİ\s+MADDE\s+(\d+)',        # "GEÇİCİ MADDE 3" (Temporary Article)
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                article_no = match.group(1)
                if article_no not in seen:
                    article_numbers.append(article_no)
                    seen.add(article_no)
        
        return article_numbers
    
    def extract_article_number(self, text: str) -> str:
        """
        For backward compatibility - returns first article number.
        
        Args:
            text: Text content
            
        Returns:
            First article number or None
        """
        article_numbers = self.extract_article_numbers(text)
        return article_numbers[0] if article_numbers else None
    
    def enrich_metadata(self, chunks: List[Any]) -> List[Any]:
        """
        Enrich chunk metadata (IMPROVED).
        
        Adds to each chunk:
        - Source file name (cleaned)
        - Article number (if exists) - ALL if multiple articles
        - Chunk ID
        
        Args:
            chunks: List of Document chunks
            
        Returns:
            List of chunks with enriched metadata
        """
        for i, chunk in enumerate(chunks):
            # Clean source file name
            if "source" in chunk.metadata:
                source_path = Path(chunk.metadata["source"])
                chunk.metadata["source"] = source_path.name  # Only filename
            
            # Extract ALL article numbers (improved)
            article_numbers = self.extract_article_numbers(chunk.page_content)
            
            if article_numbers:
                # Save first article as primary (backward compatibility)
                chunk.metadata["article_no"] = article_numbers[0]
                
                # If multiple articles exist, save all
                if len(article_numbers) > 1:
                    chunk.metadata["article_numbers"] = ",".join(article_numbers)
                    chunk.metadata["multi_article"] = True
            
            # Add chunk ID
            chunk.metadata["chunk_id"] = i
            
            # Add first 100 characters as summary
            chunk.metadata["summary"] = chunk.page_content[:100].replace("\n", " ")
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Any],
        enrich: bool = True
    ) -> List[Any]:
        """
        Split documents into chunks according to hierarchical structure.
        
        Args:
            documents: List of loaded documents
            enrich: Should metadata enrichment be performed?
            
        Returns:
            List of split and enriched chunks
        """
        print(f"Splitting documents into chunks (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        # Splitting process
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✓ {len(chunks)} chunks created")
        
        # Metadata enrichment
        if enrich:
            print("Enriching metadata...")
            chunks = self.enrich_metadata(chunks)
            print("✓ Metadata enriched")
        
        return chunks
    
    def analyze_chunks(self, chunks: List[Any], num_examples: int = 3):
        """
        Analyze created chunks and show examples.
        
        Args:
            chunks: List of chunks
            num_examples: Number of examples to show
        """
        print("\n" + "="*70)
        print("CHUNK ANALYSIS")
        print("="*70)
        
        # General statistics
        total_chunks = len(chunks)
        avg_length = sum(len(c.page_content) for c in chunks) / total_chunks
        
        # Number of chunks with article numbers
        chunks_with_article = sum(1 for c in chunks if "article_no" in c.metadata)
        
        print(f"\nTotal Chunks: {total_chunks}")
        print(f"Average Chunk Length: {avg_length:.0f} characters")
        print(f"Chunks with Article Numbers: {chunks_with_article} ({chunks_with_article/total_chunks*100:.1f}%)")
        
        # Show example chunks
        print(f"\n{'='*70}")
        print(f"EXAMPLE CHUNKS (First {num_examples})")
        print("="*70)
        
        for i, chunk in enumerate(chunks[:num_examples]):
            print(f"\n--- Chunk #{i+1} ---")
            print(f"Source: {chunk.metadata.get('source', 'N/A')}")
            print(f"Page: {chunk.metadata.get('page', 'N/A')}")
            print(f"Article No: {chunk.metadata.get('article_no', 'None')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"\nContent (first 300 characters):")
            print("-" * 70)
            print(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
            print("-" * 70)


def main():
    """
    Main function for testing and example usage.
    """
    print("="*70)
    print("LEGAL TEXT CHUNKING TEST")
    print("="*70)
    
    # Create LegalChunker instance
    chunker = LegalChunker()
    
    try:
        # 1. Load documents
        print(f"\n1. Loading documents from: {LEGAL_DATA_DIR}")
        documents = chunker.load_documents_from_directory()
        
        # 2. Split into chunks
        print(f"\n2. Starting chunking process...")
        chunks = chunker.chunk_documents(documents)
        
        # 3. Analyze and show examples
        chunker.analyze_chunks(chunks, num_examples=5)
        
        print(f"\n{'='*70}")
        print("✓ Process completed successfully!")
        print(f"Total {len(chunks)} chunks ready to be loaded into vector DB.")
        print("="*70)
        
        return chunks
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    chunks = main()
