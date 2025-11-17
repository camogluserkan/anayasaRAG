"""
Legal RAG System - Main Application

Interactive CLI interface for Q&A on the Turkish Constitution.
"""

import os
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import sys
from pathlib import Path

# Import local modules
sys.path.append(str(Path(__file__).parent))
from src.query_engine import LegalRAGEngine
from config import check_data_exists, check_model_exists


def print_welcome():
    """Welcome message"""
    print("\n" + "="*70)
    print(" " * 15 + "LEGAL RAG SYSTEM")
    print(" " * 10 + "Turkish Constitution Q&A System")
    print("="*70)
    print("\nWelcome! You can ask questions about the Turkish Constitution.")
    print("\nCommands:")
    print("  - Ask a question: Just type your question")
    print("  - 'exit' or 'quit': Exit the program")
    print("  - 'help': Show this message")
    print("="*70 + "\n")


def print_help():
    """Help message"""
    print("\n" + "="*70)
    print("HELP")
    print("="*70)
    print("\nExample Questions:")
    print("  - Milletvekili seçilme yaşı kaçtır? (What is the MP election age?)")
    print("  - Cumhurbaşkanı nasıl seçilir? (How is the president elected?)")
    print("  - Türkiye Cumhuriyeti'nin başkenti neresidir? (What is Turkey's capital?)")
    print("  - Anayasa'nın değiştirilemez maddeleri nelerdir? (What are immutable articles?)")
    print("\nTips:")
    print("  - Ask clear and specific questions")
    print("  - System works only with Constitutional text")
    print("  - Source articles are shown with each answer")
    print("="*70 + "\n")


def main():
    """Main application loop"""
    
    # Welcome message
    print_welcome()
    
    # System check
    print("[System Check]")
    if not check_data_exists():
        print("❌ Data files not found!")
        print("Please run 'python src/indexing.py' first.")
        return
    
    if not check_model_exists():
        print("❌ LLM model not found!")
        print("Please run 'python download_model.py' first.")
        return
    
    print("✓ All files ready\n")
    
    # Start RAG Engine
    try:
        print("[Starting System...]")
        engine = LegalRAGEngine()
        
        # First access for lazy loading
        _ = engine.collection
        _ = engine.embedding_model
        _ = engine.llm
        
        print("\n✅ System ready! Waiting for your questions...\n")
        
    except Exception as e:
        print(f"\n❌ System startup error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Main loop
    while True:
        try:
            # Get question from user
            question = input("\n💬 Question: ").strip()
            
            if not question:
                continue
            
            # Check commands
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Closing Legal RAG System...")
                break
            
            if question.lower() in ['help', 'h']:
                print_help()
                continue
            
            # Process question
            print(f"\n{'─'*70}")
            print(f"🔍 Processing your query...")
            print(f"{'─'*70}")
            
            response = engine.query(question)
            
            # Show answer
            print(f"\n{'='*70}")
            print("📝 ANSWER:")
            print(f"{'='*70}")
            print(f"\n{response['answer']}\n")
            
            # Show sources
            print(f"{'='*70}")
            print("📚 SOURCE ARTICLES:")
            print(f"{'='*70}")
            
            for i, chunk in enumerate(response['sources'], 1):
                source = chunk['metadata'].get('source', 'Unknown')
                article = chunk['metadata'].get('article_no', '')
                page = chunk['metadata'].get('page', 'N/A')
                
                print(f"\n[{i}] {source}")
                if article:
                    print(f"    Article: {article}")
                print(f"    Page: {page}")
                
                # Show first 200 characters
                text_preview = chunk['text'][:200].replace('\n', ' ')
                print(f"    Preview: {text_preview}...")
            
            # Statistics
            print(f"\n{'─'*70}")
            print(f"📊 Token usage: {response['prompt_tokens']} prompt + {response['completion_tokens']} completion")
            print(f"{'─'*70}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Program terminated. Goodbye!")
            break
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or ask a different question.\n")
            continue


if __name__ == "__main__":
    main()
