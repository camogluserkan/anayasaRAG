"""
Test Queries - Test RAG System Accuracy
"""

import os
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.query_engine import LegalRAGEngine

def test_rag_system():
    """Test with known correct answers"""
    
    print("="*70)
    print("RAG SYSTEM TEST - ACCURACY CHECK")
    print("="*70)
    
    # Test questions and expected answers
    test_cases = [
        {
            "question": "Türkiye Cumhuriyeti'nin başkenti neresidir?",
            "expected_article": "3",
            "expected_keyword": "Ankara"
        },
        {
            "question": "Milletvekili seçilme yaşı kaçtır?",
            "expected_article": "76",
            "expected_keyword": "25"
        },
        {
            "question": "Cumhurbaşkanı nasıl seçilir?",
            "expected_article": "101",
            "expected_keyword": "halk"  # Turkish word for "people"
        },
        {
            "question": "Anayasa'nın değiştirilemez maddeleri nelerdir?",
            "expected_article": "4",
            "expected_keyword": "Cumhuriyet"  # Turkish word for "Republic"
        }
    ]
    
    # Start engine
    print("\n[Starting System...]")
    engine = LegalRAGEngine()
    
    # Lazy loading
    _ = engine.collection
    _ = engine.embedding_model
    _ = engine.llm
    
    print("\n✅ System ready!\n")
    
    # Test results
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'='*70}")
        print(f"Question: {test['question']}")
        
        try:
            response = engine.query(test['question'])
            
            # Check answer (IMPROVED)
            answer = response['answer'].lower()
            answer_original = response['answer']  # For article check, keep original
            sources = response['sources']
            
            # Extract source articles (scope fix)
            source_article_nos = [s['metadata'].get('article_no', '') for s in sources]
            multi_article_nos = []
            for s in sources:
                if 'article_numbers' in s['metadata']:
                    multi_article_nos.extend(s['metadata']['article_numbers'].split(','))
            all_article_nos = source_article_nos + multi_article_nos
            
            # Is expected keyword present?
            has_keyword = test['expected_keyword'].lower() in answer
            
            # Is expected article in ANSWER? (Not in sources!)
            # LLM usually writes in "MADDE 76" format (Turkish word for "ARTICLE")
            import re
            article_pattern = rf"[Mm][Aa][Dd][Dd][Ee]\s*{test['expected_article']}\b"
            has_article = bool(re.search(article_pattern, answer_original))
            
            # Alternative: Check in sources too (backup)
            if not has_article:
                has_article = test['expected_article'] in all_article_nos
            
            print(f"\n📝 Answer: {response['answer'][:200]}...")
            print(f"\n📚 Found Articles: {', '.join(filter(None, source_article_nos[:3]))}")
            
            # Evaluation
            if has_keyword and has_article:
                print(f"\n✅ TEST SUCCESSFUL")
                print(f"   ✓ Expected keyword found: '{test['expected_keyword']}'")
                print(f"   ✓ Correct article found: {test['expected_article']}")
                passed += 1
            elif has_keyword:
                print(f"\n⚠ PARTIALLY SUCCESSFUL")
                print(f"   ✓ Expected keyword found: '{test['expected_keyword']}'")
                print(f"   ✗ Expected article not found: {test['expected_article']}")
                passed += 0.5
                failed += 0.5
            elif has_article:
                print(f"\n⚠ PARTIALLY SUCCESSFUL")
                print(f"   ✗ Expected keyword not found: '{test['expected_keyword']}'")
                print(f"   ✓ Correct article found: {test['expected_article']}")
                passed += 0.5
                failed += 0.5
            else:
                print(f"\n❌ TEST FAILED")
                print(f"   ✗ Expected keyword not found: '{test['expected_keyword']}'")
                print(f"   ✗ Expected article not found: {test['expected_article']}")
                failed += 1
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_rag_system()
