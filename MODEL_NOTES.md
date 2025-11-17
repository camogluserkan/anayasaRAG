# Model History and Notes

## ✅ Currently Active Model
- **Model**: TheBloke/Mistral-7B-Instruct-v0.2-GGUF
- **File**: mistral-7b-instruct-v0.2.Q4_K_M.gguf
- **Quantization**: Q4_K_M (4-bit, mixed)
- **Size**: ~4.37GB
- **GPU Layers**: 15
- **Context**: 4096 tokens
- **Prompt Format**: Mistral Instruct (`<s>[INST] ... [/INST]`)
- **Test Results (v2 - After improved chunking)**:
  - ✅ Question 1: Capital - CORRECT (Ankara, ARTICLE 3)
  - ⚠️ Question 2: MP Election Age - PARTIALLY CORRECT (Found correct article 76, answer mentions "18" instead of "25")
  - ❌ Question 3: Presidential Election - INCORRECT (Expected article 101 not found/mentioned)
  - ❌ Question 4: Immutable Articles - INCORRECT (Expected article 4 not found)
  - **Success Rate**: ~37.5%
- **Previous Test (v1)**:
  - ✅ Question 1: Capital - CORRECT
  - ✅ Question 2: MP Election Age - CORRECT (18 years, Article 76)
  - ⚠️ Question 3: President - PARTIALLY CORRECT (answer truncated, MAX_NEW_TOKENS increased)
  - ❌ Question 4: Immutable Articles - INCORRECT (Article 4 was missing from chunks)
- **Strengths**: Good Turkish performance, reliable
- **Weaknesses**: Sometimes truncates long answers

---

## ❌ Tested - Failed
### Qwen2.5-3B (bartowski/Qwen2.5-3B-GGUF)
- **File**: Qwen2.5-3B-Q4_K_M.gguf
- **Size**: ~1.93GB
- **Test Date**: November 9, 2025
- **Result**: ❌ FAILED
- **Problem**: 
  - With ChatML format (`<|im_start|>...<|im_end|>`) repeats system prompt
  - Generates nonsensical outputs (system prompt in infinite loop)
  - Stop tokens don't work properly
- **Decision**: Reverted to Mistral 7B

---

## Notes and Improvements
- **Chunking Improvements**: 
  - Added `extract_article_numbers()` for multi-article support
  - CHUNK_SIZE: 1200, CHUNK_OVERLAP: 200
  - LEGAL_SEPARATORS made more aggressive (ARTICLE headers prioritized)
- **Retrieval Improvements**:
  - SIMILARITY_TOP_K: 5
  - Query expansion added
- **Generation Improvements**:
  - MAX_NEW_TOKENS: 512 → 768 (for longer answers)
  - Prompt engineering (Mistral Instruct format optimized)
- **Future Tests**: Larger Turkish fine-tuned models could be tested
