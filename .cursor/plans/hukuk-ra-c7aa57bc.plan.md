<!-- c7aa57bc-0aa0-4314-b073-d85a07a4d3be 27a90eae-1e52-42df-934e-b93c00a16aa7 -->
# Türk Hukuk Metinleri RAG Sistemi İmplementasyon Planı

## Proje Yapısı ve Temel Kurulum

### Klasör Yapısı

```
anayasaRAG/
├── mevzuat_data/          # PDF hukuk metinleri
│   └── anayasa.pdf
├── models/                # İndirilecek GGUF modelleri
├── hukuk_db/             # ChromaDB persistent storage
├── src/
│   ├── chunking.py       # Adım 1: Veri parçalama
│   ├── indexing.py       # Adım 2: Vektör DB oluşturma
│   ├── query_engine.py   # Adım 5: RAG pipeline
│   └── evaluation.py     # Adım 6: RAGAS değerlendirme
├── app.py                # Ana uygulama
├── requirements.txt      # Bağımlılıklar
└── config.py            # Konfigürasyon ayarları
```

## Adım 1: Proje Yapısı ve Dependencies

### 1.1 Temel Klasörlerin Oluşturulması

- `mevzuat_data/`, `models/`, `hukuk_db/`, `src/` klasörlerini oluştur
- `docs/anayasa.pdf`'yi `mevzuat_data/` klasörüne kopyala

### 1.2 Requirements.txt Hazırlama

Kritik bağımlılıklar:

- `llama-index` (core RAG framework)
- `llama-index-llms-llama-cpp` (GGUF LLM)
- `llama-index-embeddings-huggingface` (Türkçe embedding)
- `llama-index-vector-stores-chroma` (Vector store)
- `chromadb` (Persistent vector DB)
- `sentence-transformers` (Embedding backend)
- `pypdf` (PDF parsing)
- `ragas` (Evaluation framework)
- `langchain`, `langchain-community` (RAGAS için)
- `llama-cpp-python` (CUDA desteğiyle)

### 1.3 Config.py Oluşturma

Tüm model adları, yollar ve parametreler merkezi bir config dosyasında:

- Embedding model: `trmteb/turkish-embedding-model-fine-tuned`
- GGUF model path: `./models/sayhan-mistral-7b-instruct-v0.2-turkish-q4_k_m.gguf`
- ChromaDB path: `./hukuk_db`
- Chunk size: 1000, overlap: 100
- GPU layers: 15 (3050 GPU için)

## Adım 2: Hukuki Metin Parçalama (Chunking)

### 2.1 src/chunking.py Implementasyonu

**Kritik Özellikler:**

- `RecursiveCharacterTextSplitter` kullanımı
- Regex tabanlı hiyerarşik ayırıcılar:
  - `r"(\nMadde \d+\s*[-–—]\s*)"` (Madde başlıkları)
  - `r"(\n\n\([1-9][0-9]*\))"` (Fıkra numaraları)
  - `\n\n`, `\n`, `. `, ` ` (Fallback ayırıcılar)
- Metadata zenginleştirme (madde numarası çıkarma)

### 2.2 Test ve Validasyon

- `mevzuat_data/anayasa.pdf` üzerinde chunking test et
- Chunk'ların madde bütünlüğünü koruduğunu doğrula
- Örnek chunk'ları konsola yazdır

## Adım 3: Embedding Modeli Entegrasyonu

### 3.1 TR-MTEB Modelini İndirme

- `trmteb/turkish-embedding-model-fine-tuned` modelini ilk çalıştırmada otomatik indir
- Hugging Face cache mekanizmasını kullan

### 3.2 Embedding Test

- Örnek hukuki cümle çiftleri ile embedding kalitesini test et
- Cosine similarity skorlarını kontrol et

## Adım 4: ChromaDB Vector Store Kurulumu

### 4.1 src/indexing.py - Offline İndeksleme

**İki mod:**

- **CREATE MODE**: İlk kez çalıştırma
  - PDF'leri yükle
  - Chunking yap
  - Embedding hesapla
  - ChromaDB'ye kaydet (persistent)

- **UPDATE MODE**: Mevcut DB'ye yeni doküman ekleme

### 4.2 Persistent Client Yapılandırması

- `chromadb.PersistentClient(path="./hukuk_db")`
- Collection adı: `hukuk_mevzuat`
- Metadata filtreleme için indexing yapılandırması

## Adım 5: GGUF LLM Modeli Hazırlığı

### 5.1 Model İndirme Scripti

- Hugging Face'ten `sayhan/Mistral-7B-Instruct-v0.2-turkish-GGUF` indir
- Spesifik dosya: `sayhan-mistral-7b-instruct-v0.2-turkish-q4_k_m.gguf`
- `./models/` klasörüne kaydet

### 5.2 llama-cpp-python CUDA Kurulumu

**Kritik Kurulum:**

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

- Windows için CUDA toolkit kontrolü
- Kurulum sonrası GPU testi

### 5.3 LlamaCPP Konfigürasyonu

- `n_gpu_layers=15` (3050 4GB VRAM için başlangıç)
- `n_ctx=4096` (Context window)
- `temperature=0.1` (Hukuki kesinlik için düşük)
- `max_new_tokens=512`

## Adım 6: RAG Pipeline Entegrasyonu (LlamaIndex)

### 6.1 src/query_engine.py Implementasyonu

**LlamaIndex Global Settings:**

- `Settings.llm` = LlamaCPP instance
- `Settings.embed_model` = HuggingFaceEmbedding
- `Settings.chunk_size` = 1024
- `Settings.chunk_overlap` = 100

**Query Engine Yapılandırması:**

- `similarity_top_k=3` (En ilgili 3 chunk)
- Response mode: `compact` veya `refine`
- Streaming desteği (opsiyonel)

### 6.2 Citation (Kaynak Gösterme) Sistemi

- `response.source_nodes` üzerinden metadata çıkar
- Her cevapla birlikte:
  - Kaynak dosya adı
  - Madde numarası
  - Sayfa numarası
  - Similarity score

## Adım 7: Ana Uygulama (app.py)

### 7.1 CLI Interface

**Temel Özellikler:**

- Başlangıçta DB yükleme kontrolü
- İlk çalıştırmada otomatik indeksleme
- Interaktif soru-cevap döngüsü
- "exit" komutu ile çıkış

### 7.2 Hata Yönetimi

- GPU VRAM overflow kontrolü
- Model yükleme hataları
- Chunk bulunamama durumları
- Graceful degradation (GPU -> CPU fallback)

## Adım 8: RAGAS Evaluation Sistemi

### 8.1 src/evaluation.py Implementasyonu

**Test Dataset Oluşturma:**

- 5-10 örnek hukuki soru hazırla
- Ground truth cevapları (Anayasa'dan)
- Beklenen kaynak maddeleri

**RAGAS Metrikleri:**

- `context_precision`: Gürültüsüz retrieval
- `context_recall`: Tam bilgi retrieval
- `faithfulness`: Halüsinasyon kontrolü
- `answer_relevancy`: Odaklanmış cevaplar

### 8.2 Lokal LLM Evaluator

- RAGAS için aynı GGUF modelini kullan
- `LangchainLLMWrapper` ile sarmalama
- Sonuçları CSV/JSON export

## Adım 9: İyileştirmeler ve Optimizasyon

### 9.1 Retrieval İyileştirmeleri

- Hybrid search (Dense + BM25)
- Reranking modeli ekleme (opsiyonel)
- Query expansion/rewriting

### 9.2 Performans Optimizasyonu

- `n_gpu_layers` fine-tuning (memory profiling)
- Batch processing için embedding caching
- ChromaDB index optimizasyonu

### 9.3 Ek Özellikler

- Gradio/Streamlit web arayüzü (opsiyonel)
- Konuşma geçmişi (chat history)
- Multi-turn conversation desteği

## Başarı Kriterleri

1. **Doğruluk**: RAGAS faithfulness > 0.8
2. **İlgililik**: context_precision > 0.85
3. **Performans**: Sorgu başına < 10 saniye
4. **Kararlılık**: 4GB VRAM içinde stabil çalışma
5. **Kaynak Gösterme**: Her cevap doğru maddeyi referans etmeli

## Risk Azaltma

- **GPU Memory Overflow**: `n_gpu_layers` değerini kademeli düşür (15 -> 10 -> 5)
- **Düşük Kalite Cevaplar**: Retrieval'a odaklan, daha fazla context sağla
- **Yavaş İnferans**: Q4_K_M yerine Q4_0 dene (daha hızlı ama düşük kalite)
- **Türkçe Tokenization**: Modelin Türkçe tokenizer'ını kontrol et

### To-dos

- [ ] Proje klasör yapısını oluştur (mevzuat_data, models, hukuk_db, src) ve docs/anayasa.pdf'yi taşı
- [ ] requirements.txt ve config.py dosyalarını oluştur, tüm bağımlılıkları tanımla
- [ ] Python paketlerini kur (llama-index, chromadb, sentence-transformers, vb.) ve CUDA destekli llama-cpp-python kur
- [ ] src/chunking.py - Regex tabanlı hiyerarşik chunking implementasyonu (RecursiveCharacterTextSplitter)
- [ ] Chunking'i anayasa.pdf üzerinde test et, madde bütünlüğünü doğrula
- [ ] src/indexing.py - ChromaDB ile offline indeksleme scripti (CREATE ve UPDATE modları)
- [ ] Hugging Face'ten sayhan/Mistral-7B-Instruct-v0.2-turkish-GGUF (Q4_K_M) indir ve models/ klasörüne kaydet
- [ ] src/query_engine.py - LlamaIndex ile RAG pipeline, LlamaCPP entegrasyonu, retrieval ve generation
- [ ] app.py - CLI interface ile ana uygulama, interaktif soru-cevap döngüsü, citation sistemi
- [ ] RAG sistemini test et, örnek sorularla doğruluk ve performansı değerlendir, GPU memory kullanımını kontrol et
- [ ] src/evaluation.py - RAGAS ile evaluation framework, test dataset hazırlama, metrik hesaplama
- [ ] n_gpu_layers fine-tuning, retrieval parametrelerini optimize et, memory profiling yap