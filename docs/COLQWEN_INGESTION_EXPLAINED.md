# ColQwen Ingestion Pipeline: Technical Deep Dive

This document explains **how ColQwen ingestion works** with references to actual code and research papers.

---

## Overview: What Makes ColQwen Different?

### Regular RAG vs. ColQwen RAG

| Aspect | Regular RAG (Ollama) | ColQwen RAG |
|--------|---------------------|-------------|
| **Input** | Text chunks from PDF | Full page images |
| **Embedding** | Single 768-dim vector per chunk | **Multi-vector** per page (1024 vectors × 128-dim each) |
| **Model** | `nomic-embed-text` | `ColQwen2.5` (vision-language model) |
| **Granularity** | Chunk-level (paragraphs, tables) | Page-level (entire visual layout) |
| **What it captures** | Semantic text meaning | **Visual + semantic** (diagrams, layouts, tables, charts) |

**Source**: [ColQwen README](file:///Users/lab/Documents/vsm_demo_v02/colqwen/README.md) - "ColQwen2.5 is a multimodal late-interaction model"

---

## Step-by-Step: ColQwen Ingestion Process

### **Step 1: Load ColQwen2.5 Model**

**Code**: [`scripts/colqwen_ingest.py:24-37`](file:///Users/lab/Documents/vsm_demo_v02/scripts/colqwen_ingest.py#L24-L37)

```python
def initialize_colqwen(device="mps"):
    model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",  # ← Hugging Face model ID
        dtype=torch.bfloat16,       # ← Memory-efficient 16-bit precision
        device_map=device,          # ← Apple Silicon MPS support
    ).eval()
    
    processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
    return model, processor
```

**What happens:**
1. Downloads **ColQwen2.5-v0.2** from Hugging Face (~8GB)
2. Loads model onto Apple Silicon GPU (MPS) for hardware acceleration
3. Uses `bfloat16` precision (reduces memory by 50% vs. float32)

**Model Architecture** (from ColPali paper):
- Based on **Qwen2-VL-2B** vision-language model
- Modified with **ColBERT-style late interaction**
- Produces **1024 patch vectors** (128-dim each) per image

**Source**: [colpali_engine.models.ColQwen2_5](https://github.com/illuin-tech/colpali-engine)

---

### **Step 2: Convert PDF to Images**

**Code**: [`scripts/colqwen_ingest.py:67-72`](file:///Users/lab/Documents/vsm_demo_v02/scripts/colqwen_ingest.py#L67-L72)

```python
def pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    images = convert_from_path(pdf_path)  # Uses pdf2image + poppler
    return images
```

**Why images?**
- ColQwen2.5 is a **vision model** - it needs pixel data, not text
- Captures **visual layout**: diagrams, charts, tables, fonts, spacing
- Each page becomes a PIL Image (RGB, variable resolution)

**Source**: This is standard for multimodal document understanding. See ColPali paper: "We process each page as an image..."

---

### **Step 3: Generate Multi-Vector Embeddings**

**Code**: [`scripts/colqwen_ingest.py:74-98`](file:///Users/lab/Documents/vsm_demo_v02/scripts/colqwen_ingest.py#L74-L98)

```python
def generate_multivector_embeddings(images, model, processor, device):
    embeddings = []
    batch_size = 4  # Process 4 pages at a time
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # Preprocess images (resize, normalize, tokenize)
        batch_input = processor.process_images(batch).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = model(**batch_input)
        
        # Each embedding has shape: (seq_len, embed_dim)
        # For ColQwen2.5: (~1024, 128)
        for emb in batch_embeddings:
            embeddings.append(emb.cpu().numpy())  # Shape: (1024, 128)
    
    return embeddings
```

**What happens:**
1. **Preprocessing** (`processor.process_images`):
   - Resizes images to model's expected input size
   - Divides image into **patches** (similar to ViT - Vision Transformer)
   - Converts to tensors suitable for the vision encoder

2. **Forward pass** (`model(**batch_input)`):
   - Vision encoder processes each patch
   - ColBERT-style projection creates embeddings for each patch
   - **Result**: ~1024 vectors (one per patch) × 128 dimensions

3. **Output**: Each page → multi-vector embedding (numpy array of shape `[1024, 128]`)

**Source**: [colqwen/api/rag.py:27-31](file:///Users/lab/Documents/vsm_demo_v02/colqwen/api/rag.py#L27-L31) - Same logic used in retrieval

**Technical Detail**: 
- Number of vectors varies based on image size (typically 1024 for standard PDFs)
- Each vector represents a **visual patch** of the page
- This is called **"late interaction"** because similarity is computed patch-by-patch, not as a single pooled vector

---

### **Step 4: Create Weaviate Collection with Multi-Vector Support**

**Code**: [`scripts/colqwen_ingest.py:39-65`](file:///Users/lab/Documents/vsm_demo_v02/scripts/colqwen_ingest.py#L39-L65)

```python
def create_multivector_collection(client):
    coll = client.collections.create(
        name="PDFDocuments",  # Different from AssetManual
        properties=[
            Property(name="page_id", data_type=DataType.INT),
            Property(name="asset_manual", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="image_path", data_type=DataType.TEXT),
        ],
        # NOTE: Multi-vector support is enabled by providing
        # list[list[float]] as the vector during ingestion
    )
    return coll
```

**Weaviate Multi-Vector Configuration**:
- Weaviate v1.29+ supports multi-vector embeddings natively
- No special `vector_config` needed - just pass a 2D array as the vector
- Storage: Each object stores a **matrix** of vectors, not a single vector

**Source**: [Weaviate Multi-Vector Docs](https://weaviate.io/developers/weaviate/config-refs/schema/multi-vector) - "Multi-vector embeddings allow for more nuanced comparisons"

---

### **Step 5: Ingest Pages with Multi-Vector Embeddings**

**Code**: [`scripts/colqwen_ingest.py:100-121`](file:///Users/lab/Documents/vsm_demo_v02/scripts/colqwen_ingest.py#L100-L121)

```python
def ingest_pages(client, manual_name, images, embeddings):
    coll = client.collections.get("PDFDocuments")
    
    with coll.batch.fixed_size(batch_size=10) as batch:
        for page_num, (image, embedding) in enumerate(zip(images, embeddings), start=1):
            # Convert numpy array (1024, 128) to nested list
            multi_vector = embedding.tolist()  # [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
            
            props = {
                "page_id": page_num,
                "asset_manual": manual_name,
                "page_number": page_num,
                "image_path": f"static/previews/{manual_name.lower()}/page-{page_num}.png",
            }
            
            # Store object with multi-vector
            batch.add_object(properties=props, vector=multi_vector)
```

**Weaviate Object Structure**:
```json
{
  "properties": {
    "page_id": 47,
    "asset_manual": "UK Firmware Manual",
    "page_number": 47,
    "image_path": "static/previews/uk_firmware_manual/page-47.png"
  },
  "vector": [
    [0.023, -0.15, 0.42, ...],  // Vector for patch 1 (128-dim)
    [0.11, 0.08, -0.21, ...],   // Vector for patch 2 (128-dim)
    ...                          // ~1024 vectors total
  ]
}
```

**Key Difference from Regular RAG**:
- Regular RAG: 1 object per chunk → 1 vector per object (768-dim)
- ColQwen: 1 object per page → **1024 vectors per object** (128-dim each)

---

## How Multi-Vector Search Works (Late Interaction)

**Code**: [colqwen/api/rag.py:82-89](file:///Users/lab/Documents/vsm_demo_v02/colqwen/api/rag.py#L82-L89)

```python
def retrieve(self, query, top_k=3):
    # 1. Generate multi-vector embedding for query
    query_embedding = self.embedder.multi_vectorize_text(query)  # Shape: (N_query_tokens, 128)
    
    # 2. Search Weaviate using MaxSim (late interaction)
    response = self.collection.query.near_vector(
        near_vector=query_embedding.cpu().numpy(),
        limit=top_k
    )
```

**Late Interaction Similarity (MaxSim)**:
```
For each document page:
  For each query vector q_i:
    similarity_i = max(cosine_sim(q_i, d_j) for all document vectors d_j)
  
  Total score = sum(similarity_i for all query vectors)
```

**Why is this better?**
1. **Fine-grained matching**: Each query token can match its most relevant document patch
2. **Visual grounding**: Can match specific diagram elements, table cells, etc.
3. **Context-aware**: Preserves spatial relationships between patches

**Source**: ColBERT paper (Khattab & Zaharia, 2020) - "Late interaction allows for more expressive matching"

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ColQwen Ingestion                         │
└─────────────────────────────────────────────────────────────┘

1. PDF File (uk_firmware.pdf)
     ↓
2. Convert to Images (pdf2image)
     ↓ [132 PIL Images]
3. ColQwen2.5 Processor
     ↓ [Resize, normalize, patchify]
4. ColQwen2.5 Model (Vision Encoder)
     ↓ [Forward pass on GPU/MPS]
5. Multi-Vector Embeddings
     ↓ [132 pages × ~1024 vectors × 128-dim]
6. Weaviate Ingestion
     ↓ [Store in PDFDocuments collection]
7. Ready for Search
     ↓
8. Query Processing (same as ingestion: text → multi-vector)
     ↓
9. Late Interaction Search (MaxSim)
     ↓
10. Top-K Pages Retrieved

┌──────────────────────────────────────────────────────────────┐
│                   Regular RAG (for comparison)                │
└──────────────────────────────────────────────────────────────┘

1. PDF File
     ↓
2. LandingAI ADE Parser (text extraction + bbox)
     ↓ [811 text chunks]
3. Ollama nomic-embed-text
     ↓ [811 chunks × 768-dim single vectors]
4. Weaviate Ingestion (AssetManual collection)
     ↓
5. Query → Ollama → single 768-dim vector
     ↓
6. Cosine similarity search
     ↓
7. Top-K Chunks Retrieved
```

---

## Memory and Performance Considerations

### Storage Requirements (Per Page)

**ColQwen**:
- Multi-vector: 1024 vectors × 128-dim × 4 bytes (float32) = **524 KB per page**
- Metadata: ~200 bytes
- **Total**: ~525 KB per page

**Regular RAG (Per Chunk)**:
- Single vector: 768-dim × 4 bytes = **3 KB per chunk**
- Metadata + text: ~2 KB
- **Total**: ~5 KB per chunk

**For 132-page PDF**:
- ColQwen: 132 pages × 525 KB = **~68 MB**
- Regular RAG: ~811 chunks × 5 KB = **~4 MB**

**Trade-off**: ColQwen uses 17× more storage but captures visual information

**Source**: Calculated from model specifications in [colqwen/api/rag.py](file:///Users/lab/Documents/vsm_demo_v02/colqwen/api/rag.py)

### Processing Speed

**Ingestion** (132-page PDF on Mac Studio M3):
- ColQwen: ~10-15 minutes (model download + GPU inference)
- Regular RAG: ~5 minutes (LandingAI API + Ollama embedding)

**Search** (per query):
- ColQwen: ~3-5 seconds (multi-vector generation + MaxSim)
- Regular RAG: ~0.5 seconds (single vector + cosine sim)

**Source**: Empirical timing from TESTING.md and architecture plan

---

## Key Technical Concepts

### 1. **Multi-Vector Embeddings**
Each page is represented by a **set of vectors** (not a single pooled vector).
- Preserves fine-grained information
- Enables patch-level matching

**Source**: [Weaviate Multi-Vector Docs](https://weaviate.io/blog/multi-vector-support)

### 2. **Late Interaction (ColBERT/ColPali style)**
Similarity is computed between **individual query tokens** and **individual document patches**.
- Query vectors: one per token in the query
- Document vectors: one per visual patch in the page
- MaxSim: for each query vector, find the best matching document vector

**Source**: ColBERT paper (arXiv:2004.12832)

### 3. **Vision-Language Model**
ColQwen2.5 is built on Qwen2-VL:
- Vision encoder: processes images into patch embeddings
- Language-aligned: embeddings can be compared with text queries
- Pre-trained on 5M+ document-image pairs

**Source**: ColPali paper and [Qwen2-VL model card](https://huggingface.co/Qwen/Qwen2-VL-2B)

---

## References & Sources

1. **Code Files**:
   - [scripts/colqwen_ingest.py](file:///Users/lab/Documents/vsm_demo_v02/scripts/colqwen_ingest.py) - Ingestion script I created
   - [colqwen/api/rag.py](file:///Users/lab/Documents/vsm_demo_v02/colqwen/api/rag.py) - Retrieval and generation logic
   - [colqwen/README.md](file:///Users/lab/Documents/vsm_demo_v02/colqwen/README.md) - ColQwen system overview

2. **Model & Libraries**:
   - [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) - Hugging Face model
   - [colpali-engine](https://github.com/illuin-tech/colpali-engine) - Python library
   - [Weaviate Multi-Vector Support](https://weaviate.io/developers/weaviate/config-refs/schema/multi-vector)

3. **Research Papers**:
   - ColBERT (2020): "Contextualized Late Interaction over BERT" - introduced late interaction
   - ColPali (2024): "Efficient Document Retrieval with Vision Language Models" - applied to document images

4. **Architecture Plan**:
   - [implementation_plan.md](file:///Users/lab/.gemini/antigravity/brain/6224a07b-64ff-4127-83dd-ce146aed5fc8/implementation_plan.md) - Agentic RAG design decisions

---

## Summary

ColQwen ingestion works by:
1. **Converting PDF pages to images** (preserves visual layout)
2. **Encoding each image** into ~1024 patch vectors (multi-vector embedding)
3. **Storing in Weaviate** with multi-vector support
4. **Searching via late interaction** (MaxSim) for fine-grained matching

This enables **visual grounding** - the ability to match queries to specific diagrams, tables, and visual elements in documents, which single-vector RAG cannot do effectively.
