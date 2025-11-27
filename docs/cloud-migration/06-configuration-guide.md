# Configuration Guide

**Last Updated:** 2025-11-26  
**Target System:** VSM v0.3

---

## Environment Variables

The system behavior is controlled by the `VSM_MODE` environment variable.

### 1. Mode Selection
```bash
# Options: "local" | "cloud"
export VSM_MODE=cloud
```

### 2. Common Settings
```bash
# API & Frontend
API_HOST=0.0.0.0
API_PORT=8001
FRONTEND_URL=http://localhost:3000
```

### 3. Local Mode Settings (Existing)
```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_EMBED_MODEL=bge-m3

# MLX VLM
MLX_VLM_BASE_URL=http://localhost:8000

# Weaviate (Local Docker)
WEAVIATE_LOCAL_URL=http://localhost:8080
```

### 4. Cloud Mode Settings (New)
```bash
# Gemini (LLM & VLM)
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_THINKING_BUDGET=4096  # 0=off, 1-24576=tokens

# Jina (Embeddings)
JINA_API_KEY=jina_...
JINA_DENSE_MODEL=jina-embeddings-v4
JINA_COLBERT_MODEL=jina-colbert-v2

# Serverless Worker (The "Hacker" Component)
# Endpoint that runs Jina v4 on GPU and returns multi-vectors
JINA_WORKER_URL=https://api.myserverless.com/v1/embed

# Weaviate Cloud
WEAVIATE_CLOUD_URL=https://xxx.weaviate.cloud
WEAVIATE_CLOUD_API_KEY=xxx
```

---

## Deployment Checklist

### Local Mode
1. [ ] Start Ollama: `ollama serve`
2. [ ] Start Weaviate Docker: `docker-compose up -d`
3. [ ] Start MLX VLM: `python -m mlx_vlm.server --model Qwen/Qwen3-VL-8B-Instruct`
4. [ ] Verify `VSM_MODE=local`

### Cloud Mode
1. [ ] Verify Gemini API key has `generative-language` scope.
2. [ ] Verify Jina API key.
3. [ ] **Deploy Serverless Worker**:
   - Create handler for Jina v4.
   - Deploy to RunPod/Modal.
   - Set `JINA_WORKER_URL`.
4. [ ] Verify Weaviate Cloud instance is running.
5. [ ] Run Cloud Ingestion: `python scripts/cloud_ingest.py ...`
6. [ ] Verify `VSM_MODE=cloud`

---

## Serverless Worker Setup

To enable visual multi-vector search in the cloud, you must deploy the following worker.

### 1. Handler Code (Example for Modal)

```python
# worker.py
import modal
from transformers import AutoModel
import torch

app = modal.App("jina-v4-worker")

image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "pillow", "einops"
)

@app.cls(image=image, gpu="T4")
class JinaModel:
    @modal.enter()
    def load(self):
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v4", 
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to("cuda")

    @modal.method()
    def embed_image(self, image_bytes: bytes):
        # The magic: return_multivector=True
        embeddings = self.model.encode_image(
            image_bytes, 
            return_multivector=True
        )
        return embeddings.tolist()

@app.function(image=image)
@modal.web_endpoint(method="POST")
def embed(item: dict):
    # Wrapper for web request
    ...
```

### 2. Deploy
```bash
modal deploy worker.py
```

### 3. Update Config
```bash
export JINA_WORKER_URL=https://...modal.run/embed
```

---

## Troubleshooting

| Issue | Mode | Check |
|-------|------|-------|
| **Search returns 0 results** | Cloud | Did you run `scripts/cloud_ingest.py`? Cloud/Local collections are separate. |
| **Visual search fails** | Cloud | Check `JINA_WORKER_URL` is reachable and worker is active. |
| **LLM errors** | Cloud | Check `GEMINI_API_KEY` and quota. |
| **"Thinking" not showing** | Cloud | Verify `GEMINI_THINKING_BUDGET` > 0. |
