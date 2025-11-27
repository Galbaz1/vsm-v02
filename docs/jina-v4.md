Yep, you can hit **jina-embeddings-v4** through a nice, boring HTTP API. It’s very “OpenAI-style”, so it plugs into existing RAG stacks easily.

I’ll show you:

1. The basic HTTP call
2. How to pick **model / adapter / dimensions**
3. Text vs image (multimodal) payloads
4. A small end-to-end example in Python

---

## 1. Endpoint + auth

Jina exposes a **Universal Embeddings API** at something like:

* Base: their Universal API (Swagger shows `/v1/embeddings`)

Conceptually:

```http
POST https://api.jina.ai/v1/embeddings
Authorization: Bearer YOUR_JINA_API_KEY
Content-Type: application/json
```

You generate the API key in their dashboard and drop it into `Authorization: Bearer …`.

---

## 2. Minimal embedding request (text)

The API schema describes an `EmbeddingInput` where you give:

* `model`: e.g. `"jina-embeddings-v4"` (or a specific adapter variant)
* `input`: your text(s)
* optional extras like `task` and `dimensions` (Matryoshka compression).

Example `curl`:

```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": [
      "Find invoices about ACME Corp from Q2.",
      "Another sentence to embed."
    ],
    "task": "retrieval.passage",
    "dimensions": 1024
  }'
```

The response will look conceptually like:

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.0123, -0.0456, ...],
      "usage": {...}
    },
    {
      "index": 1,
      "embedding": [...]
    }
  ],
  "model": "jina-embeddings-v4",
  "usage": {...}
}
```

Key knobs:

* `task`: e.g. `retrieval.passage`, `text-matching`, `classification`, `code` (task-specific adapters).
* `dimensions`: can be 128–2048 (Matryoshka representation), default 2048.

---

## 3. Multimodal: sending images / visual docs

Because v4 is **multimodal**, the API supports **image inputs** as well as text. The OpenAPI spec exposes an `ImageEmbeddingInput` and mixed `TextOrImageDoc` types.

You usually send images as **base64-encoded** bytes or URLs, depending on the helper you’re using. Rough pattern:

```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": [
      {
        "type": "image",
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA..."
      }
    ],
    "task": "retrieval.passage",
    "dimensions": 1024
  }'
```

Then you store that vector in your DB alongside text embeddings. Because it’s a **shared embedding space**, you can query with text and retrieve images or scanned PDF pages.

Many integrations (Pinecone, Qdrant) now have example code where they call this API and store the vectors directly, especially for `jina-embeddings-v4`.

---

## 4. Multi-vector (late interaction) endpoint

If you want ColBERT-style **multi-vector** outputs instead of a single pooled embedding, the Universal API exposes:

* `POST /v1/multi-vector` and `/v1/multi-embeddings` for ColBERT-style outputs.

Conceptually:

```bash
curl https://api.jina.ai/v1/multi-embeddings \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": ["Long passage of text..."],
    "task": "retrieval.passage"
  }'
```

The response will contain multiple vectors per input (token-level / span-level) suitable for MaxSim scoring in a late-interaction index.

You’d then:

* store those token vectors in a multi-vector index (e.g. Vespa, ColBERT-style layer, or Qdrant’s experimental multi-vector support).
* at query time, also ask `/v1/multi-embeddings` for query multi-vectors, then run MaxSim.

---

## 5. Small Python example (single-vector RAG-ish)

Here’s a minimal Python client using `requests` that will feel very OpenAI-like:

```python
import os
import requests

JINA_API_KEY = os.environ["JINA_API_KEY"]
BASE_URL = "https://api.jina.ai/v1"

def embed_texts(texts, model="jina-embeddings-v4",
                task="retrieval.passage", dimensions=1024):
    payload = {
        "model": model,
        "input": texts,
        "task": task,
        "dimensions": dimensions,
    }
    resp = requests.post(
        f"{BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # return list of (text, embedding)
    return [
        (texts[item["index"]], item["embedding"])
        for item in data["data"]
    ]

if __name__ == "__main__":
    texts = [
        "How do I file a reimbursement for travel expenses?",
        "GPU utilization troubleshooting guide",
    ]
    embedded = embed_texts(texts)
    print(len(embedded), "embeddings")
    print("First vector length:", len(embedded[0][1]))
```

Drop those vectors into your vector DB of choice, and you’re in RAG-land.

---

## 6. Rerank & bulk endpoints (bonus)

The same Universal API also has:

* `POST /v1/rerank` – cross-encoder like reranking of candidates.
* `POST /v1/bulk-embeddings` – async bulk jobs for mass indexing.

Those are handy for:

* **Ingestion at scale** (bulk embeddings).
* **Second-stage rerank** on top-k documents in your RAG pipeline.

---

So the punchline:

* Use `/v1/embeddings` for normal dense embeddings (text or image).
* Use `/v1/multi-embeddings` for late-interaction stuff.
* Pick `task` (`retrieval.passage`, `text-matching`, `code`, etc.) and `dimensions` according to your use case.
* Treat it like an OpenAI-style embedding API and wire it into your existing RAG plumbing.
