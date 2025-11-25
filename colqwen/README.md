# Multimodal RAG over PDFs

A FastAPI-based system for querying PDF documents using ColQwen2.5 embeddings and Qwen2.5-VL for answer generation.

## 📁 Project Structure

```
colqwen/
├── api/                    # API package
│   ├── __init__.py
│   ├── server.py          # FastAPI REST server
│   └── rag.py             # Core RAG logic
├── data/                   # PDF documents and benchmarks
│   ├── techman.pdf
│   ├── uk_firmware.pdf
│   └── benchmark.json
├── notebooks/              # Original Jupyter notebooks
│   ├── Multi_Vector_ColQwen_VSM.ipynb
│   └── multi_vector_colqwen_vsm.py
├── .env                    # Environment variables (not in git)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
conda activate vsm-hva
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your credentials:

```env
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token  # optional
```

### 3. Run the API Server

```bash
python -m api.server
```

The server will start on http://localhost:8002

## 📚 API Documentation

Once the server is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

### Query (Retrieve + Generate Answer)
```bash
POST /query
Content-Type: application/json

{
  "text": "How to connect RS-485?",
  "top_k": 3
}
```

### Retrieve Only (No Answer Generation)
```bash
POST /retrieve
Content-Type: application/json

{
  "text": "firmware update",
  "top_k": 5
}
```

## 📝 Example Usage

### Using curl

```bash
# Health check
curl http://localhost:8002/health

# Ask a question
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"text": "How to connect RS-485?", "top_k": 3}'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8002/query",
    json={"text": "How to connect RS-485?", "top_k": 3}
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['retrieved_pages']}")
```

### Using JavaScript

```javascript
const response = await fetch('http://localhost:8002/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'How to connect RS-485?', top_k: 3 })
});

const data = await response.json();
console.log('Answer:', data.answer);
console.log('Sources:', data.retrieved_pages);
```

## 🏗️ Architecture

1. **ColQwen2.5** - Generates multi-vector embeddings for PDF pages and queries
2. **Weaviate** - Vector database for similarity search
3. **Qwen2.5-VL** - Vision-language model for answer generation
4. **FastAPI** - REST API server with automatic documentation

## 🎯 Features

✅ Multimodal PDF document retrieval  
✅ Multi-vector embeddings with ColQwen2.5  
✅ Answer generation with Qwen2.5-VL  
✅ FastAPI with automatic OpenAPI docs  
✅ CORS enabled for frontend integration  
✅ Type-safe with Pydantic models  

## 📦 Dependencies

- PyTorch (MPS support for Apple Silicon)
- Transformers
- ColPali Engine
- Weaviate Client
- FastAPI + Uvicorn
- Qwen VL Utils

See `requirements.txt` for complete list.

## 🔧 Development

### Running in Development Mode

```bash
# Auto-reload on code changes
uvicorn api.server:app --reload --port 8002
```

### Testing the API

Use the interactive Swagger UI at `/docs` to test endpoints directly in your browser.

## 📄 License

[Your License Here]
