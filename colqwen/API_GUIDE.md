# Multimodal RAG FastAPI Setup

## Overview

The notebook has been converted to a standalone FastAPI service with automatic OpenAPI documentation.

## Files Created

1. **`rag_api.py`** - Core RAG logic (models, retrieval, generation)
2. **`api_server.py`** - FastAPI REST API server with Swagger docs
3. **`requirements.txt`** - All Python dependencies

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Running the API Server

```bash
# Make sure your .env file is configured with:
# WEAVIATE_URL=your_url
# WEAVIATE_API_KEY=your_key
# HF_TOKEN=your_token (optional)

python api_server.py
```

The API will start on `http://localhost:8000`

## 📚 Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test all endpoints directly in the browser!

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Query (Retrieve + Generate Answer)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "How to connect RS-485?",
    "top_k": 3
  }'
```

Response:
```json
{
  "answer": "Generated answer here...",
  "retrieved_pages": [
    {
      "page_id": 1,
      "asset_manual": "techman",
      "page_number": 5,
      "distance": 0.234
    }
  ]
}
```

### 3. Retrieve Only (No Answer Generation)
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "text": "firmware update",
    "top_k": 5
  }'
```

## Building a Custom Frontend

### Example: Simple HTML/JavaScript Frontend

```html
<!DOCTYPE html>
<html>
<head>
    <title>PDF RAG Chat</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
        input { width: 70%; padding: 10px; }
        button { padding: 10px 20px; }
        #answer { margin-top: 20px; padding: 15px; background: #f0f0f0; }
        #pages { margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Ask Questions About PDFs</h1>
    <input type="text" id="query" placeholder="Enter your question">
    <button onclick="askQuestion()">Ask</button>
    <div id="answer"></div>
    <div id="pages"></div>

    <script>
        async function askQuestion() {
            const query = document.getElementById('query').value;
            const response = await fetch('http://localhost:8000/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: query, top_k: 3})
            });
            const data = await response.json();
            document.getElementById('answer').innerHTML = 
                '<h3>Answer:</h3><p>' + data.answer + '</p>';
            document.getElementById('pages').innerHTML = 
                '<h3>Sources:</h3><ul>' + 
                data.retrieved_pages.map(p => 
                    `<li>${p.asset_manual} - Page ${p.page_number} (distance: ${p.distance.toFixed(3)})</li>`
                ).join('') + '</ul>';
        }
    </script>
</body>
</html>
```

Save this as `index.html` and open it in your browser!

### Example: React Frontend

```javascript
import React, { useState } from 'react';

function RAGChat() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: query, top_k: 3 })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '800px', margin: '50px auto', fontFamily: 'Arial' }}>
      <h1>PDF RAG Chat</h1>
      <div>
        <input 
          value={query} 
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question"
          style={{ width: '70%', padding: '10px' }}
        />
        <button onClick={askQuestion} disabled={loading} style={{ padding: '10px 20px', marginLeft: '10px' }}>
          {loading ? 'Loading...' : 'Ask'}
        </button>
      </div>
      {result && (
        <div style={{ marginTop: '20px' }}>
          <div style={{ padding: '15px', background: '#f0f0f0' }}>
            <h3>Answer:</h3>
            <p>{result.answer}</p>
          </div>
          <div style={{ marginTop: '10px' }}>
            <h3>Sources:</h3>
            <ul>
              {result.retrieved_pages.map((page, i) => (
                <li key={i}>
                  {page.asset_manual} - Page {page.page_number} 
                  (distance: {page.distance.toFixed(3)})
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default RAGChat;
```

## Deployment

### Option 1: Local Testing
```bash
python api_server.py
```

### Option 2: Production with Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Option 3: Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "api_server.py"]
```

## Troubleshooting

### "No module named 'torch'" Error
Make sure you're in the correct environment:
```bash
# Use the vsm-hva environment that has all ML packages
conda activate vsm-hva
pip install fastapi uvicorn
python api_server.py
```

### Port Already in Use
Change the port in `api_server.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=9000)  # Use port 9000 instead
```

## Next Steps

1. **Start the API**: `python api_server.py`
2. **Open Swagger docs**: http://localhost:8000/docs
3. **Test in browser**: Try the endpoints interactively
4. **Build your frontend**: Use the examples above or create your own

## API Features

✅ **Automatic OpenAPI docs** - No manual documentation needed  
✅ **Type validation** - Request/response models with Pydantic  
✅ **CORS enabled** - Works with any frontend framework  
✅ **Interactive testing** - Swagger UI for easy testing  
✅ **Fast & async** - Built on modern async Python
