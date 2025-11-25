# Deployment Checklist

## ✅ Required Files (Must Copy)

```
colqwen/
├── api/                    # Complete folder
│   ├── __init__.py
│   ├── server.py
│   └── rag.py
├── .env                    # Your credentials
└── requirements.txt        # Dependencies
```

## 📋 Optional Files

```
├── data/                   # Only if using the same PDFs
│   ├── techman.pdf
│   ├── uk_firmware.pdf  
│   └── benchmark.json
├── README.md              # Documentation
└── API_GUIDE.md          # Usage guide
```

## ❌ NOT Needed

```
notebooks/                  # Original Jupyter notebooks (not needed for API)
```

---

## 🚀 Deployment Steps

### 1. Copy Files

```bash
# Minimum deployment
scp -r api/ user@server:/path/to/project/
scp .env requirements.txt user@server:/path/to/project/

# With PDFs (if needed)
scp -r data/ user@server:/path/to/project/
```

### 2. On Target Server

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env file
nano .env  # Add your WEAVIATE_URL, WEAVIATE_API_KEY, HF_TOKEN

# Run the server
python -m api.server
```

---

## 📦 Create Portable Archive

### Option 1: Minimal (API only)

```bash
# From colqwen directory
tar -czf colqwen-api.tar.gz api/ .env requirements.txt README.md

# Deploy anywhere
scp colqwen-api.tar.gz user@server:
ssh user@server
tar -xzf colqwen-api.tar.gz
pip install -r requirements.txt
python -m api.server
```

### Option 2: Complete (with data)

```bash
tar -czf colqwen-complete.tar.gz api/ data/ .env requirements.txt README.md API_GUIDE.md
```

---

## 🐳 Docker Deployment (Recommended)

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api/ ./api/
COPY .env .

# Expose port
EXPOSE 8002

# Run server
CMD ["python", "-m", "api.server"]
```

Build and run:

```bash
docker build -t colqwen-api .
docker run -p 8002:8002 --env-file .env colqwen-api
```

---

## ☁️ Cloud Deployment

### AWS EC2 / Google Cloud / Azure VM

1. Copy files: `api/`, `.env`, `requirements.txt`
2. Install Python 3.12+
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python -m api.server`
5. (Optional) Use systemd or supervisor for auto-restart

### Serverless (AWS Lambda / Google Cloud Functions)

Not recommended due to:
- Large model files (multiple GB)
- Long cold start time
- Memory requirements

Better suited for VM or container deployment.

---

## 🔑 Important Notes

### .env File Security

**Never commit `.env` to git!** It contains secrets. Use:

```bash
# .gitignore
.env
*.env
```

For production, use:
- Environment variables
- Secret managers (AWS Secrets Manager, Google Secret Manager)
- Vault

### GPU Requirements

The models work on:
- ✅ Apple Silicon (MPS) - What you're using
- ✅ NVIDIA GPUs (CUDA)
- ✅ CPU (slower)

If deploying to cloud:
- AWS: Use instances with GPU (g4dn, p3, etc.)
- Google Cloud: Use GPU-enabled instances
- Azure: Use NC-series VMs

### Memory Requirements

Minimum:
- RAM: 16GB (32GB+ recommended)
- Disk: 20GB for models

---

## 📝 Quick Deployment Script

Create `deploy.sh`:

```bash
#!/bin/bash
set -e

echo "📦 Creating deployment package..."
tar -czf deploy.tar.gz api/ .env requirements.txt README.md

echo "📤 Uploading to server..."
scp deploy.tar.gz user@your-server.com:~/

echo "🚀 Deploying on server..."
ssh user@your-server.com << 'EOF'
  cd ~
  tar -xzf deploy.tar.gz
  pip install -r requirements.txt
  
  # Kill old process if running
  pkill -f "api.server" || true
  
  # Start new process in background
  nohup python -m api.server > server.log 2>&1 &
  
  echo "✅ Server started!"
EOF

echo "✅ Deployment complete!"
echo "🔗 API: http://your-server.com:8002"
```

Make executable: `chmod +x deploy.sh`
Run: `./deploy.sh`
