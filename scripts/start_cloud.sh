#!/bin/bash
# Start VSM in Cloud Mode
# Usage: ./scripts/start_cloud.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "🌩️  VSM Cloud Mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "❌ .env file not found!"
    exit 1
fi

# Set cloud mode
export VSM_MODE=cloud

# Verify required keys
missing_keys=""
[ -z "$GEMINI_API_KEY" ] && missing_keys="$missing_keys GEMINI_API_KEY"
[ -z "$JINA_API_KEY" ] && missing_keys="$missing_keys JINA_API_KEY"
[ -z "$WEAVIATE_URL" ] && missing_keys="$missing_keys WEAVIATE_URL"
[ -z "$WEAVIATE_API_KEY" ] && missing_keys="$missing_keys WEAVIATE_API_KEY"

if [ -n "$missing_keys" ]; then
    echo "❌ Missing keys in .env:$missing_keys"
    exit 1
fi

echo "✅ API keys loaded from .env"

# Optional: GPT-5.1 fallback for reliability
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set (GPT-5.1 fallback disabled)"
else
    echo "✅ GPT-5.1 fallback enabled"
fi

# Activate conda
eval "$(conda shell.bash hook)"
conda activate vsm-hva

# Check frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Clean up on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    [ -n "$API_PID" ] && kill $API_PID 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    wait 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start API in background with uvicorn
echo "🚀 Starting API server..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 2>&1 | sed 's/^/   [API] /' &
API_PID=$!

# Wait for API to be ready (check if port 8001 is listening)
echo "   Waiting for API..."
for i in {1..15}; do
    if curl -s http://localhost:8001/healthz > /dev/null 2>&1; then
        echo "   ✓ API ready"
        break
    fi
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "   ❌ API crashed! Check logs above."
        exit 1
    fi
    sleep 1
done

# Start Frontend (use npx to ensure next is found)
echo "🎨 Starting frontend..."
cd frontend
npx next dev &
FRONTEND_PID=$!
cd "$PROJECT_DIR"

# Wait for frontend to be ready
echo "   Waiting for frontend..."
for i in {1..15}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "   ✓ Frontend ready"
        break
    fi
    sleep 1
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ VSM is running!"
echo ""
echo "  👉 Open: http://localhost:3000"
echo ""
echo "  API Docs: http://localhost:8001/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep script running - wait for both processes
wait $API_PID $FRONTEND_PID

