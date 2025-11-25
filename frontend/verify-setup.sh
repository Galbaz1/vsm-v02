#!/bin/bash
# Quick verification script for frontend setup

echo "🔍 Verifying frontend setup..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found"
    exit 1
fi
echo "✅ Node.js $(node --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found"
    exit 1
fi
echo "✅ npm $(npm --version)"

# Check dependencies
if [ ! -d "node_modules" ]; then
    echo "⚠️  node_modules not found. Run: npm install"
else
    echo "✅ Dependencies installed"
fi

# Check environment
if [ ! -f ".env.local" ]; then
    echo "⚠️  .env.local not found. Copy from .env.example"
else
    echo "✅ Environment file exists"
fi

# Check API connectivity (if backend is running)
API_URL="${NEXT_PUBLIC_API_BASE_URL:-http://localhost:8001}"
if curl -s "$API_URL/healthz" > /dev/null 2>&1; then
    echo "✅ Backend API is reachable at $API_URL"
else
    echo "⚠️  Backend API not reachable at $API_URL (make sure FastAPI is running)"
fi

echo ""
echo "🎉 Setup verification complete!"
echo ""
echo "To start development:"
echo "  npm run dev"
echo ""
echo "Make sure the FastAPI backend is running on port 8001"

