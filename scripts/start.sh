#!/bin/bash
# VSM Start Script - Launches all services for the Visual Search Manual system
# Usage: ./scripts/start.sh [--no-ollama] [--no-frontend]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root (script is in scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
SKIP_OLLAMA=false
SKIP_FRONTEND=false
for arg in "$@"; do
    case $arg in
        --no-ollama) SKIP_OLLAMA=true ;;
        --no-frontend) SKIP_FRONTEND=true ;;
        --help|-h)
            echo "Usage: ./scripts/start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-ollama    Skip Ollama configuration/restart"
            echo "  --no-frontend  Skip frontend dev server"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          VSM - Visual Search Manual Start Script           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# 1. Check Prerequisites
# =============================================================================
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "vsm-hva" ]]; then
    echo -e "${RED}Error: Please activate conda environment first:${NC}"
    echo "  conda activate vsm-hva"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Conda environment: vsm-hva"

# Check Docker (for Weaviate)
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not found. Please install Docker.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Docker available"

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Error: Ollama not found. Install from https://ollama.ai${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Ollama available"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Node.js $(node --version)"

echo ""

# =============================================================================
# 2. Configure and Start Ollama Server (Native, not Docker)
# =============================================================================
if [[ "$SKIP_OLLAMA" == false ]]; then
    echo -e "${YELLOW}[2/6] Configuring Native Ollama for gpt-oss:120b (65GB)...${NC}"
    
    # Kill any Docker Ollama containers that might conflict
    echo -e "  Cleaning up conflicting Ollama instances..."
    for container in $(docker ps --format '{{.Names}}' 2>/dev/null | grep -i ollama); do
        docker stop "$container" 2>/dev/null && docker rm "$container" 2>/dev/null
        echo -e "  ${GREEN}✓${NC} Stopped Docker container: $container"
    done
    
    # Kill the Ollama GUI app if running (we use ollama serve instead)
    pkill -f "Ollama.app" 2>/dev/null || true
    pkill -f "ollama serve" 2>/dev/null || true
    
    # Wait for port to be released
    sleep 2
    
    # Check if port 11434 is still in use
    if lsof -ti:11434 > /dev/null 2>&1; then
        echo -e "  ${YELLOW}!${NC} Port 11434 still in use, force killing..."
        lsof -ti:11434 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Set environment variables for memory optimization
    export OLLAMA_FLASH_ATTENTION=1
    export OLLAMA_KV_CACHE_TYPE=q8_0
    export OLLAMA_NUM_PARALLEL=1
    # Bind to 0.0.0.0 so Docker containers can access via host.docker.internal
    export OLLAMA_HOST="0.0.0.0:11434"
    
    echo -e "  ${GREEN}✓${NC} OLLAMA_FLASH_ATTENTION=1"
    echo -e "  ${GREEN}✓${NC} OLLAMA_KV_CACHE_TYPE=q8_0"
    echo -e "  ${GREEN}✓${NC} OLLAMA_HOST=0.0.0.0:11434"
    
    # Start ollama serve in background (NOT the GUI app)
    echo -e "  Starting ollama serve..."
    nohup ollama serve > /tmp/vsm-ollama.log 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    
    # Verify Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Native Ollama running on port 11434"
        echo -e "      PID: $OLLAMA_PID | Log: /tmp/vsm-ollama.log"
    else
        echo -e "  ${RED}✗${NC} Ollama failed to start. Check /tmp/vsm-ollama.log"
        cat /tmp/vsm-ollama.log | tail -10
        exit 1
    fi
    
    # Check required models
    if ollama list | grep -q "gpt-oss:120b"; then
        echo -e "  ${GREEN}✓${NC} gpt-oss:120b model available (LLM)"
    else
        echo -e "  ${YELLOW}!${NC} gpt-oss:120b not found. Pull it with:"
        echo "      ollama pull gpt-oss:120b"
    fi
    
    if ollama list | grep -q "bge-m3"; then
        echo -e "  ${GREEN}✓${NC} bge-m3 model available (embeddings)"
    else
        echo -e "  ${YELLOW}!${NC} bge-m3 not found. Pull it with:"
        echo "      ollama pull bge-m3"
    fi
else
    echo -e "${YELLOW}[2/6] Skipping Ollama configuration (--no-ollama)${NC}"
fi
echo ""

# =============================================================================
# 3. Start Weaviate (Docker) - Must use THIS project's container
# =============================================================================
echo -e "${YELLOW}[3/6] Starting Weaviate...${NC}"

# Stop any Weaviate containers from OTHER projects (they have wrong Ollama config)
for container in $(docker ps --format '{{.Names}}' 2>/dev/null | grep -i weaviate | grep -v "vsm-v02"); do
    echo -e "  ${YELLOW}!${NC} Stopping conflicting Weaviate: $container"
    docker stop "$container" 2>/dev/null && docker rm "$container" 2>/dev/null
done

# Check if OUR Weaviate container is running
if docker ps --format '{{.Names}}' | grep -q "vsm-v02-weaviate"; then
    echo -e "  ${GREEN}✓${NC} vsm-v02-weaviate-1 already running"
else
    # Start with docker compose from THIS project
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        cd "$PROJECT_ROOT"
        docker compose up -d 2>/dev/null || docker-compose up -d
        sleep 5
        echo -e "  ${GREEN}✓${NC} Weaviate started via docker-compose"
    else
        echo -e "  ${RED}✗${NC} No docker-compose.yml found"
        exit 1
    fi
fi

# Verify Weaviate
if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Weaviate ready on port 8080"
else
    echo -e "  ${RED}✗${NC} Weaviate not responding on port 8080"
    echo "      Check: docker logs vsm-v02-weaviate-1"
    exit 1
fi
echo ""

# =============================================================================
# 4. Kill existing processes on our ports
# =============================================================================
echo -e "${YELLOW}[4/6] Cleaning up existing processes...${NC}"

# Kill any process on port 8001 (API)
if lsof -ti:8001 > /dev/null 2>&1; then
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Cleared port 8001"
fi

# Kill any process on port 3000 (Frontend)
if [[ "$SKIP_FRONTEND" == false ]]; then
    if lsof -ti:3000 > /dev/null 2>&1; then
        lsof -ti:3000 | xargs kill -9 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Cleared port 3000"
    fi
fi

sleep 1
echo ""

# =============================================================================
# 5. Start Backend API
# =============================================================================
echo -e "${YELLOW}[5/6] Starting Backend API...${NC}"

# Set environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Start uvicorn in background
cd "$PROJECT_ROOT"
nohup uvicorn api.main:app --host 127.0.0.1 --port 8001 > /tmp/vsm-api.log 2>&1 &
API_PID=$!

# Wait for API to start
sleep 3

if curl -s http://localhost:8001/docs > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} API running on http://localhost:8001"
    echo -e "  ${GREEN}✓${NC} API docs: http://localhost:8001/docs"
    echo -e "      PID: $API_PID | Log: /tmp/vsm-api.log"
else
    echo -e "  ${RED}✗${NC} API failed to start. Check /tmp/vsm-api.log"
    cat /tmp/vsm-api.log | tail -20
    exit 1
fi
echo ""

# =============================================================================
# 6. Start Frontend
# =============================================================================
if [[ "$SKIP_FRONTEND" == false ]]; then
    echo -e "${YELLOW}[6/6] Starting Frontend...${NC}"
    
    cd "$PROJECT_ROOT/frontend"
    
    # Install dependencies if needed
    if [[ ! -d "node_modules" ]]; then
        echo -e "  Installing npm dependencies..."
        npm install --silent
    fi
    
    # Start Next.js dev server in background
    nohup npm run dev > /tmp/vsm-frontend.log 2>&1 &
    FRONTEND_PID=$!
    
    # Wait for frontend to start
    sleep 5
    
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Frontend running on http://localhost:3000"
        echo -e "      PID: $FRONTEND_PID | Log: /tmp/vsm-frontend.log"
    else
        echo -e "  ${YELLOW}!${NC} Frontend starting... (check /tmp/vsm-frontend.log)"
    fi
else
    echo -e "${YELLOW}[6/6] Skipping Frontend (--no-frontend)${NC}"
fi

cd "$PROJECT_ROOT"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    All Services Started                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}Weaviate${NC}:    http://localhost:8080"
echo -e "  ${GREEN}Ollama${NC}:      http://localhost:11434"
echo -e "  ${GREEN}API${NC}:         http://localhost:8001"
echo -e "  ${GREEN}API Docs${NC}:    http://localhost:8001/docs"
if [[ "$SKIP_FRONTEND" == false ]]; then
    echo -e "  ${GREEN}Frontend${NC}:    http://localhost:3000"
fi
echo ""
echo -e "${YELLOW}Quick Test:${NC}"
echo "  curl -N 'http://localhost:8001/agentic_search?query=voltage' | head -5"
echo ""
echo -e "${YELLOW}Logs:${NC}"
if [[ "$SKIP_OLLAMA" == false ]]; then
    echo "  tail -f /tmp/vsm-ollama.log   # Ollama"
fi
echo "  tail -f /tmp/vsm-api.log      # Backend API"
if [[ "$SKIP_FRONTEND" == false ]]; then
    echo "  tail -f /tmp/vsm-frontend.log # Frontend"
fi
echo ""
echo -e "${YELLOW}Stop all:${NC}"
echo "  ./scripts/stop.sh"
echo ""

