#!/bin/bash
# VSM Stop Script - Stops all VSM services
# Usage: ./scripts/stop.sh [--all]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

STOP_ALL=false
for arg in "$@"; do
    case $arg in
        --all) STOP_ALL=true ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              VSM - Stopping Services                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop API (port 8001)
echo -e "${YELLOW}Stopping API...${NC}"
if lsof -ti:8001 > /dev/null 2>&1; then
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} API stopped (port 8001)"
else
    echo -e "  ${YELLOW}-${NC} API not running"
fi

# Stop Frontend (port 3000)
echo -e "${YELLOW}Stopping Frontend...${NC}"
if lsof -ti:3000 > /dev/null 2>&1; then
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Frontend stopped (port 3000)"
else
    echo -e "  ${YELLOW}-${NC} Frontend not running"
fi

# Stop Weaviate and Ollama only if --all flag
if [[ "$STOP_ALL" == true ]]; then
    echo -e "${YELLOW}Stopping Weaviate (Docker)...${NC}"
    docker compose down 2>/dev/null || docker-compose down 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Weaviate stopped"
    
    echo -e "${YELLOW}Stopping Ollama server...${NC}"
    pkill -f "ollama serve" 2>/dev/null || true
    pkill -f "Ollama.app" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Ollama stopped"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo "To restart: ./scripts/start.sh"
if [[ "$STOP_ALL" == false ]]; then
    echo ""
    echo -e "${YELLOW}Note:${NC} Weaviate and Ollama still running."
    echo "Use --all to stop everything: ./scripts/stop.sh --all"
fi

