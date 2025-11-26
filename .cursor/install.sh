#!/bin/bash
#
# Cursor AI Toolkit Installer
#
# This script installs the Cursor AI sub-agents toolkit into a new project.
# It copies the necessary files and creates the required directory structure.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/yourrepo/cursor-toolkit/main/install.sh | bash
#   # OR
#   ./path/to/.cursor/install.sh /path/to/new/project
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Cursor AI Sub-Agents Toolkit Installer${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Get source directory (where this script lives)
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get target directory
if [ -n "$1" ]; then
    TARGET_DIR="$1"
else
    TARGET_DIR="$(pwd)"
fi

echo -e "\n${YELLOW}Installing to: ${TARGET_DIR}${NC}\n"

# Create directories
echo "Creating directories..."
mkdir -p "${TARGET_DIR}/.cursor/agents"
mkdir -p "${TARGET_DIR}/.cursor/hooks"
mkdir -p "${TARGET_DIR}/.cursor/rules"
mkdir -p "${TARGET_DIR}/logs/query_traces"
mkdir -p "${TARGET_DIR}/scripts"

# Copy agents package
echo "Copying agents package..."
cp -r "${SOURCE_DIR}/agents/"* "${TARGET_DIR}/.cursor/agents/" 2>/dev/null || true

# Copy hooks
echo "Copying hooks..."
cp "${SOURCE_DIR}/hooks/"*.ts "${TARGET_DIR}/.cursor/hooks/" 2>/dev/null || true
cp "${SOURCE_DIR}/hooks/package.json" "${TARGET_DIR}/.cursor/hooks/" 2>/dev/null || true
cp "${SOURCE_DIR}/hooks/tsconfig.json" "${TARGET_DIR}/.cursor/hooks/" 2>/dev/null || true
cp "${SOURCE_DIR}/hooks.json" "${TARGET_DIR}/.cursor/" 2>/dev/null || true

# Copy rules (but don't overwrite existing)
echo "Copying rules (preserving existing)..."
for rule in "${SOURCE_DIR}/rules/"*.mdc; do
    rulename=$(basename "$rule")
    if [ ! -f "${TARGET_DIR}/.cursor/rules/${rulename}" ]; then
        cp "$rule" "${TARGET_DIR}/.cursor/rules/"
        echo "  + ${rulename}"
    else
        echo "  ~ ${rulename} (exists, skipped)"
    fi
done

# Create wrapper script
echo "Creating analyze script..."
cat > "${TARGET_DIR}/scripts/analyze_with_llm.py" << 'EOF'
#!/usr/bin/env python
"""LLM-Powered Trace Analyzer - See .cursor/agents/ for implementation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cursor"))
from agents.cli import main
if __name__ == "__main__": main()
EOF
chmod +x "${TARGET_DIR}/scripts/analyze_with_llm.py"

# Install hook dependencies
echo "Installing hook dependencies..."
cd "${TARGET_DIR}/.cursor/hooks" && bun install 2>/dev/null || npm install 2>/dev/null || echo "  (skipped - install bun or npm)"

# Check for required env vars
echo ""
echo -e "${YELLOW}Checking environment...${NC}"
if [ -f "${TARGET_DIR}/.env" ]; then
    if grep -q "GEMINI_API_KEY" "${TARGET_DIR}/.env"; then
        echo -e "  ${GREEN}✓${NC} GEMINI_API_KEY found"
    else
        echo -e "  ${RED}✗${NC} GEMINI_API_KEY missing - add to .env"
    fi
    if grep -q "OPENAI_API_KEY" "${TARGET_DIR}/.env"; then
        echo -e "  ${GREEN}✓${NC} OPENAI_API_KEY found"
    else
        echo -e "  ${RED}✗${NC} OPENAI_API_KEY missing - add to .env"
    fi
else
    echo -e "  ${RED}✗${NC} No .env file found"
    echo "  Create .env with GEMINI_API_KEY and OPENAI_API_KEY"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit .cursor/agents/config.py to configure your project's critical files"
echo "  2. Ensure GEMINI_API_KEY and OPENAI_API_KEY are in .env"
echo "  3. Run: python scripts/analyze_with_llm.py --help"
echo ""

