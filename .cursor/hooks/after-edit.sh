#!/bin/bash
# afterFileEdit hook - track edits to agent-critical files

# Read JSON input from stdin
input=$(cat)

# Extract file path
file_path=$(echo "$input" | grep -o '"file_path":"[^"]*"' | cut -d'"' -f4)

# Define critical files that affect agent behavior
CRITICAL_PATTERNS="agent.py|llm.py|search.py|search_tools.py|environment.py|weaviate_ingest"

# Log all edits
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Edited: $file_path" >> /tmp/cursor-edits.log

# If critical file, also log to a separate file for progress tracking
if echo "$file_path" | grep -qE "$CRITICAL_PATTERNS"; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CRITICAL: $file_path" >> /tmp/cursor-critical-edits.log
fi

# Exit successfully (afterFileEdit doesn't need output)
exit 0
