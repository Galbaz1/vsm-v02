#!/bin/bash
# beforeSubmitPrompt hook - detect debugging context

# Read JSON from stdin
input=$(cat)

# Check for looping traces in the last 10 minutes
TRACE_DIR="logs/query_traces"
LOOPS=0

if [ -d "$TRACE_DIR" ]; then
  for f in $(find "$TRACE_DIR" -name "*.json" -mmin -10 2>/dev/null); do
    if grep -q '"max_iterations"' "$f" 2>/dev/null; then
      LOOPS=$((LOOPS + 1))
    fi
  done
fi

# If recent loops exist, add a user message
if [ $LOOPS -gt 0 ]; then
  cat << EOF
{
  "continue": true,
  "user_message": "⚠️ $LOOPS recent queries hit max iterations. Use: python scripts/analyze_with_llm.py --gemini-only <trace_id>"
}
EOF
else
  # Allow prompt to continue without modification
  echo '{"continue": true}'
fi

