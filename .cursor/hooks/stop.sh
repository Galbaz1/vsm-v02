#!/bin/bash
# stop hook - session summary and maintenance reminders

# Read JSON input from stdin
input=$(cat)

# Extract status
status=$(echo "$input" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

# Log completion
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Agent stopped: $status" >> /tmp/cursor-sessions.log

# Check if critical files were edited this session
if [ -f /tmp/cursor-critical-edits.log ]; then
  RECENT=$(tail -5 /tmp/cursor-critical-edits.log 2>/dev/null)
  if [ -n "$RECENT" ]; then
    # Return a follow-up message reminding to update docs
    cat << EOF
{
  "followup_message": "Critical files edited. Please update:\n1. .cursor/rules/progress.mdc - Add fix to Recent Fixes table\n2. TODO.md - Move completed items, update In Progress\n3. Run benchmark if applicable: python scripts/run_benchmark.py"
}
EOF
    # Clear the critical edits log for next session
    > /tmp/cursor-critical-edits.log
    exit 0
  fi
fi

# No follow-up needed
echo '{}'
