/**
 * AfterFileEdit Hook - Tracks edits for debugging
 * 
 * PURPOSE: Create audit trail of what files were changed
 * - Logs all edits to /tmp/vsm-edits.log
 * - Warns if agent-critical files changed (suggests re-running benchmark)
 * 
 * DOES NOT: 
 * - Linting (Cursor does this)
 * - Syntax checking (Cursor does this)
 * - Formatting (use prettier/eslint config instead)
 */

import type { AfterFileEditPayload, AfterFileEditResponse } from "cursor-hooks";
import { appendFileSync, existsSync, mkdirSync } from "fs";
import { basename, dirname } from "path";

const EDIT_LOG = "/tmp/vsm-edits.log";

// Files that are critical to agent behavior - suggest benchmark re-run
const AGENT_CRITICAL_PATTERNS = [
  "api/services/agent.py",
  "api/services/llm.py",
  "api/services/environment.py",
  "api/services/tools/",
  "api/services/search.py",
];

async function main() {
  const payload: AfterFileEditPayload = await Bun.stdin.json();
  const { path: filePath, newContent } = payload;

  if (!filePath) {
    console.log(JSON.stringify({}));
    return;
  }

  const timestamp = new Date().toISOString().slice(0, 19).replace("T", " ");
  const fileName = basename(filePath);
  const lineCount = newContent?.split("\n").length || 0;

  // Log the edit
  const logLine = `${timestamp} ${filePath} (${lineCount} lines)\n`;
  appendFileSync(EDIT_LOG, logLine);

  // Check if this is an agent-critical file
  const isCritical = AGENT_CRITICAL_PATTERNS.some(pattern => 
    filePath.includes(pattern)
  );

  const response: AfterFileEditResponse = {};

  if (isCritical) {
    response.message = `📝 Agent file edited: ${fileName}\n` +
      `Consider re-running benchmark:\n` +
      `  python scripts/run_benchmark.py --output results.json`;
  }

  console.log(JSON.stringify(response));
}

main().catch(console.error);

