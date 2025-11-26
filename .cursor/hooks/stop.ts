/**
 * Stop Hook - Captures session metadata when agent task completes
 * 
 * PURPOSE: Create audit trail for debugging complex agentic workflows
 * - Saves session info to logs/cursor_sessions/
 * - Links with our backend query traces (logs/query_traces/)
 * - Shows notification with quick stats
 * 
 * DOES NOT: Duplicate what Cursor already does (linting, diffs, etc.)
 */

import type { StopPayload, StopResponse } from "cursor-hooks";
import { existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from "fs";
import { join } from "path";

const SESSION_LOG_DIR = "logs/cursor_sessions";
const QUERY_TRACE_DIR = "logs/query_traces";

interface SessionLog {
  conversationId: string;
  timestamp: string;
  prompt: string;
  durationEstimate: string;
  queryTracesInWindow: string[];
  editedFiles: string[];
  summary: {
    traceCount: number;
    loopingQueries: number;
  };
}

async function main() {
  const payload: StopPayload = await Bun.stdin.json();
  const { conversationId, prompt } = payload;

  const timestamp = new Date().toISOString();
  const sessionId = conversationId?.slice(0, 12) || `session-${Date.now()}`;

  // Find query traces created in last 10 minutes (likely from this session)
  const recentTraces: string[] = [];
  let loopingCount = 0;

  if (existsSync(QUERY_TRACE_DIR)) {
    const tenMinAgo = Date.now() - 10 * 60 * 1000;
    const files = readdirSync(QUERY_TRACE_DIR).filter(f => f.endsWith(".json"));
    
    for (const file of files) {
      const filePath = join(QUERY_TRACE_DIR, file);
      const stat = statSync(filePath);
      
      if (stat.mtimeMs > tenMinAgo) {
        recentTraces.push(file.replace(".json", ""));
        
        // Check if this trace hit max iterations (looping)
        try {
          const trace = JSON.parse(await Bun.file(filePath).text());
          if (trace.final_outcome === "max_iterations") {
            loopingCount++;
          }
        } catch {}
      }
    }
  }

  // Read recent file edits from our log
  const editedFiles: string[] = [];
  const editLogPath = "/tmp/vsm-edits.log";
  if (existsSync(editLogPath)) {
    const content = await Bun.file(editLogPath).text();
    const lines = content.trim().split("\n").slice(-20);
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString().slice(0, 19);
    
    for (const line of lines) {
      const [ts, ...pathParts] = line.split(" ");
      if (ts >= fiveMinAgo) {
        editedFiles.push(pathParts.join(" "));
      }
    }
  }

  // Create session log
  const sessionLog: SessionLog = {
    conversationId: conversationId || "unknown",
    timestamp,
    prompt: prompt?.slice(0, 200) || "",
    durationEstimate: "unknown",
    queryTracesInWindow: recentTraces,
    editedFiles: [...new Set(editedFiles)], // dedupe
    summary: {
      traceCount: recentTraces.length,
      loopingQueries: loopingCount,
    },
  };

  // Save session log
  if (!existsSync(SESSION_LOG_DIR)) {
    mkdirSync(SESSION_LOG_DIR, { recursive: true });
  }
  
  const logPath = join(SESSION_LOG_DIR, `${sessionId}.json`);
  writeFileSync(logPath, JSON.stringify(sessionLog, null, 2));

  // Build status message
  let statusMessage = `📊 Session ${sessionId}\n`;
  statusMessage += `━━━━━━━━━━━━━━━━━━━━\n`;
  statusMessage += `Files edited: ${editedFiles.length}\n`;
  statusMessage += `Query traces: ${recentTraces.length}\n`;
  
  if (loopingCount > 0) {
    statusMessage += `⚠️ Looping queries: ${loopingCount}\n`;
    statusMessage += `Run: python scripts/analyze_traces.py --loops\n`;
  }

  // macOS notification
  if (process.platform === "darwin") {
    const notification = loopingCount > 0 
      ? `⚠️ ${loopingCount} looping queries detected`
      : `✅ ${editedFiles.length} files, ${recentTraces.length} traces`;
    
    Bun.spawn(["osascript", "-e", 
      `display notification "${notification}" with title "VSM Session Complete" sound name "Glass"`
    ]);
  }

  const response: StopResponse = {
    message: statusMessage,
  };

  console.log(JSON.stringify(response));
}

main().catch(console.error);

