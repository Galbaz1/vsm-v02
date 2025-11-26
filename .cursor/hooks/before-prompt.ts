/**
 * BeforeSubmitPrompt Hook - Injects RAG debugging context
 * 
 * PURPOSE: Help Context Agent by auto-injecting relevant context
 * - Warns if recent queries had looping issues
 * - Suggests checking query traces when debugging agent
 * - Reminds about benchmark commands
 * 
 * DOES NOT: Block or modify prompts (just adds context)
 */

import type { BeforeSubmitPromptPayload, BeforeSubmitPromptResponse } from "cursor-hooks";
import { existsSync, readdirSync, statSync } from "fs";
import { join } from "path";

const QUERY_TRACE_DIR = "logs/query_traces";

// Keywords that suggest user is debugging agent issues
const DEBUG_KEYWORDS = [
  "loop", "looping", "not working", "broken", "bug", "issue",
  "agent", "decision", "trace", "benchmark", "query", "search",
  "max iteration", "doesn't find", "wrong result"
];

async function main() {
  const payload: BeforeSubmitPromptPayload = await Bun.stdin.json();
  const { prompt } = payload;
  
  const promptLower = prompt?.toLowerCase() || "";
  const isDebuggingAgent = DEBUG_KEYWORDS.some(kw => promptLower.includes(kw));

  // Check for recent looping queries
  let loopingTraces: string[] = [];
  let recentTraceCount = 0;

  if (existsSync(QUERY_TRACE_DIR)) {
    const tenMinAgo = Date.now() - 10 * 60 * 1000;
    const files = readdirSync(QUERY_TRACE_DIR).filter(f => f.endsWith(".json"));
    
    for (const file of files) {
      const filePath = join(QUERY_TRACE_DIR, file);
      const stat = statSync(filePath);
      
      if (stat.mtimeMs > tenMinAgo) {
        recentTraceCount++;
        
        try {
          const trace = JSON.parse(await Bun.file(filePath).text());
          if (trace.final_outcome === "max_iterations") {
            loopingTraces.push(trace.user_query?.slice(0, 50) || file);
          }
        } catch {}
      }
    }
  }

  // Build context injection
  const contextParts: string[] = [];

  // Always show if there are looping queries
  if (loopingTraces.length > 0) {
    contextParts.push(
      `⚠️ ${loopingTraces.length} recent queries hit max iterations:`,
      ...loopingTraces.map(q => `  - "${q}..."`),
      `Check: python scripts/analyze_traces.py --loops`
    );
  }

  // If debugging, add helpful context
  if (isDebuggingAgent && recentTraceCount > 0) {
    contextParts.push(
      `\n📊 Debugging context available:`,
      `  - ${recentTraceCount} query traces in logs/query_traces/`,
      `  - Analyze: python scripts/analyze_traces.py`,
      `  - Context Agent rule: .cursor/rules/context_agent.mdc`
    );
  }

  // Build response
  const response: BeforeSubmitPromptResponse = {
    // Don't block - continue is default
    continue: true,
  };

  // Only add system message if we have context to inject
  if (contextParts.length > 0) {
    response.systemMessage = contextParts.join("\n");
  }

  console.log(JSON.stringify(response));
}

main().catch(console.error);

