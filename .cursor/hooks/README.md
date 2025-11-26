# VSM Cursor Hooks

Type-safe TypeScript hooks for debugging complex agentic RAG workflows.

## What These Hooks Do

| Hook | Trigger | Purpose |
|------|---------|---------|
| `before-prompt.ts` | Before prompt sent | Injects debugging context if recent queries looped |
| `after-edit.ts` | After file edit | Logs edits, warns on agent-critical file changes |
| `stop.ts` | Task complete | Saves session metadata, links with query traces, shows notification |

## What They DON'T Do (Cursor handles these)

- ❌ Linting/syntax checking → Cursor shows lint errors natively
- ❌ Diff display → Cursor shows diffs natively  
- ❌ Code reviews → Cursor 2.1 has AI code reviews
- ❌ Formatting → Use prettier/eslint config instead

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Cursor Agent Session                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │ beforeSubmit    │───▶│ afterFileEdit   │───▶│    stop     │ │
│  │ Prompt          │    │                 │    │             │ │
│  └────────┬────────┘    └────────┬────────┘    └──────┬──────┘ │
│           │                      │                     │        │
└───────────┼──────────────────────┼─────────────────────┼────────┘
            │                      │                     │
            ▼                      ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │ Inject context  │   │ /tmp/vsm-edits  │   │ cursor_sessions │
   │ if debugging    │   │ .log            │   │ /{id}.json      │
   └─────────────────┘   └─────────────────┘   └────────┬────────┘
                                                        │
                              Links to ─────────────────┤
                                                        ▼
                                               ┌─────────────────┐
                                               │ query_traces/   │
                                               │ {id}.json       │
                                               └─────────────────┘
```

## Log Locations

| Log | Location | Contents |
|-----|----------|----------|
| File edits | `/tmp/vsm-edits.log` | Timestamped list of edited files |
| Cursor sessions | `logs/cursor_sessions/` | Session metadata, linked traces |
| Query traces | `logs/query_traces/` | Full agent decision traces |

## Quick Commands

```bash
# Check service status
curl -s localhost:8001/docs > /dev/null && echo "API ✅" || echo "API ❌"
curl -s localhost:8080/v1/.well-known/ready > /dev/null && echo "Weaviate ✅"

# View recent query traces
python scripts/analyze_traces.py

# Find looping queries
python scripts/analyze_traces.py --loops

# Run benchmark after changes
python scripts/run_benchmark.py --output results.json

# View recent edits
tail -20 /tmp/vsm-edits.log

# View session logs
ls -la logs/cursor_sessions/
```

## Development

Hooks use [Bun](https://bun.sh) + TypeScript with the `cursor-hooks` package for type safety.

```bash
# Install dependencies (already done)
cd .cursor/hooks && bun install

# Test a hook manually
echo '{"prompt": "test", "conversationId": "abc"}' | bun run stop.ts
```

## Troubleshooting

**Hooks not running?**
1. Check Cursor version ≥1.7 (hooks are beta)
2. Verify `bun` is in PATH
3. Check Cursor console for errors

**No notifications on macOS?**
- Allow Cursor in System Preferences → Notifications

**Missing query traces?**
- Ensure API is running with tracer enabled
- Check `logs/query_traces/` directory exists
