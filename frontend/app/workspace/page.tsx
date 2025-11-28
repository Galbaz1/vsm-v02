'use client';

/* eslint-disable @next/next/no-img-element */
import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { PreviewPanel } from '@/components/PreviewPanel';
import { ResultCard, cleanContent } from '@/components/ResultCard';
import { Skeleton } from '@/components/ui/skeleton';
import { useAgenticSearch } from '@/lib/hooks/useAgenticSearch';
import { useChat } from '@/lib/hooks/useChat';
import { useDebounce } from '@/lib/hooks/useDebounce';
import { useManualSearch } from '@/lib/hooks/useManualSearch';
import type { SearchHit } from '@/lib/types';

type WorkspaceMode = 'search' | 'agent' | 'chat';
type EvidenceTab = 'text' | 'visual' | 'trace';

function recordTelemetry(event: string, data?: Record<string, unknown>) {
  console.info('[telemetry]', event, data || {});
}

interface EvidenceSource {
  id: string;
  manual: string;
  page: number | string | null;
  type: 'text' | 'visual';
  score?: number | null;
  preview_url?: string | null;
  pdf_page_url?: string;
  snippet?: string;
}

export default function WorkspacePage() {
  const [mode, setMode] = useState<WorkspaceMode>('search');
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 350);
  const [showInlineCitations, setShowInlineCitations] = useState(false);

  const manualQuery = mode === 'search' ? debouncedQuery : '';
  const { data: manualData, isLoading: manualLoading, error: manualError } = useManualSearch(
    manualQuery,
    10,
    undefined,
    false
  );

  const agentic = useAgenticSearch();

  const {
    messages,
    decisions: chatDecisions,
    isLoading: chatLoading,
    error: chatError,
    sendMessage,
    startNewChat,
    currentChatId,
  } = useChat();

  useEffect(() => {
    if (mode === 'chat' && !currentChatId && messages.length === 0) {
      startNewChat();
    }
  }, [mode, currentChatId, messages.length, startNewChat]);

  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);

  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [evidencePinned, setEvidencePinned] = useState(false);
  const [activeEvidenceTab, setActiveEvidenceTab] = useState<EvidenceTab>('text');

  const handlePreview = (hit: SearchHit) => {
    setSelectedHit(hit);
    setPreviewOpen(true);
  };

  const handleRun = () => {
    if (!query.trim()) return;
    if (mode === 'agent') {
      agentic.search(query.trim());
    } else if (mode === 'chat') {
      sendMessage(query.trim());
      setQuery('');
    } else {
      // Manual search auto-runs via debounced query
    }
  };

  const manualHits = manualData?.hits;

  const manualSources = useMemo(() => {
    if (!manualHits) return [];
    const seen = new Set<string>();
    return manualHits
      .map((hit) => {
        const key = `${hit.manual_name}:${hit.page_number}:text`;
        if (seen.has(key)) return null;
        seen.add(key);
        return {
          id: hit.anchor_id,
          manual: hit.manual_name,
          page: hit.page_number !== null ? hit.page_number + 1 : null,
          type: 'text' as const,
          score: hit.score ?? null,
          preview_url: hit.page_image_url,
          pdf_page_url: hit.pdf_page_url,
          snippet: cleanContent(hit.content).slice(0, 320),
        };
      })
      .filter(Boolean) as EvidenceSource[];
  }, [manualHits]);

  const agentSources = useMemo(() => {
    const sources: Record<string, EvidenceSource> = {};

    const addOrMerge = (key: string, source: EvidenceSource) => {
      if (!sources[key]) {
        sources[key] = source;
      } else {
        sources[key] = {
          ...sources[key],
          ...source,
        };
      }
    };

    agentic.sources.forEach((s, i) => {
      const pageNumber = typeof s.page === 'string' ? Number(s.page) : s.page;
      const key = `${s.manual || 'Manual'}:${pageNumber}:${s.type || 'text'}`;
      addOrMerge(key, {
        id: `source-${i}`,
        manual: s.manual || 'Manual',
        page: pageNumber ?? null,
        type: (s.type || 'text') as 'text' | 'visual',
        score: s.score ?? null,
        preview_url: s.preview_url,
        pdf_page_url: s.pdf_page_url,
      });
    });

    agentic.textResults.forEach((t, i) => {
      const key = `${t.manual_name}:${t.page_number}:text`;
      addOrMerge(key, {
        id: `text-${i}`,
        manual: t.manual_name,
        page: t.page_number + 1,
        type: 'text',
        score: t.score ?? null,
        preview_url: t.page_image_url,
        pdf_page_url: t.pdf_page_url,
        snippet: t.content.slice(0, 320),
      });
    });

    agentic.visualResults.forEach((v, i) => {
      const key = `${v.asset_manual}:${v.page_number}:visual`;
      addOrMerge(key, {
        id: `visual-${i}`,
        manual: v.asset_manual,
        page: v.page_number,
        type: 'visual',
        score: v.maxsim_score ?? v.score ?? null,
        preview_url: v.preview_url,
      });
    });

    return Object.values(sources);
  }, [agentic.sources, agentic.textResults, agentic.visualResults]);

  const lastAssistantMessage = useMemo(
    () => [...messages].reverse().find((m) => m.role === 'assistant'),
    [messages]
  );

  const chatSources = useMemo(() => {
    if (!lastAssistantMessage?.sources) return [];
    return lastAssistantMessage.sources.map((s, i) => ({
      id: `chat-source-${i}`,
      manual: s.manual || 'Manual',
      page: s.page ?? null,
      type: (s.type || 'text') as 'text' | 'visual',
      score: s.score ?? null,
      preview_url: s.preview_url,
      pdf_page_url: s.pdf_page_url,
    }));
  }, [lastAssistantMessage]);

  const activeSources = useMemo(() => {
    if (mode === 'search') return manualSources;
    if (mode === 'agent') return agentSources;
    return chatSources;
  }, [mode, manualSources, agentSources, chatSources]);

  const textSources = activeSources.filter((s) => s.type === 'text');
  const visualSources = activeSources.filter((s) => s.type === 'visual');
  const sourceCount = textSources.length + visualSources.length;

  const traceSteps =
    mode === 'agent' ? agentic.decisions : mode === 'chat' ? chatDecisions : [];

  const manualPageHits = manualData?.page_hits || null;

  const handleSourceOpen = () => {
    setEvidenceOpen(true);
    recordTelemetry('evidence_open', { mode, sourceCount });
  };

  const handleSourceClose = () => {
    setEvidenceOpen(false);
    recordTelemetry('evidence_close', { mode });
  };

  const handleEvidencePin = () => {
    setEvidencePinned((v) => {
      const next = !v;
      recordTelemetry('evidence_pin_toggle', { pinned: next });
      return next;
    });
  };

  const handleEvidenceTabChange = (tab: EvidenceTab) => {
    setActiveEvidenceTab(tab);
    recordTelemetry('evidence_tab_change', { tab });
  };

  const handleInlineCitationsToggle = () => {
    setShowInlineCitations((v) => {
      const next = !v;
      recordTelemetry('inline_citations_toggle', { enabled: next });
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-foreground text-background flex items-center justify-center font-semibold">
              V
            </div>
            <div>
              <h1 className="text-xl font-semibold">VSM Workspace</h1>
              <p className="text-sm text-muted-foreground">
                Unified manual search, agent runs, and chat in one view
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/benchmark"
              className="text-sm text-muted-foreground hover:text-foreground underline underline-offset-4"
            >
              Benchmark
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        <section className="p-4 border rounded-lg bg-card">
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <ModeChip label="Quick Search" active={mode === 'search'} onClick={() => setMode('search')} />
            <ModeChip label="Agent" active={mode === 'agent'} onClick={() => setMode('agent')} />
            <ModeChip label="Thread" active={mode === 'chat'} onClick={() => setMode('chat')} />
            {sourceCount > 0 && (
              <Button variant="outline" size="sm" onClick={handleSourceOpen} className="ml-auto">
                Sources ({sourceCount})
              </Button>
            )}
          </div>
          <div className="flex flex-col gap-3 md:flex-row md:items-center">
            <div className="flex-1">
              <Input
                placeholder={
                  mode === 'search'
                    ? "Search manuals (e.g., 'battery test procedure')"
                    : mode === 'agent'
                      ? 'Ask the agent a question'
                      : 'Send a chat message'
                }
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleRun();
                  }
                }}
              />
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={handleRun} disabled={!query.trim()}>
                {mode === 'chat' ? 'Send' : 'Run'}
              </Button>
              {(mode === 'agent' && agentic.isLoading) || (mode === 'chat' && chatLoading) ? (
                <Badge variant="secondary">Running…</Badge>
              ) : null}
            </div>
          </div>
        </section>

        <div className="grid gap-6 lg:grid-cols-[1.7fr,1fr]">
          <section className="space-y-4">
            {mode === 'search' && (
              <SearchResultsSection
                query={manualQuery}
                loading={manualLoading}
                error={manualError instanceof Error ? manualError.message : null}
                data={manualData}
                onPreview={handlePreview}
                onOpenSources={handleSourceOpen}
                sourcesCount={sourceCount}
              />
            )}

            {mode === 'agent' && (
              <AgentSection
                agentic={agentic}
                query={query}
                onPreview={handlePreview}
                sourcesCount={sourceCount}
                onOpenSources={handleSourceOpen}
                showInlineCitations={showInlineCitations}
                onToggleInlineCitations={handleInlineCitationsToggle}
              />
            )}

            {mode === 'chat' && (
              <ChatSection
                messages={messages}
                decisions={chatDecisions}
                error={chatError}
                loading={chatLoading}
                onOpenSources={handleSourceOpen}
              />
            )}
          </section>

          {(evidenceOpen || evidencePinned) && (
            <EvidencePanel
              textSources={textSources}
              visualSources={visualSources}
              trace={traceSteps}
              onClose={handleSourceClose}
              onPinToggle={handleEvidencePin}
              pinned={evidencePinned}
              activeTab={activeEvidenceTab}
              onTabChange={handleEvidenceTabChange}
            />
          )}
        </div>
      </main>

      <PreviewPanel
        hit={selectedHit}
        pageHits={manualPageHits}
        open={previewOpen}
        onClose={() => setPreviewOpen(false)}
      />
    </div>
  );
}

function ModeChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`text-sm px-3 py-1 rounded-full border transition-colors ${
        active ? 'bg-foreground text-background border-foreground' : 'hover:border-foreground/60'
      }`}
    >
      {label}
    </button>
  );
}

function SearchResultsSection({
  query,
  loading,
  error,
  data,
  onPreview,
  onOpenSources,
  sourcesCount,
}: {
  query: string;
  loading: boolean;
  error: string | null;
  data: ReturnType<typeof useManualSearch>['data'];
  onPreview: (hit: SearchHit) => void;
  onOpenSources: () => void;
  sourcesCount: number;
}) {
  return (
    <div className="p-4 border rounded-lg bg-card space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-semibold">Manual Results</h2>
          <p className="text-sm text-muted-foreground">
            Direct semantic search over manuals with previews
          </p>
        </div>
        <div className="flex items-center gap-2">
          {sourcesCount > 0 && (
            <Button variant="outline" size="sm" onClick={onOpenSources}>
              Sources ({sourcesCount})
            </Button>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive p-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {loading && query.length >= 3 && (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="border rounded-lg p-4">
              <Skeleton className="h-5 w-1/3 mb-2" />
              <Skeleton className="h-4 w-1/4 mb-3" />
              <Skeleton className="h-16 w-full" />
            </div>
          ))}
        </div>
      )}

      {!loading && data?.hits && data.hits.length > 0 && (
        <div className="space-y-3">
          {data.hits.map((hit) => (
            <ResultCard key={hit.anchor_id} hit={hit} query={data.query} onPreview={onPreview} />
          ))}
        </div>
      )}

      {!loading && query.length >= 3 && data?.hits?.length === 0 && (
        <div className="text-sm text-muted-foreground">No results found for “{query}”.</div>
      )}
    </div>
  );
}

function AgentSection({
  agentic,
  query,
  onPreview,
  sourcesCount,
  onOpenSources,
  showInlineCitations,
  onToggleInlineCitations,
}: {
  agentic: ReturnType<typeof useAgenticSearch>;
  query: string;
  onPreview: (hit: SearchHit) => void;
  sourcesCount: number;
  onOpenSources: () => void;
  showInlineCitations: boolean;
  onToggleInlineCitations: () => void;
}) {
  const renderResponse = () => {
    if (!agentic.response) return null;
    if (!showInlineCitations || agentic.sources.length === 0) {
      return <p className="text-foreground whitespace-pre-wrap text-sm">{agentic.response}</p>;
    }

    return (
      <div className="space-y-3">
        <p className="text-foreground whitespace-pre-wrap text-sm">
          {agentic.response}{' '}
          <span className="align-super text-[10px] text-muted-foreground">
            {agentic.sources.map((_, idx) => (
              <sup key={idx} className="mr-1">
                {idx + 1}
              </sup>
            ))}
          </span>
        </p>
        <div className="flex flex-wrap gap-2">
          {agentic.sources.map((s, idx) => (
            <Badge key={idx} variant="outline" className="text-[11px]">
              {idx + 1}. {s.manual || 'Manual'} — Page {s.page ?? '?'}
              {typeof s.score === 'number' && (
                <span className="ml-1 text-muted-foreground">({s.score.toFixed(2)})</span>
              )}
            </Badge>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="p-4 border rounded-lg bg-card space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-semibold">Agent</h2>
          <p className="text-sm text-muted-foreground">Tool-using agent with text + visual context</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleInlineCitations}
            className={`text-xs px-3 py-1 rounded-full border ${
              showInlineCitations ? 'bg-foreground text-background border-foreground' : 'hover:border-foreground/60'
            }`}
          >
            Inline citations {showInlineCitations ? 'On' : 'Off'}
          </button>
          {agentic.status && (
            <Badge variant="secondary">
              {agentic.isLoading ? 'Running' : 'Ready'} • {agentic.status}
            </Badge>
          )}
          {sourcesCount > 0 && (
            <Button variant="outline" size="sm" onClick={onOpenSources}>
              Sources ({sourcesCount})
            </Button>
          )}
        </div>
      </div>

      {agentic.error && (
        <div className="bg-destructive/10 text-destructive p-3 rounded-lg text-sm">
          {agentic.error}
        </div>
      )}

      {agentic.decisions.length > 0 && (
        <div className="p-3 border rounded-lg bg-muted/40 space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Trace</p>
          {agentic.decisions.map((d, i) => (
            <div key={i} className="text-sm flex items-start gap-2">
              <Badge variant="outline" className="text-[11px]">
                {d.tool}
              </Badge>
              <span className="text-muted-foreground">{d.reasoning}</span>
            </div>
          ))}
        </div>
      )}

      {agentic.response && (
        <div className="p-4 border rounded-lg bg-emerald-50/60">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-semibold text-emerald-700">AI Response</p>
            {sourcesCount > 0 && (
              <button
                className="text-xs text-emerald-700 underline"
                onClick={onOpenSources}
              >
                View sources
              </button>
            )}
          </div>
          {renderResponse()}
        </div>
      )}

      {agentic.textResults.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Text Results</h3>
            <span className="text-xs text-muted-foreground">{agentic.textResults.length}</span>
          </div>
          <div className="space-y-2">
            {agentic.textResults.map((t, i) => {
              const hit: SearchHit = {
                anchor_id: `agent-hit-${i}`,
                manual_name: t.manual_name,
                content: t.content,
                page_number: t.page_number,
                score: t.score ?? null,
                bbox: t.bbox || null,
                pdf_page_url: t.pdf_page_url || '',
                page_image_url: t.page_image_url || null,
                chunk_type: t.chunk_type,
                section_title: t.section_title,
                content_hash: null,
              };
              return <ResultCard key={hit.anchor_id} hit={hit} query={query} onPreview={onPreview} />;
            })}
          </div>
        </div>
      )}

      {agentic.visualResults.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Visual Results</h3>
            <span className="text-xs text-muted-foreground">{agentic.visualResults.length}</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {agentic.visualResults.map((v, i) => (
              <div key={i} className="border rounded-lg overflow-hidden">
                {v.preview_url && (
                  <img
                    src={v.preview_url}
                    alt={`Page ${v.page_number}`}
                    className="w-full h-28 object-cover"
                  />
                )}
                <div className="p-2 space-y-1">
                  <p className="text-sm font-medium leading-tight">Page {v.page_number}</p>
                  <p className="text-xs text-muted-foreground line-clamp-1">{v.asset_manual}</p>
                  {typeof v.maxsim_score === 'number' && (
                    <p className="text-xs text-muted-foreground">Score: {v.maxsim_score.toFixed(3)}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!agentic.isLoading &&
        !agentic.error &&
        !agentic.response &&
        agentic.textResults.length === 0 &&
        agentic.visualResults.length === 0 && (
          <div className="text-sm text-muted-foreground">
            Run an agentic search to see results. Try a multi-step question for best coverage.
          </div>
        )}
    </div>
  );
}

function ChatSection({
  messages,
  decisions,
  error,
  loading,
  onOpenSources,
}: {
  messages: ReturnType<typeof useChat>['messages'];
  decisions: ReturnType<typeof useChat>['decisions'];
  error: ReturnType<typeof useChat>['error'];
  loading: boolean;
  onOpenSources: () => void;
}) {
  const latestAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
  const hasSources = (latestAssistant?.sources?.length || 0) > 0;

  return (
    <div className="p-4 border rounded-lg bg-card space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-semibold">Thread</h2>
          <p className="text-sm text-muted-foreground">Chat with inline sources on demand</p>
        </div>
        {hasSources && (
          <Button variant="outline" size="sm" onClick={onOpenSources}>
            Sources ({latestAssistant?.sources?.length})
          </Button>
        )}
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive p-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      <div className="space-y-3">
        {messages.length === 0 && (
          <div className="text-sm text-muted-foreground">
            Start chatting to see answers with citations.
          </div>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 border ${
                m.role === 'user'
                  ? 'bg-foreground text-background border-foreground'
                  : 'bg-muted text-foreground'
              }`}
            >
              <p className="text-sm whitespace-pre-wrap">{m.content}</p>
              {m.sources && m.sources.length > 0 && (
                <div className="mt-2 text-xs text-muted-foreground">
                  Sources: {m.sources.length}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="w-2 h-2 bg-foreground rounded-full animate-pulse" />
            Thinking…
          </div>
        )}
      </div>

      {decisions.length > 0 && (
        <div className="p-3 border rounded-lg bg-muted/50 space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Trace</p>
          {decisions.map((d, i) => (
            <div key={i} className="text-sm flex items-start gap-2">
              <Badge variant="outline" className="text-[11px]">
                {d.tool}
              </Badge>
              <span className="text-muted-foreground">{d.reasoning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EvidencePanel({
  textSources,
  visualSources,
  trace,
  pinned,
  onPinToggle,
  onClose,
  activeTab,
  onTabChange,
}: {
  textSources: EvidenceSource[];
  visualSources: EvidenceSource[];
  trace: { tool: string; reasoning: string }[];
  pinned: boolean;
  onPinToggle: () => void;
  onClose: () => void;
  activeTab: EvidenceTab;
  onTabChange: (tab: EvidenceTab) => void;
}) {
  return (
    <aside className="h-full border rounded-lg bg-card p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">Evidence</p>
          <p className="text-xs text-muted-foreground">Text, visual, and trace on demand</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            className={`text-xs px-2 py-1 rounded border ${
              pinned ? 'bg-foreground text-background' : ''
            }`}
            onClick={onPinToggle}
          >
            {pinned ? 'Pinned' : 'Pin'}
          </button>
          <button className="text-xs text-muted-foreground hover:text-foreground" onClick={onClose}>
            Close
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <TabButton active={activeTab === 'text'} onClick={() => onTabChange('text')}>
          Text ({textSources.length})
        </TabButton>
        <TabButton active={activeTab === 'visual'} onClick={() => onTabChange('visual')}>
          Visual ({visualSources.length})
        </TabButton>
        <TabButton active={activeTab === 'trace'} onClick={() => onTabChange('trace')}>
          Trace ({trace.length})
        </TabButton>
      </div>

      {activeTab === 'text' && (
        <div className="space-y-3 max-h-[65vh] overflow-y-auto pr-1">
          {textSources.length === 0 && (
            <p className="text-sm text-muted-foreground">No text sources yet.</p>
          )}
          {textSources.map((s) => (
            <div key={s.id} className="border rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-sm font-semibold">
                  {s.manual} — {s.page ?? 'Page'}
                </div>
                {typeof s.score === 'number' && (
                  <span className="text-xs text-muted-foreground">{s.score.toFixed(2)}</span>
                )}
              </div>
              {s.snippet && (
                <p className="text-sm text-muted-foreground whitespace-pre-wrap">{s.snippet}</p>
              )}
              <div className="flex items-center gap-2">
                {s.preview_url && (
                  <img
                    src={s.preview_url}
                    alt="preview"
                    className="h-16 w-12 object-cover rounded border"
                  />
                )}
                {s.pdf_page_url && (
                  <a
                    href={s.pdf_page_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs underline text-muted-foreground hover:text-foreground"
                  >
                    Open PDF
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'visual' && (
        <div className="space-y-3 max-h-[65vh] overflow-y-auto pr-1">
          {visualSources.length === 0 && (
            <p className="text-sm text-muted-foreground">No visual sources yet.</p>
          )}
          {visualSources.map((s) => (
            <div key={s.id} className="border rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-sm font-semibold">
                  {s.manual} — {s.page ?? 'Page'}
                </div>
                {typeof s.score === 'number' && (
                  <span className="text-xs text-muted-foreground">{s.score.toFixed(2)}</span>
                )}
              </div>
              {s.preview_url && (
                <img
                  src={s.preview_url}
                  alt="visual preview"
                  className="w-full h-40 object-cover rounded border"
                />
              )}
              {s.pdf_page_url && (
                <a
                  href={s.pdf_page_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs underline text-muted-foreground hover:text-foreground"
                >
                  Open PDF
                </a>
              )}
            </div>
          ))}
        </div>
      )}

      {activeTab === 'trace' && (
        <div className="space-y-2 max-h-[65vh] overflow-y-auto pr-1">
          {trace.length === 0 && <p className="text-sm text-muted-foreground">No trace yet.</p>}
          {trace.map((t, i) => (
            <div key={i} className="border rounded-lg p-3 space-y-1">
              <p className="text-sm font-semibold">{t.tool}</p>
              <p className="text-sm text-muted-foreground">{t.reasoning}</p>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}

function TabButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`text-xs px-3 py-1 rounded-full border transition-colors ${
        active ? 'bg-foreground text-background border-foreground' : 'hover:border-foreground/60'
      }`}
    >
      {children}
    </button>
  );
}
