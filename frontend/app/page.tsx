'use client';

import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { SearchBar } from '@/components/SearchBar';
import { ResultCard } from '@/components/ResultCard';
import { PreviewPanel } from '@/components/PreviewPanel';
import { useManualSearch } from '@/lib/hooks/useManualSearch';
import { useAgenticSearch } from '@/lib/hooks/useAgenticSearch';
import { useDebounce } from '@/lib/hooks/useDebounce';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { fetchBenchmarkQuestions, runSingleBenchmark } from '@/lib/api';
import type { SearchHit, SourceRef, BenchmarkQuestion, SingleBenchmarkResult } from '@/lib/types';

export default function Home() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [chunkType, setChunkType] = useState<string | undefined>(undefined);
  const [groupByPage, setGroupByPage] = useState(false);
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [isAgenticMode, setIsAgenticMode] = useState(false);
  
  // Benchmark suggested queries
  const [benchmarkQuestions, setBenchmarkQuestions] = useState<BenchmarkQuestion[]>([]);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<SingleBenchmarkResult | null>(null);
  const [activeBenchmarkQuery, setActiveBenchmarkQuery] = useState<BenchmarkQuestion | null>(null);

  const debouncedQuery = useDebounce(searchQuery, 400);
  
  // Hooks must be declared before they're used in effects
  const { data, isLoading, error } = useManualSearch(
    isAgenticMode ? '' : debouncedQuery, 
    10, 
    chunkType, 
    groupByPage
  );
  
  // Agentic search state
  const agentic = useAgenticSearch();
  
  // Load benchmark questions on mount
  useEffect(() => {
    fetchBenchmarkQuestions(5)
      .then(setBenchmarkQuestions)
      .catch(console.error);
  }, []);
  
  // Restore state from sessionStorage on mount
  useEffect(() => {
    const savedState = sessionStorage.getItem('searchPageState');
    if (savedState) {
      try {
        const state = JSON.parse(savedState);
        if (state.searchQuery) setSearchQuery(state.searchQuery);
        if (state.isAgenticMode) setIsAgenticMode(state.isAgenticMode);
        if (state.benchmarkResult) setBenchmarkResult(state.benchmarkResult);
        if (state.activeBenchmarkQuery) setActiveBenchmarkQuery(state.activeBenchmarkQuery);
        // Restore agentic results if available
        if (state.agenticState && state.agenticState.response) {
          agentic.restore(state.agenticState);
        }
      } catch (e) {
        console.error('Failed to restore state:', e);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  
  // Save state to sessionStorage when key values change
  useEffect(() => {
    if (searchQuery || isAgenticMode || benchmarkResult || agentic.response) {
      const state = {
        searchQuery,
        isAgenticMode,
        benchmarkResult,
        activeBenchmarkQuery,
        agenticState: agentic.isComplete ? {
          response: agentic.response,
          sources: agentic.sources,
          textResults: agentic.textResults,
          visualResults: agentic.visualResults,
          visualInterpretations: agentic.visualInterpretations,
          decisions: agentic.decisions,
        } : null,
      };
      sessionStorage.setItem('searchPageState', JSON.stringify(state));
    }
  }, [searchQuery, isAgenticMode, benchmarkResult, activeBenchmarkQuery, agentic.isComplete, agentic.response, agentic.sources, agentic.textResults, agentic.visualResults, agentic.visualInterpretations, agentic.decisions]);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (isAgenticMode && query.length >= 3) {
      agentic.search(query);
    }
  };

  const handleChunkTypeChange = (type: string | undefined) => {
    setChunkType(type);
  };

  const handlePreview = (hit: SearchHit) => {
    setSelectedHit(hit);
    setPreviewOpen(true);
  };

  const handleAgenticModeToggle = (enabled: boolean) => {
    setIsAgenticMode(enabled);
    if (!enabled) {
      agentic.reset();
    }
  };
  
  // Handle clicking a benchmark suggested query
  const handleBenchmarkQueryClick = async (question: BenchmarkQuestion) => {
    // Enable agentic mode and set query
    setIsAgenticMode(true);
    setSearchQuery(question.query);
    setActiveBenchmarkQuery(question);
    setIsBenchmarking(true);
    setBenchmarkResult(null);
    
    // Start agentic search
    agentic.search(question.query);
    
    // Run benchmark in parallel
    try {
      const result = await runSingleBenchmark(
        question.query,
        question.expected_answer,
        question.id,
      );
      setBenchmarkResult(result);
      
      // Store result in sessionStorage for benchmark dashboard
      sessionStorage.setItem('lastBenchmarkResult', JSON.stringify(result));
    } catch (err) {
      console.error('Benchmark failed:', err);
    } finally {
      setIsBenchmarking(false);
    }
  };
  
  // Navigate to benchmark dashboard with result
  const viewBenchmarkResult = () => {
    if (benchmarkResult) {
      router.push(`/benchmark?highlight=${benchmarkResult.id}`);
    }
  };

  // Group hits by (manual, page) or content_hash
  const groupedHits = useMemo(() => {
    if (!data?.hits) return [];
    
    if (!groupByPage) {
      // Simple grouping by content_hash for deduplication
      const seen = new Set<string>();
      const unique: SearchHit[] = [];
      const duplicates: Record<string, SearchHit[]> = {};
      
      for (const hit of data.hits) {
        const key = hit.content_hash || hit.anchor_id;
        if (!seen.has(key)) {
          seen.add(key);
          unique.push(hit);
          duplicates[key] = [hit];
        } else {
          duplicates[key].push(hit);
        }
      }
      
      return unique.map((hit) => {
        const key = hit.content_hash || hit.anchor_id;
        return {
          hit,
          duplicates: duplicates[key].slice(1),
          groupKey: key,
        };
      });
    } else {
      // Group by (manual, page)
      const groups: Record<string, SearchHit[]> = {};
      for (const hit of data.hits) {
        if (hit.page_number !== null && hit.page_number !== undefined) {
          const key = `${hit.manual_name}:${hit.page_number}`;
          if (!groups[key]) {
            groups[key] = [];
          }
          groups[key].push(hit);
        } else {
          // Handle hits without page numbers
          const key = `${hit.manual_name}:no-page`;
          if (!groups[key]) {
            groups[key] = [];
          }
          groups[key].push(hit);
        }
      }
      
      return Object.entries(groups).map(([groupKey, hits]) => ({
        hit: hits[0],
        duplicates: hits.slice(1),
        groupKey,
      }));
    }
  }, [data?.hits, groupByPage]);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Manual Search</h1>
              <p className="text-muted-foreground mt-2">
                Semantic search over technical asset manuals with visual grounding
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Link
                href="/chat"
                className="text-sm font-medium text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                💬 Chat
              </Link>
              <Link
                href="/benchmark"
                className="text-sm font-medium underline underline-offset-4 hover:text-foreground"
              >
                Benchmark
              </Link>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={isAgenticMode}
                  onChange={(e) => handleAgenticModeToggle(e.target.checked)}
                  className="rounded w-4 h-4"
                />
                <span className="text-sm font-medium">Agentic Mode</span>
              </label>
              {isAgenticMode && (
                <Badge variant="secondary" className="bg-emerald-100 text-emerald-700">
                  AI-Powered
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <SearchBar
            onSearch={handleSearch}
            onChunkTypeChange={handleChunkTypeChange}
            isLoading={isAgenticMode ? agentic.isLoading : isLoading}
            isAutoRefresh={!isAgenticMode && debouncedQuery.length >= 3}
          />
        </div>

        {/* Agentic Mode UI */}
        {isAgenticMode && (
          <>
            {/* Benchmark Progress/Result Panel */}
            {(isBenchmarking || benchmarkResult) && activeBenchmarkQuery && (
              <BenchmarkResultPanel
                isLoading={isBenchmarking}
                result={benchmarkResult}
                expectedAnswer={activeBenchmarkQuery.expected_answer}
                onViewDetails={viewBenchmarkResult}
              />
            )}
            
            <AgenticResults 
              agentic={agentic} 
              onPreview={handlePreview}
              searchQuery={searchQuery}
            />
          </>
        )}

        {/* Regular Search Mode UI */}
        {!isAgenticMode && (
          <>
            {error && (
              <div className="bg-destructive/10 text-destructive p-4 rounded-lg mb-6">
                <p className="font-semibold">Error</p>
                <p className="text-sm">{error instanceof Error ? error.message : 'Failed to search manuals'}</p>
              </div>
            )}

            {isLoading && debouncedQuery.length >= 3 && (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="border rounded-lg p-6">
                    <Skeleton className="h-6 w-1/3 mb-2" />
                    <Skeleton className="h-4 w-1/4 mb-4" />
                    <Skeleton className="h-20 w-full" />
                  </div>
                ))}
              </div>
            )}

            {!isLoading && !error && groupedHits.length > 0 && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">
                    Found {groupedHits.length} unique result{groupedHits.length !== 1 ? 's' : ''} for &quot;{data?.query}&quot;
                    {data && data.hits.length > groupedHits.length && (
                      <span className="ml-2">
                        ({data.hits.length} total, {data.hits.length - groupedHits.length} duplicates hidden)
                      </span>
                    )}
                  </p>
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={groupByPage}
                      onChange={(e) => setGroupByPage(e.target.checked)}
                      className="rounded"
                    />
                    Group by page
                  </label>
                </div>
                {groupedHits.map(({ hit, duplicates, groupKey }) => (
                  <div key={hit.anchor_id}>
                    <ResultCard
                      hit={hit}
                      query={data?.query || ''}
                      onPreview={handlePreview}
                    />
                    {duplicates.length > 0 && (
                      <div className="mt-2 ml-4">
                        {expandedGroups.has(groupKey) ? (
                          <>
                            <button
                              onClick={() => {
                                const newExpanded = new Set(expandedGroups);
                                newExpanded.delete(groupKey);
                                setExpandedGroups(newExpanded);
                              }}
                              className="text-xs text-muted-foreground hover:text-foreground"
                            >
                              Hide {duplicates.length} duplicate{duplicates.length !== 1 ? 's' : ''}
                            </button>
                            {duplicates.map((dup) => (
                              <div key={dup.anchor_id} className="mt-2">
                                <ResultCard
                                  hit={dup}
                                  query={data?.query || ''}
                                  onPreview={handlePreview}
                                />
                              </div>
                            ))}
                          </>
                        ) : (
                          <button
                            onClick={() => {
                              const newExpanded = new Set(expandedGroups);
                              newExpanded.add(groupKey);
                              setExpandedGroups(newExpanded);
                            }}
                            className="text-xs text-muted-foreground hover:text-foreground"
                          >
                            Show {duplicates.length} duplicate{duplicates.length !== 1 ? 's' : ''} from same {groupByPage ? 'page' : 'content'}
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {!isLoading && !error && debouncedQuery.length >= 3 && data && groupedHits.length === 0 && (
              <div className="text-center py-12">
                <p className="text-lg text-muted-foreground">No results found</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Try a different search query or check your spelling
                </p>
              </div>
            )}

            {searchQuery.length === 0 && (
              <div className="text-center py-12">
                <p className="text-lg text-muted-foreground">Enter a search query to get started</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Search for procedures, specifications, or any content from the manuals
                </p>
                
                {/* Suggested Benchmark Queries */}
                {benchmarkQuestions.length > 0 && (
                  <div className="mt-8 max-w-3xl mx-auto text-left">
                    <p className="text-sm font-medium text-muted-foreground mb-3">
                      Try a benchmark query:
                    </p>
                    <div className="space-y-2">
                      {benchmarkQuestions.map((q) => (
                        <button
                          key={q.id}
                          onClick={() => handleBenchmarkQueryClick(q)}
                          className="w-full text-left p-3 rounded-lg border hover:border-primary hover:bg-muted/50 transition-colors group"
                        >
                          <div className="flex items-start gap-2">
                            <Badge variant="outline" className="shrink-0 text-xs">
                              {q.category}
                            </Badge>
                            <span className="text-sm text-foreground group-hover:text-primary line-clamp-2">
                              {q.query}
                            </span>
                          </div>
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-3">
                      Clicking runs the query and evaluates against expected answer
                    </p>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </main>

      <PreviewPanel
        hit={selectedHit}
        pageHits={data?.page_hits || null}
        open={previewOpen}
        onClose={() => setPreviewOpen(false)}
      />
    </div>
  );
}

// Agentic Results Component
interface AgenticResultsProps {
  agentic: ReturnType<typeof useAgenticSearch>;
  onPreview: (hit: SearchHit) => void;
  searchQuery: string;
}

function AgenticResults({ agentic, onPreview, searchQuery }: AgenticResultsProps) {
  const handleSourceClick = (source: SourceRef, index: number) => {
    const pageNumber = typeof source.page === 'string' ? parseInt(source.page, 10) : source.page;
    if (Number.isNaN(pageNumber)) return;
    
    const textMatch = agentic.textResults.find(
      (t) => t.page_number === pageNumber && (!source.manual || t.manual_name === source.manual)
    );
    const visualMatch = agentic.visualResults.find(
      (v) => v.page_number === pageNumber && (!source.manual || v.asset_manual === source.manual)
    );
    
    const manualName = source.manual || textMatch?.manual_name || visualMatch?.asset_manual || 'Manual';
    const pageImageUrl = source.preview_url || visualMatch?.preview_url || textMatch?.page_image_url || null;
    const pdfPageUrl = source.pdf_page_url || textMatch?.pdf_page_url || '';
    const content = textMatch?.content || agentic.response || 'Source from agentic search.';
    const score = source.score ?? textMatch?.score ?? visualMatch?.maxsim_score ?? visualMatch?.score ?? null;
    
    const hit: SearchHit = {
      anchor_id: `source-${manualName}-${pageNumber}-${index}`,
      manual_name: manualName,
      content,
      page_number: pageNumber ?? null,
      bbox: textMatch?.bbox || null,
      pdf_page_url: pdfPageUrl || '',
      page_image_url: pageImageUrl || null,
      chunk_type: textMatch?.chunk_type,
      section_title: textMatch?.section_title,
      content_hash: null,
      score,
    };
    
    onPreview(hit);
  };
  
  return (
    <div className="space-y-6">
      {/* Status Bar */}
      {agentic.status && (
        <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
          {agentic.isLoading && (
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          )}
          <span className="text-sm">{agentic.status}</span>
          {agentic.isLoading && (
            <button
              onClick={agentic.cancel}
              className="ml-auto text-xs text-muted-foreground hover:text-foreground"
            >
              Cancel
            </button>
          )}
        </div>
      )}

      {/* Error Display */}
      {agentic.error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg">
          <p className="font-semibold">Error</p>
          <p className="text-sm">{agentic.error}</p>
        </div>
      )}

      {/* Decision Transparency */}
      {agentic.decisions.length > 0 && (
        <div className="p-4 border rounded-lg bg-muted/30">
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <span className="w-2 h-2 bg-emerald-500 rounded-full" />
            Agent Decisions
          </h3>
          <div className="space-y-2">
            {agentic.decisions.map((decision, i) => (
              <div key={i} className="text-sm">
                <span className="font-mono text-xs bg-muted px-1.5 py-0.5 rounded">
                  {decision.tool}
                </span>
                <span className="text-muted-foreground ml-2">{decision.reasoning}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Final Response */}
      {agentic.response && (
        <div className="p-6 border-2 border-emerald-200 rounded-lg bg-emerald-50/50">
          <h3 className="text-sm font-medium text-emerald-700 mb-3">AI Response</h3>
          <p className="text-foreground whitespace-pre-wrap">{agentic.response}</p>
          {agentic.sources.length > 0 && (
            <div className="mt-4 pt-4 border-t border-emerald-200">
              <p className="text-xs text-muted-foreground mb-2">Sources:</p>
              <div className="flex flex-wrap gap-2">
                {agentic.sources.map((source, i) => (
                  <Badge 
                    key={`${source.manual}-${source.page}-${i}`} 
                    variant="outline" 
                    className="text-xs cursor-pointer hover:bg-emerald-50"
                    onClick={() => handleSourceClick(source, i)}
                  >
                    {source.manual || 'Manual'} - Page {source.page}
                    {source.type && (
                      <span className="ml-1 text-[10px] uppercase text-muted-foreground">
                        • {source.type}
                      </span>
                    )}
                    {typeof source.score === 'number' && (
                      <span className="ml-1 text-[10px] text-muted-foreground">
                        • {source.score.toFixed(2)}
                      </span>
                    )}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Visual Results */}
      {agentic.visualResults.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <span className="w-2 h-2 bg-blue-500 rounded-full" />
            Visual Results ({agentic.visualResults.length})
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {agentic.visualResults.map((result, i) => (
              <div 
                key={i} 
                className="border rounded-lg overflow-hidden hover:shadow-md transition-shadow"
              >
                {result.preview_url && (
                  <img 
                    src={result.preview_url} 
                    alt={`Page ${result.page_number}`}
                    className="w-full h-40 object-cover"
                  />
                )}
                <div className="p-3">
                  <p className="text-sm font-medium">Page {result.page_number}</p>
                  <p className="text-xs text-muted-foreground">{result.asset_manual}</p>
                  <p className="text-xs text-muted-foreground">
                    {(() => {
                      const score = typeof result.maxsim_score === 'number' ? result.maxsim_score : result.score;
                      return `Score: ${typeof score === 'number' ? score.toFixed(3) : '—'}`;
                    })()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Visual Interpretations */}
      {agentic.visualInterpretations.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <span className="w-2 h-2 bg-purple-500 rounded-full" />
            Visual Interpretations ({agentic.visualInterpretations.length})
          </h3>
          <div className="space-y-4">
            {agentic.visualInterpretations.map((interp, i) => (
              <div key={i} className="p-4 border rounded-lg bg-purple-50/50">
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="outline">Page {interp.page_number}</Badge>
                  <span className="text-xs text-muted-foreground">{interp.asset_manual}</span>
                </div>
                {interp.error ? (
                  <p className="text-sm text-destructive">{interp.error}</p>
                ) : (
                  <p className="text-sm whitespace-pre-wrap">{interp.interpretation}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Text Results */}
      {agentic.textResults.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <span className="w-2 h-2 bg-amber-500 rounded-full" />
            Text Results ({agentic.textResults.length})
          </h3>
          <div className="space-y-3">
            {agentic.textResults.map((result, i) => {
              // Convert to SearchHit format for ResultCard
              const hit: SearchHit = {
                anchor_id: `agentic-${i}`,
                manual_name: result.manual_name,
                content: result.content,
                page_number: result.page_number,
                bbox: result.bbox || null,
                pdf_page_url: result.pdf_page_url || '',
                page_image_url: result.page_image_url || null,
                chunk_type: result.chunk_type,
                section_title: result.section_title,
                score: result.score ?? null,
              };
              return (
                <ResultCard
                  key={i}
                  hit={hit}
                  query={searchQuery}
                  onPreview={onPreview}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!agentic.isLoading && 
       !agentic.error && 
       agentic.textResults.length === 0 && 
       agentic.visualResults.length === 0 && 
       !agentic.response && (
        <div className="text-center py-12">
          <p className="text-lg text-muted-foreground">
            Enter a query and press Enter to start an agentic search
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            The AI agent will decide the best tools to answer your question
          </p>
        </div>
      )}
    </div>
  );
}


// Benchmark Result Panel Component
interface BenchmarkResultPanelProps {
  isLoading: boolean;
  result: SingleBenchmarkResult | null;
  expectedAnswer: string;
  onViewDetails: () => void;
}

// Static class mappings for Tailwind JIT (dynamic interpolation doesn't work)
const scoreStyles = {
  emerald: {
    container: 'bg-emerald-50/50 border-emerald-200',
    dot: 'bg-emerald-500',
    text: 'text-emerald-600',
  },
  amber: {
    container: 'bg-amber-50/50 border-amber-200',
    dot: 'bg-amber-500',
    text: 'text-amber-600',
  },
  red: {
    container: 'bg-red-50/50 border-red-200',
    dot: 'bg-red-500',
    text: 'text-red-600',
  },
} as const;

function BenchmarkResultPanel({ isLoading, result, expectedAnswer, onViewDetails }: BenchmarkResultPanelProps) {
  if (isLoading) {
    return (
      <div className="mb-6 p-4 border rounded-lg bg-blue-50/50 border-blue-200">
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-medium text-blue-700">Running benchmark evaluation...</span>
        </div>
      </div>
    );
  }
  
  if (!result) return null;
  
  const scoreKey = result.judge_score >= 0.7 ? 'emerald' : result.judge_score >= 0.4 ? 'amber' : 'red';
  const styles = scoreStyles[scoreKey];
  const scorePercent = Math.round(result.judge_score * 100);
  
  return (
    <div className={`mb-6 p-4 border rounded-lg ${styles.container}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${styles.dot}`} />
          Benchmark Result
        </h3>
        <div className="flex items-center gap-3">
          <span className={`text-lg font-bold ${styles.text}`}>
            {scorePercent}%
          </span>
          <button
            onClick={onViewDetails}
            className="text-xs text-muted-foreground hover:text-foreground underline"
          >
            View in Dashboard
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-xs text-muted-foreground mb-1">Expected Answer</p>
          <p className="text-foreground line-clamp-3">{expectedAnswer}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-1">Judge Rationale</p>
          <p className="text-foreground line-clamp-3">{result.judge_rationale}</p>
        </div>
      </div>
      
      <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
        <span>Latency: {Math.round(result.latency_ms)}ms</span>
        <span>Iterations: {result.iterations}</span>
        <span>Tools: {result.tools_used.join(' → ')}</span>
      </div>
    </div>
  );
}
