'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { fetchBenchmarkReport } from '@/lib/api';
import type { BenchmarkRun, BenchmarkRecord } from '@/lib/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';

function formatMs(ms?: number) {
  if (ms === undefined || ms === null) return '—';
  if (ms > 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${ms.toFixed(0)}ms`;
}

function formatScore(score?: number) {
  if (score === undefined || score === null) return '—';
  return score.toFixed(2);
}

export default function BenchmarkReportPage({ params }: { params: { filename: string } }) {
  const [report, setReport] = useState<BenchmarkRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setLoading(true);
    fetchBenchmarkReport(params.filename)
      .then((data) => setReport(data))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load report'))
      .finally(() => setLoading(false));
  }, [params.filename]);

  const maxScore = useMemo(() => {
    if (!report || report.records.length === 0) return 1;
    return Math.max(...report.records.map((r) => r.judge_score));
  }, [report]);
  const maxLatency = useMemo(() => {
    if (!report || report.records.length === 0) return 1;
    return Math.max(...report.records.map((r) => r.latency_ms));
  }, [report]);
  const topSources = useMemo(() => {
    if (!report) return [];
    const counts: Record<string, { manual: string; page: string | number; type?: string; count: number }> = {};
    report.records.forEach((rec) => {
      rec.sources.forEach((src) => {
        const manual = src.manual || 'Manual';
        const page = src.page ?? '?';
        const key = `${manual}#${page}`;
        if (!counts[key]) {
          counts[key] = { manual, page, type: src.type, count: 0 };
        }
        counts[key].count += 1;
      });
    });
    return Object.values(counts).sort((a, b) => b.count - a.count).slice(0, 5);
  }, [report]);

  const toggleExpanded = (id: string) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-6 flex items-center justify-between">
          <div>
            <p className="text-sm uppercase tracking-wide text-muted-foreground">Run Report</p>
            <h1 className="text-3xl font-bold break-words">{params.filename}</h1>
            {report?.dataset && (
              <p className="text-muted-foreground text-sm mt-1">Dataset: {report.dataset}</p>
            )}
          </div>
          <Link href="/benchmark" className="text-sm font-medium underline underline-offset-4">
            Back to Dashboard
          </Link>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 space-y-8">
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          </div>
        ) : error ? (
          <div className="text-destructive text-sm">Failed to load report: {error}</div>
        ) : report ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-muted-foreground">Average Score</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-semibold">{formatScore(report.metrics.avg_score)}</p>
                  <p className="text-xs text-muted-foreground mt-1">TechnicalJudge quality</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-muted-foreground">Hit@3 / MRR</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-semibold">
                    {Math.round(report.metrics.hit_at_3 * 100)}% / {report.metrics.mrr.toFixed(3)}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">Retrieval coverage</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-muted-foreground">Latency</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-lg font-semibold">
                    p50 {formatMs(report.metrics.latency_p50_ms)}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    p95 {formatMs(report.metrics.latency_p95_ms)}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-muted-foreground">Completion</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-semibold">
                    {report.completed}/{report.total}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">Iterations avg {report.metrics.avg_iterations.toFixed(1)}</p>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Score & Latency Distribution</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Each bar is a query. Click a row below to drill into decisions, sources, and rationale.
                  </p>
                </CardHeader>
                <CardContent className="space-y-3">
                  {report.records.map((record) => {
                    const scoreWidth = maxScore > 0 ? Math.max((record.judge_score / maxScore) * 100, 4) : 0;
                    const latencyWidth = maxLatency > 0 ? Math.max((record.latency_ms / maxLatency) * 100, 4) : 0;
                    return (
                      <div key={`${record.id}-bar`} className="space-y-1">
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span className="truncate max-w-[60%]">{record.query}</span>
                          <span>{formatScore(record.judge_score)} • {formatMs(record.latency_ms)}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full bg-primary"
                              style={{ width: `${scoreWidth}%` }}
                              title={`Score ${record.judge_score.toFixed(2)}`}
                            />
                          </div>
                          <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full bg-amber-500/80"
                              style={{ width: `${latencyWidth}%` }}
                              title={`Latency ${formatMs(record.latency_ms)}`}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Tool Usage</CardTitle>
                  <p className="text-sm text-muted-foreground">Frequency across the run.</p>
                </CardHeader>
                <CardContent className="space-y-3">
                  {Object.keys(report.metrics.tool_distribution || {}).length === 0 ? (
                    <p className="text-sm text-muted-foreground">No tool data captured.</p>
                  ) : (
                    Object.entries(report.metrics.tool_distribution).map(([tool, count]) => {
                      const maxCount = Math.max(...Object.values(report.metrics.tool_distribution));
                      const width = maxCount > 0 ? (count / maxCount) * 100 : 0;
                      return (
                        <div key={tool} className="space-y-1">
                          <div className="flex items-center justify-between text-sm">
                            <Badge variant="outline">{tool}</Badge>
                            <span className="text-muted-foreground">{count}</span>
                          </div>
                          <div className="h-2 rounded-full bg-muted overflow-hidden">
                            <div className="h-full bg-emerald-500" style={{ width: `${width}%` }} />
                          </div>
                        </div>
                      );
                    })
                  )}
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Source Coverage</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Most cited manual pages across this run.
                </p>
              </CardHeader>
              <CardContent className="space-y-2">
                {topSources.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No sources captured yet.</p>
                ) : (
                  topSources.map((src) => (
                    <div key={`${src.manual}-${src.page}`} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{src.manual}</Badge>
                        <span className="text-sm">Page {src.page}</span>
                        {src.type && (
                          <span className="text-[10px] uppercase text-muted-foreground">• {src.type}</span>
                        )}
                      </div>
                      <span className="text-sm text-muted-foreground">{src.count} hits</span>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Per-Query Results</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Expand a query to see the model answer, judge rationale, and cited sources. Red rows flag low scores or errors.
                </p>
              </CardHeader>
              <CardContent className="space-y-3">
                {report.records.map((record) => {
                  const isIssue = !!record.error || record.judge_score < 0.6;
                  return (
                    <div
                      key={record.id}
                      className={`rounded-lg border ${isIssue ? 'border-destructive/50 bg-destructive/5' : 'border-muted'}`}
                    >
                      <button
                        className="w-full text-left p-4 flex items-start gap-3"
                        onClick={() => toggleExpanded(record.id)}
                      >
                        <Badge variant={record.error ? 'destructive' : 'secondary'}>
                          {record.error ? 'Error' : `Score ${record.judge_score.toFixed(2)}`}
                        </Badge>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium line-clamp-2">{record.query}</p>
                          <p className="text-xs text-muted-foreground mt-1 flex flex-wrap gap-2">
                            <span>{formatMs(record.latency_ms)}</span>
                            <span>• Iter {record.iterations}</span>
                            <span>• Tools {record.tools_used.length ? record.tools_used.join(', ') : '—'}</span>
                            {record.trace_id && <span>• Trace {record.trace_id}</span>}
                          </p>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {expanded[record.id] ? 'Hide' : 'Details'}
                        </span>
                      </button>

                      {expanded[record.id] && (
                        <div className="border-t px-4 py-3 space-y-3 bg-muted/30">
                          {record.error && (
                            <p className="text-sm text-destructive">Error: {record.error}</p>
                          )}
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            <div className="space-y-1">
                              <p className="text-xs uppercase text-muted-foreground">Model Answer</p>
                              <p className="text-sm whitespace-pre-wrap">{record.model_answer || 'No answer captured.'}</p>
                            </div>
                            <div className="space-y-1">
                              <p className="text-xs uppercase text-muted-foreground">Expected</p>
                              <p className="text-sm whitespace-pre-wrap text-muted-foreground">{record.expected_answer}</p>
                            </div>
                          </div>
                          <div className="space-y-1">
                            <p className="text-xs uppercase text-muted-foreground">Judge Rationale</p>
                            <p className="text-sm whitespace-pre-wrap">{record.judge_rationale}</p>
                          </div>
                          <div className="space-y-2">
                            <p className="text-xs uppercase text-muted-foreground">Sources</p>
                            <div className="flex flex-wrap gap-2">
                              {record.sources.length === 0 && (
                                <span className="text-sm text-muted-foreground">No sources captured.</span>
                              )}
                              {record.sources.map((src, idx) => (
                                <Badge key={`${record.id}-src-${idx}`} variant="outline" className="text-xs">
                                  {src.manual || 'Manual'} • Page {src.page}
                                  {src.type && <span className="ml-1 uppercase text-[10px] text-muted-foreground">{src.type}</span>}
                                  {typeof src.score === 'number' && <span className="ml-1 text-[10px] text-muted-foreground">{src.score.toFixed(2)}</span>}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </>
        ) : null}
      </main>
    </div>
  );
}
