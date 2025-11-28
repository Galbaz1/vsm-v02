'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { listBenchmarkReports } from '@/lib/api';
import type { BenchmarkListItem, SingleBenchmarkResult } from '@/lib/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';

function formatTimestamp(ts: string) {
  if (!ts) return 'Unknown';
  const year = ts.slice(0, 4);
  const month = ts.slice(4, 6);
  const day = ts.slice(6, 8);
  const time = ts.slice(9, 15);
  return `${year}-${month}-${day} ${time}`;
}

export default function BenchmarkLanding() {
  const [reports, setReports] = useState<BenchmarkListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastSingleResult, setLastSingleResult] = useState<SingleBenchmarkResult | null>(null);

  useEffect(() => {
    listBenchmarkReports(50)
      .then(setReports)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load reports'))
      .finally(() => setLoading(false));
    
    // Check for single benchmark result from search page
    const savedResult = sessionStorage.getItem('lastBenchmarkResult');
    if (savedResult) {
      try {
        setLastSingleResult(JSON.parse(savedResult));
      } catch (e) {
        console.error('Failed to parse saved benchmark result:', e);
      }
    }
  }, []);

  const latest = useMemo(() => reports[0], [reports]);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-6 flex items-center justify-between">
          <div>
            <p className="text-sm uppercase tracking-wide text-muted-foreground">Phase 8</p>
            <h1 className="text-3xl font-bold">Benchmark Dashboard</h1>
            <p className="text-muted-foreground mt-2">
              Compare local vs cloud runs with quality, latency, and stability signals.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link href="/" className="text-sm font-medium underline underline-offset-4">
              Back to Search
            </Link>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Single Query Result from Search Page */}
        {lastSingleResult && (
          <Card className="border-2 border-primary/50 bg-primary/5">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-primary rounded-full animate-pulse" />
                  Just Evaluated
                </CardTitle>
                <button
                  onClick={() => {
                    sessionStorage.removeItem('lastBenchmarkResult');
                    setLastSingleResult(null);
                  }}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Dismiss
                </button>
              </div>
              <p className="text-sm text-muted-foreground line-clamp-2">
                {lastSingleResult.query}
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div className="p-3 rounded-lg border bg-background">
                  <p className="text-xs text-muted-foreground">Score</p>
                  <p className={`text-2xl font-bold ${
                    lastSingleResult.judge_score >= 0.7 ? 'text-emerald-600' :
                    lastSingleResult.judge_score >= 0.4 ? 'text-amber-600' : 'text-red-600'
                  }`}>
                    {Math.round(lastSingleResult.judge_score * 100)}%
                  </p>
                </div>
                <div className="p-3 rounded-lg border bg-background">
                  <p className="text-xs text-muted-foreground">Latency</p>
                  <p className="text-2xl font-bold">{Math.round(lastSingleResult.latency_ms)}ms</p>
                </div>
                <div className="p-3 rounded-lg border bg-background">
                  <p className="text-xs text-muted-foreground">Iterations</p>
                  <p className="text-2xl font-bold">{lastSingleResult.iterations}</p>
                </div>
                <div className="p-3 rounded-lg border bg-background">
                  <p className="text-xs text-muted-foreground">Sources</p>
                  <p className="text-2xl font-bold">{lastSingleResult.sources.length}</p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Model Answer</p>
                  <p className="bg-muted/50 p-3 rounded-lg text-foreground line-clamp-4">
                    {lastSingleResult.model_answer || '(No answer generated)'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Expected Answer</p>
                  <p className="bg-muted/50 p-3 rounded-lg text-foreground line-clamp-4">
                    {lastSingleResult.expected_answer}
                  </p>
                </div>
              </div>
              
              <div className="mt-4">
                <p className="text-xs text-muted-foreground mb-1">Judge Rationale</p>
                <p className="text-sm text-foreground">{lastSingleResult.judge_rationale}</p>
              </div>
              
              <div className="mt-4 flex flex-wrap gap-2">
                <span className="text-xs text-muted-foreground">Tools used:</span>
                {lastSingleResult.tools_used.map((tool, i) => (
                  <Badge key={i} variant="outline" className="text-xs">
                    {tool}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Latest Run
                {latest?.mode && (
                  <Badge variant="secondary" className="uppercase">
                    {latest.mode}
                  </Badge>
                )}
              </CardTitle>
              <p className="text-muted-foreground text-sm">
                {latest ? formatTimestamp(latest.timestamp) : 'No runs yet'}
              </p>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {loading ? (
                <>
                  <Skeleton className="h-20 w-full" />
                  <Skeleton className="h-20 w-full" />
                  <Skeleton className="h-20 w-full" />
                </>
              ) : latest ? (
                <>
                  <div className="p-4 rounded-lg border bg-muted/30">
                    <p className="text-xs text-muted-foreground">Average Score</p>
                    <p className="text-2xl font-semibold">
                      {typeof latest.summary?.avg_score === 'string'
                        ? Number(latest.summary.avg_score).toFixed(2)
                        : '—'}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">Hit quality via TechnicalJudge</p>
                  </div>
                  <div className="p-4 rounded-lg border bg-muted/30">
                    <p className="text-xs text-muted-foreground">Latency p50</p>
                    <p className="text-2xl font-semibold">
                      {typeof latest.summary?.latency_p50 === 'string'
                        ? latest.summary.latency_p50
                        : '—'}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">Median end-to-end</p>
                  </div>
                  <div className="p-4 rounded-lg border bg-muted/30">
                    <p className="text-xs text-muted-foreground">Completion</p>
                    <p className="text-2xl font-semibold">
                      {latest.completed}/{latest.total}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">Finished queries</p>
                  </div>
                </>
              ) : (
                <div className="col-span-3 text-muted-foreground">No reports yet.</div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Run Insights</CardTitle>
              <p className="text-muted-foreground text-sm">
                Quick health indicators pulled from the most recent run.
              </p>
            </CardHeader>
            <CardContent className="space-y-3">
              {loading ? (
                <>
                  <Skeleton className="h-4 w-2/3" />
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-4 w-3/4" />
                </>
              ) : latest ? (
                <>
                  <div className="flex items-center justify-between text-sm">
                    <span>Mode</span>
                    <Badge variant="outline">{latest.mode}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span>Files</span>
                    <span className="text-muted-foreground truncate">{latest.filename || 'logs/benchmarks'}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span>Success Rate</span>
                    <span className="font-medium">
                      {latest.total > 0 ? `${Math.round((latest.completed / latest.total) * 100)}%` : '—'}
                    </span>
                  </div>
                </>
              ) : (
                <p className="text-muted-foreground text-sm">No telemetry yet.</p>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Historical Runs</CardTitle>
            <p className="text-sm text-muted-foreground">
              Click into a run to see per-query traces, scores, and latencies.
            </p>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-2">
                {[...Array(4)].map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : error ? (
              <div className="text-destructive text-sm">Failed to load reports: {error}</div>
            ) : reports.length === 0 ? (
              <div className="text-muted-foreground text-sm">No reports available yet.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="text-left text-muted-foreground">
                    <tr>
                      <th className="py-2 pr-4">Timestamp</th>
                      <th className="py-2 pr-4">Mode</th>
                      <th className="py-2 pr-4">Avg Score</th>
                      <th className="py-2 pr-4">Latency p50</th>
                      <th className="py-2 pr-4">Completed</th>
                      <th className="py-2">Details</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {reports.map((report) => (
                      <tr key={report.filename}>
                        <td className="py-3 pr-4 font-medium">{formatTimestamp(report.timestamp)}</td>
                        <td className="py-3 pr-4">
                          <Badge variant="outline" className="uppercase">
                            {report.mode}
                          </Badge>
                        </td>
                        <td className="py-3 pr-4">
                          {typeof report.summary?.avg_score === 'string'
                            ? Number(report.summary.avg_score).toFixed(2)
                            : '—'}
                        </td>
                        <td className="py-3 pr-4">
                          {typeof report.summary?.latency_p50 === 'string'
                            ? report.summary.latency_p50
                            : '—'}
                        </td>
                        <td className="py-3 pr-4">
                          {report.completed}/{report.total}
                        </td>
                        <td className="py-3">
                          <Link
                            href={`/benchmark/${encodeURIComponent(report.filename)}`}
                            className="text-primary hover:underline"
                          >
                            View
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
