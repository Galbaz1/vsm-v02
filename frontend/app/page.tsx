'use client';

import { useState, useMemo } from 'react';
import { SearchBar } from '@/components/SearchBar';
import { ResultCard } from '@/components/ResultCard';
import { PreviewPanel } from '@/components/PreviewPanel';
import { useManualSearch } from '@/lib/hooks/useManualSearch';
import { useDebounce } from '@/lib/hooks/useDebounce';
import { Skeleton } from '@/components/ui/skeleton';
import type { SearchHit } from '@/lib/types';

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [chunkType, setChunkType] = useState<string | undefined>(undefined);
  const [groupByPage, setGroupByPage] = useState(false);
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  const debouncedQuery = useDebounce(searchQuery, 400);
  const { data, isLoading, error } = useManualSearch(debouncedQuery, 10, chunkType, groupByPage);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
  };

  const handleChunkTypeChange = (type: string | undefined) => {
    setChunkType(type);
  };

  const handlePreview = (hit: SearchHit) => {
    setSelectedHit(hit);
    setPreviewOpen(true);
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
          <h1 className="text-3xl font-bold">Manual Search</h1>
          <p className="text-muted-foreground mt-2">
            Semantic search over technical asset manuals with visual grounding
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <SearchBar
            onSearch={handleSearch}
            onChunkTypeChange={handleChunkTypeChange}
            isLoading={isLoading}
            isAutoRefresh={debouncedQuery.length >= 3}
          />
        </div>

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
          </div>
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
