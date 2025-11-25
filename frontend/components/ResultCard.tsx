'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ExternalLink, FileText } from 'lucide-react';
import type { SearchHit } from '@/lib/types';
import { highlightText } from '@/lib/utils/highlight';

interface ResultCardProps {
  hit: SearchHit;
  query: string;
  onPreview: (hit: SearchHit) => void;
}

// Clean HTML anchor tags and other unwanted HTML from content
export function cleanContent(content: string): string {
  // Remove HTML anchor tags like <a id='...'></a> and empty anchor tags
  let cleaned = content.replace(/<a\s+id=['"][^'"]*['"][^>]*><\/a>\s*/gi, '');
  // Remove any remaining empty anchor tags
  cleaned = cleaned.replace(/<a[^>]*><\/a>\s*/gi, '');
  // Clean up multiple newlines
  cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
  // Trim whitespace
  return cleaned.trim();
}

// Format chunk type for display
function formatChunkType(chunkType: string | null | undefined): string {
  if (!chunkType) return '';
  return chunkType.charAt(0).toUpperCase() + chunkType.slice(1).toLowerCase();
}

export function ResultCard({ hit, query, onPreview }: ResultCardProps) {
  const cleanedContent = cleanContent(hit.content);
  const highlightedContent = highlightText(cleanedContent, query);
  
  const sourceInfo = [];
  if (hit.page_number !== null && hit.page_number !== undefined) {
    sourceInfo.push(`Page ${hit.page_number + 1}`);
  }

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-lg mb-1">{hit.manual_name}</CardTitle>
            {hit.section_title && (
              <p className="text-sm font-medium text-muted-foreground mb-2">
                {hit.section_title}
              </p>
            )}
            <div className="flex flex-wrap items-center gap-2">
              {sourceInfo.length > 0 && (
                <CardDescription className="text-xs">
                  {sourceInfo.join(' • ')}
                </CardDescription>
              )}
              {hit.chunk_type && (
                <Badge variant="secondary" className="text-xs">
                  {formatChunkType(hit.chunk_type)}
                </Badge>
              )}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div
          className="text-sm text-muted-foreground mb-4 line-clamp-4 prose prose-sm max-w-none"
          dangerouslySetInnerHTML={{ __html: highlightedContent }}
        />
        <div className="flex flex-wrap gap-2 pt-2 border-t">
          {hit.page_image_url ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onPreview(hit)}
            >
              <FileText className="h-4 w-4 mr-2" />
              Preview Page
            </Button>
          ) : null}
          <Button
            variant="outline"
            size="sm"
            asChild
          >
            <a href={hit.pdf_page_url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-4 w-4 mr-2" />
              Open PDF
            </a>
          </Button>
          {hit.page_number !== null && hit.page_number !== undefined && (
            <Button
              variant="ghost"
              size="sm"
              asChild
            >
              <a href={hit.pdf_page_url} target="_blank" rel="noopener noreferrer">
                Page {hit.page_number + 1}
              </a>
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

