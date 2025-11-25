'use client';

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { ExternalLink, X, ChevronLeft, ChevronRight } from 'lucide-react';
import type { SearchHit, PageHit } from '@/lib/types';
import { useState, useEffect, useMemo } from 'react';
import { cleanContent } from './ResultCard';

interface PreviewPanelProps {
  hit: SearchHit | null;
  pageHits?: Record<string, PageHit> | null;
  open: boolean;
  onClose: () => void;
}

const BBOX_COLORS = [
  'border-blue-500 bg-blue-500/20',
  'border-green-500 bg-green-500/20',
  'border-yellow-500 bg-yellow-500/20',
  'border-purple-500 bg-purple-500/20',
  'border-red-500 bg-red-500/20',
];

export function PreviewPanel({ hit, pageHits, open, onClose }: PreviewPanelProps) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [currentHitIndex, setCurrentHitIndex] = useState(0);

  useEffect(() => {
    if (open && hit) {
      setImageLoaded(false);
      setImageError(false);
      setCurrentHitIndex(0);
    }
  }, [open, hit]);

  if (!hit) return null;

  // Get all hits for this page if available
  const pageHitKey = hit.page_number !== null
    ? `${hit.manual_name}:${hit.page_number}`
    : null;
  const pageHit = pageHitKey && pageHits ? pageHits[pageHitKey] : null;
  const allHitsOnPage = pageHit?.hits || [hit];
  const currentHit = allHitsOnPage[currentHitIndex] || hit;
  const canNavigate = allHitsOnPage.length > 1;

  const handlePrevious = () => {
    if (currentHitIndex > 0) {
      setCurrentHitIndex(currentHitIndex - 1);
    }
  };

  const handleNext = () => {
    if (currentHitIndex < allHitsOnPage.length - 1) {
      setCurrentHitIndex(currentHitIndex + 1);
    }
  };

  const bbox = currentHit.bbox;
  const hasBbox = bbox && currentHit.page_image_url;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{currentHit.manual_name}</DialogTitle>
          <DialogDescription>
            Page {currentHit.page_number !== null ? currentHit.page_number + 1 : 'unknown'}
            {canNavigate && (
              <span className="ml-2">
                ({currentHitIndex + 1} of {allHitsOnPage.length})
              </span>
            )}
            {currentHit.section_title && (
              <span className="ml-2 text-xs">• {currentHit.section_title}</span>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {currentHit.page_image_url ? (
            <div className="relative border rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800">
              {!imageLoaded && !imageError && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="animate-pulse text-muted-foreground">Loading preview...</div>
                </div>
              )}
              {imageError && (
                <div className="absolute inset-0 flex items-center justify-center text-destructive">
                  Failed to load preview image
                </div>
              )}
              <img
                src={currentHit.page_image_url}
                alt={`Page ${currentHit.page_number !== null ? currentHit.page_number + 1 : ''} preview`}
                className={`w-full h-auto ${imageLoaded ? 'block' : 'hidden'}`}
                onLoad={() => setImageLoaded(true)}
                onError={() => {
                  setImageError(true);
                  setImageLoaded(false);
                }}
              />
              {imageLoaded && pageHit && pageHit.bboxes.length > 0 && (
                <>
                  {pageHit.bboxes.map((bbox, idx) => {
                    const isActive = idx === currentHitIndex;
                    const colorClass = BBOX_COLORS[idx % BBOX_COLORS.length];
                    return (
                      <div
                        key={idx}
                        className={`absolute border-2 pointer-events-none ${
                          isActive ? colorClass : 'border-gray-400 bg-gray-400/10'
                        }`}
                        style={{
                          left: `${bbox.left * 100}%`,
                          top: `${bbox.top * 100}%`,
                          width: `${(bbox.right - bbox.left) * 100}%`,
                          height: `${(bbox.bottom - bbox.top) * 100}%`,
                        }}
                        title={`Highlight ${idx + 1}${isActive ? ' (active)' : ''}`}
                      />
                    );
                  })}
                </>
              )}
              {imageLoaded && !pageHit && hasBbox && (
                <div
                  className="absolute border-2 border-blue-500 bg-blue-500/20 pointer-events-none"
                  style={{
                    left: `${bbox.left * 100}%`,
                    top: `${bbox.top * 100}%`,
                    width: `${(bbox.right - bbox.left) * 100}%`,
                    height: `${(bbox.bottom - bbox.top) * 100}%`,
                  }}
                  title="Highlighted region"
                />
              )}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              Preview image not available
            </div>
          )}

          {canNavigate && (
            <div className="flex items-center justify-between gap-4">
              <Button
                variant="outline"
                size="sm"
                onClick={handlePrevious}
                disabled={currentHitIndex === 0}
              >
                <ChevronLeft className="h-4 w-4 mr-2" />
                Previous
              </Button>
              <span className="text-sm text-muted-foreground">
                Hit {currentHitIndex + 1} of {allHitsOnPage.length} on this page
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleNext}
                disabled={currentHitIndex === allHitsOnPage.length - 1}
              >
                Next
                <ChevronRight className="h-4 w-4 ml-2" />
              </Button>
            </div>
          )}

          {allHitsOnPage.length > 1 && (
            <div className="space-y-2">
              <p className="text-sm font-medium">All snippets on this page:</p>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {allHitsOnPage.map((h, idx) => (
                  <div
                    key={h.anchor_id}
                    className={`p-2 rounded border cursor-pointer transition-colors ${
                      idx === currentHitIndex
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:bg-muted'
                    }`}
                    onClick={() => setCurrentHitIndex(idx)}
                  >
                    <p className="text-xs text-muted-foreground line-clamp-2">
                      {cleanContent(h.content)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="text-sm text-muted-foreground whitespace-pre-wrap border-t pt-4">
            <p className="font-medium mb-2">Current snippet:</p>
            {cleanContent(currentHit.content)}
          </div>

          <div className="flex gap-2 justify-end">
            <Button variant="outline" onClick={onClose}>
              <X className="h-4 w-4 mr-2" />
              Close
            </Button>
            <Button variant="default" asChild>
              <a href={currentHit.pdf_page_url} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4 mr-2" />
                Open PDF
              </a>
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

