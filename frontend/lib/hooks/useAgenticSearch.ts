'use client';

import { useState, useCallback, useRef } from 'react';
import { streamAgenticSearch } from '../api';
import type { 
  AgenticSearchState, 
  AgenticStreamEvent,
  SourceRef,
  TextChunkResult,
  VisualPageResult,
  VisualInterpretationResult,
} from '../types';

const initialState: AgenticSearchState = {
  query_id: null,
  status: '',
  isLoading: false,
  isComplete: false,
  error: null,
  textResults: [],
  visualResults: [],
  visualInterpretations: [],
  response: null,
  sources: [],
  decisions: [],
};

/**
 * Hook for managing agentic search with streaming updates.
 * 
 * Handles all payload types from the agent:
 * - status: Progress updates
 * - decision: Tool selection transparency  
 * - result: Search results (text, visual, interpretations)
 * - response: Final synthesized answer
 * - error: Recoverable and non-recoverable errors
 * - complete: Stream finished
 */
export function useAgenticSearch() {
  const [state, setState] = useState<AgenticSearchState>(initialState);
  const abortControllerRef = useRef<AbortController | null>(null);
  
  const search = useCallback((query: string) => {
    // Cancel any existing search
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Reset state
    setState({
      ...initialState,
      isLoading: true,
      status: 'Starting search...',
    });
    
    const handleEvent = (event: AgenticStreamEvent) => {
      setState((prev) => {
        const next = { ...prev, query_id: event.query_id };
        
        switch (event.type) {
          case 'status': {
            const payload = event.payload as { message: string };
            next.status = payload.message;
            break;
          }
          
          case 'decision': {
            const payload = event.payload as { tool: string; reasoning: string };
            next.decisions = [...prev.decisions, payload];
            next.status = `Using ${payload.tool}...`;
            break;
          }
          
          case 'result': {
            const payload = event.payload as {
              objects: Record<string, unknown>[];
              name: string;
              metadata?: Record<string, unknown>;
            };
            
            // Route results based on source name
              if (payload.name === 'AssetManual' || payload.name === 'hybrid_search') {
                // Text search results
                const textChunks = payload.objects
                  .filter((obj) => 'content' in obj)
                  .map((obj) => ({
                    content: obj.content as string,
                    manual_name: (obj.manual_name || obj.asset_manual || 'Manual') as string,
                    page_number: obj.page_number as number,
                    score: (obj.score as number) ?? null,
                    chunk_type: obj.chunk_type as string | undefined,
                    section_title: obj.section_title as string | undefined,
                    bbox: obj.bbox as TextChunkResult['bbox'],
                    page_image_url: obj.page_image_url as string | undefined,
                    pdf_page_url: obj.pdf_page_url as string | undefined,
                  }));
                next.textResults = [...prev.textResults, ...textChunks];
                
                // Handle hybrid search visual pages
                if (payload.name === 'hybrid_search' && payload.objects[0]) {
                const hybrid = payload.objects[0] as {
                  text_chunks?: unknown[];
                  visual_pages?: unknown[];
                };
                if (hybrid.visual_pages) {
                  const visualPages = (hybrid.visual_pages as Record<string, unknown>[]).map((obj) => ({
                    page_number: obj.page_number as number,
                    asset_manual: (obj.asset_manual || 'Manual') as string,
                    maxsim_score: (obj.maxsim_score as number) ?? (obj.score as number),
                    image_path: obj.image_path as string,
                    preview_url: obj.preview_url as string | undefined,
                    score: (obj.score as number) ?? undefined,
                  }));
                  next.visualResults = [...prev.visualResults, ...visualPages];
                }
              }
            } else if (payload.name === 'PDFDocuments') {
              // ColQwen visual results
              const visualPages = payload.objects.map((obj) => ({
                page_number: obj.page_number as number,
                asset_manual: (obj.asset_manual || 'Manual') as string,
                maxsim_score: (obj.maxsim_score as number) ?? (obj.score as number),
                image_path: obj.image_path as string,
                preview_url: obj.preview_url as string | undefined,
                score: (obj.score as number) ?? undefined,
              }));
              next.visualResults = [...prev.visualResults, ...visualPages];
            } else if (payload.name === 'visual_interpretations') {
              // VLM interpretations
              const interpretations = payload.objects.map((obj) => ({
                page_number: obj.page_number as number,
                asset_manual: (obj.asset_manual || 'Manual') as string,
                interpretation: obj.interpretation as string,
                image_path: obj.image_path as string | undefined,
                error: obj.error as string | undefined,
              }));
              next.visualInterpretations = [...prev.visualInterpretations, ...interpretations];
            }
            break;
          }
          
          case 'response': {
            const payload = event.payload as {
              text: string;
              sources?: SourceRef[];
            };
            next.response = payload.text;
            next.sources = payload.sources || [];
            break;
          }
          
          case 'error': {
            const payload = event.payload as {
              message: string;
              recoverable: boolean;
              suggestion?: string;
            };
            if (!payload.recoverable) {
              next.error = payload.message;
            }
            next.status = `Error: ${payload.message}`;
            break;
          }
          
          case 'complete': {
            next.isLoading = false;
            next.isComplete = true;
            next.status = 'Complete';
            break;
          }
        }
        
        return next;
      });
    };
    
    const handleError = (error: Error) => {
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: error.message,
        status: `Error: ${error.message}`,
      }));
    };
    
    const handleComplete = () => {
      setState((prev) => ({
        ...prev,
        isLoading: false,
        isComplete: true,
      }));
    };
    
    abortControllerRef.current = streamAgenticSearch(
      query,
      handleEvent,
      handleError,
      handleComplete,
    );
  }, []);
  
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setState((prev) => ({
        ...prev,
        isLoading: false,
        status: 'Cancelled',
      }));
    }
  }, []);
  
  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setState(initialState);
  }, []);
  
  // Restore state from a saved snapshot (for persistence)
  const restore = useCallback((savedState: Partial<AgenticSearchState>) => {
    setState({
      ...initialState,
      ...savedState,
      isComplete: true,
      isLoading: false,
    });
  }, []);
  
  // Get current state for saving
  const getState = useCallback((): AgenticSearchState => state, [state]);
  
  return {
    ...state,
    search,
    cancel,
    reset,
    restore,
    getState,
  };
}
