import axios from 'axios';
import type { 
  SearchResponse, 
  SearchQueryParams, 
  AgenticStreamEvent,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8001';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function searchManual(
  query: string,
  limit: number = 5,
  chunkType?: string,
  groupByPage: boolean = false
): Promise<SearchResponse> {
  const params: SearchQueryParams = { query, limit, group_by_page: groupByPage };
  if (chunkType) {
    params.chunk_type = chunkType;
  }
  const response = await apiClient.get<SearchResponse>('/search', { params });
  return response.data;
}

/**
 * Stream agentic search results via NDJSON.
 * 
 * @param query - User's search query
 * @param onEvent - Callback for each streamed event
 * @param onError - Callback for errors
 * @param onComplete - Callback when stream completes
 * @returns AbortController to cancel the stream
 */
export function streamAgenticSearch(
  query: string,
  onEvent: (event: AgenticStreamEvent) => void,
  onError?: (error: Error) => void,
  onComplete?: () => void,
): AbortController {
  const controller = new AbortController();
  const url = `${API_BASE_URL}/agentic_search?query=${encodeURIComponent(query)}`;
  
  fetch(url, {
    method: 'GET',
    headers: {
      'Accept': 'application/x-ndjson',
    },
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }
      
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          // Process any remaining buffer
          if (buffer.trim()) {
            try {
              const event = JSON.parse(buffer.trim()) as AgenticStreamEvent;
              onEvent(event);
            } catch {
              // Ignore incomplete JSON
            }
          }
          break;
        }
        
        // Decode and process NDJSON lines
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last potentially incomplete line in buffer
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const event = JSON.parse(line) as AgenticStreamEvent;
              onEvent(event);
            } catch (e) {
              console.warn('Failed to parse NDJSON line:', line, e);
            }
          }
        }
      }
      
      onComplete?.();
    })
    .catch((error) => {
      if (error.name === 'AbortError') {
        // Stream was cancelled
        return;
      }
      onError?.(error);
    });
  
  return controller;
}

