import axios from 'axios';
import type {
  SearchResponse, 
  SearchQueryParams, 
  AgenticStreamEvent,
  BenchmarkListItem,
  BenchmarkRun,
  BenchmarkQuestion,
  SingleBenchmarkResult,
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

export async function listBenchmarkReports(limit: number = 20): Promise<BenchmarkListItem[]> {
  const response = await apiClient.get<BenchmarkListItem[]>('/benchmark/reports', {
    params: { limit },
  });
  return response.data;
}

export async function fetchBenchmarkReport(filename: string): Promise<BenchmarkRun> {
  const response = await apiClient.get<BenchmarkRun>(`/benchmark/reports/${filename}`);
  return response.data;
}

export async function fetchBenchmarkQuestions(limit: number = 5): Promise<BenchmarkQuestion[]> {
  const response = await apiClient.get<BenchmarkQuestion[]>('/benchmark/questions', {
    params: { limit },
  });
  return response.data;
}

export async function runSingleBenchmark(
  query: string,
  expectedAnswer: string,
  queryId?: string,
): Promise<SingleBenchmarkResult> {
  const response = await apiClient.post<SingleBenchmarkResult>('/benchmark/evaluate-single', {
    query,
    expected_answer: expectedAnswer,
    query_id: queryId,
    enable_tracing: true,
  });
  return response.data;
}

// Chat API
export interface ChatSession {
  session_id: string;
  message_count: number;
  last_query?: string;
  has_data: boolean;
}

export interface ChatSessionDetail {
  session_id: string;
  conversation_history: { role: string; content: string }[];
  environment_summary: string;
  tasks_completed_count: number;
}

export function streamChat(
  message: string,
  sessionId: string | null,
  onEvent: (event: AgenticStreamEvent & { type: 'session'; payload: { is_new: boolean; message_count: number }; session_id: string }) => void,
  onError: (error: Error) => void,
  onComplete: () => void,
): AbortController {
  const controller = new AbortController();
  
  const fetchStream = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
        signal: controller.signal,
      });
      
      if (!response.ok) {
        throw new Error(`Chat failed: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');
      
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const event = JSON.parse(line);
              onEvent(event);
            } catch (e) {
              console.warn('Failed to parse chat event:', line);
            }
          }
        }
      }
      
      onComplete();
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        onError(error as Error);
      }
    }
  };
  
  fetchStream();
  return controller;
}

export async function listChatSessions(): Promise<{ sessions: ChatSession[] }> {
  const response = await apiClient.get<{ sessions: ChatSession[] }>('/chat/sessions');
  return response.data;
}

export async function getChatSession(sessionId: string): Promise<ChatSessionDetail> {
  const response = await apiClient.get<ChatSessionDetail>(`/chat/sessions/${sessionId}`);
  return response.data;
}

export async function deleteChatSession(sessionId: string): Promise<{ deleted: boolean }> {
  const response = await apiClient.delete<{ deleted: boolean }>(`/chat/sessions/${sessionId}`);
  return response.data;
}

export async function createChatSession(): Promise<{ session_id: string }> {
  const response = await apiClient.post<{ session_id: string }>('/chat/sessions');
  return response.data;
}
