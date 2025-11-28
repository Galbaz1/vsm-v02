// TypeScript types matching FastAPI response models

export interface BoundingBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface SearchHit {
  anchor_id: string;
  manual_name: string;
  content: string;
  page_number: number | null;
  score?: number | null;
  bbox: BoundingBox | null;
  pdf_page_url: string;
  page_image_url: string | null;
  chunk_type?: string | null;
  section_title?: string | null;
  content_hash?: string | null;
}

export interface PageHit {
  page_number: number;
  manual_name: string;
  hits: SearchHit[];
  bboxes: BoundingBox[];
}

export interface SearchResponse {
  query: string;
  hits: SearchHit[];
  page_hits?: Record<string, PageHit> | null;
}

export interface SearchQueryParams {
  query: string;
  limit?: number;
  chunk_type?: string;
  group_by_page?: boolean;
}

// Agentic search types
export type AgenticPayloadType = 
  | 'result' 
  | 'error' 
  | 'status' 
  | 'response' 
  | 'decision' 
  | 'complete';

export interface AgenticResultPayload {
  objects: Record<string, unknown>[];
  metadata: Record<string, unknown>;
  name: string;
  count: number;
}

export interface AgenticErrorPayload {
  message: string;
  recoverable: boolean;
  suggestion?: string;
}

export interface AgenticStatusPayload {
  message: string;
}

export interface AgenticResponsePayload {
  text: string;
  sources?: SourceRef[];
}

export interface AgenticDecisionPayload {
  tool: string;
  reasoning: string;
}

export interface AgenticCompletePayload {}

export interface AgenticStreamEvent {
  type: AgenticPayloadType;
  query_id: string;
  payload: 
    | AgenticResultPayload 
    | AgenticErrorPayload 
    | AgenticStatusPayload 
    | AgenticResponsePayload 
    | AgenticDecisionPayload 
    | AgenticCompletePayload;
}

// Parsed result objects for display
export interface TextChunkResult {
  content: string;
  manual_name: string;
  page_number: number;
  score?: number | null;
  chunk_type?: string;
  section_title?: string;
  bbox?: BoundingBox;
  page_image_url?: string;
  pdf_page_url?: string;
}

export interface VisualPageResult {
  page_number: number;
  asset_manual: string;
  maxsim_score: number;
  image_path: string;
  score?: number;
  preview_url?: string;
}

export interface VisualInterpretationResult {
  page_number: number;
  asset_manual: string;
  interpretation: string;
  image_path?: string;
  error?: string;
}

// Aggregated state for agentic search
export interface AgenticSearchState {
  query_id: string | null;
  status: string;
  isLoading: boolean;
  isComplete: boolean;
  error: string | null;
  
  // Results
  textResults: TextChunkResult[];
  visualResults: VisualPageResult[];
  visualInterpretations: VisualInterpretationResult[];
  
  // Final response
  response: string | null;
  sources: SourceRef[];
  
  // Decision transparency
  decisions: { tool: string; reasoning: string }[];
}

export interface SourceRef {
  manual?: string;
  page: number | string;
  type?: 'text' | 'visual';
  score?: number;
  preview_url?: string;
  pdf_page_url?: string;
  origin?: string;
}

// Benchmarking
export interface BenchmarkMetrics {
  avg_score: number;
  hit_at_1: number;
  hit_at_3: number;
  mrr: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  total_queries: number;
  completed: number;
  failed: number;
  avg_iterations: number;
  tool_distribution: Record<string, number>;
}

export interface BenchmarkRecord {
  id: string;
  query: string;
  expected_answer: string;
  model_answer: string;
  sources: SourceRef[];
  judge_score: number;
  judge_rationale: string;
  cited_pages: (string | number)[];
  latency_ms: number;
  iterations: number;
  tools_used: string[];
  trace_id?: string | null;
  error?: string | null;
}

export interface BenchmarkRun {
  mode: string;
  dataset: string;
  timestamp: string;
  total: number;
  completed: number;
  summary?: Record<string, unknown>;
  metrics: BenchmarkMetrics;
  records: BenchmarkRecord[];
  report_path?: string;
  filename?: string;
}

export interface BenchmarkListItem {
  filename: string;
  mode: string;
  timestamp: string;
  total: number;
  completed: number;
  summary?: Record<string, unknown>;
}

export interface BenchmarkQuestion {
  id: string;
  category: string;
  query: string;
  expected_answer: string;
}

export interface SingleBenchmarkResult {
  id: string;
  query: string;
  expected_answer: string;
  model_answer: string;
  sources: SourceRef[];
  judge_score: number;
  judge_rationale: string;
  cited_pages: (string | number)[];
  latency_ms: number;
  iterations: number;
  tools_used: string[];
  trace_id?: string | null;
  error?: string | null;
  mode: string;
}
