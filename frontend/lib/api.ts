import axios from 'axios';
import type { SearchResponse, SearchQueryParams } from './types';

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

