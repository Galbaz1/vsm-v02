'use client';

import { useQuery } from '@tanstack/react-query';
import { searchManual } from '../api';
import type { SearchResponse } from '../types';

export function useManualSearch(
  query: string,
  limit: number = 5,
  chunkType?: string,
  groupByPage: boolean = false
) {
  return useQuery<SearchResponse>({
    queryKey: ['manualSearch', query, limit, chunkType, groupByPage],
    queryFn: () => searchManual(query, limit, chunkType, groupByPage),
    enabled: query.length >= 3, // Only search if query is at least 3 characters
    staleTime: 30000, // Cache for 30 seconds
  });
}

