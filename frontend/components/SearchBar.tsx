'use client';

import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Search, RefreshCw } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  onChunkTypeChange?: (chunkType: string | undefined) => void;
  isLoading?: boolean;
  initialQuery?: string;
  isAutoRefresh?: boolean;
}

export function SearchBar({
  onSearch,
  onChunkTypeChange,
  isLoading,
  initialQuery = '',
  isAutoRefresh = false,
}: SearchBarProps) {
  const [query, setQuery] = useState(initialQuery);
  const [chunkType, setChunkType] = useState<string>('');

  const handleChange = (value: string) => {
    setQuery(value);
    if (isAutoRefresh && value.trim().length >= 3) {
      onSearch(value.trim());
    }
  };

  const handleChunkTypeChange = (value: string) => {
    const selectedType = value === 'all' ? undefined : value;
    setChunkType(value);
    if (onChunkTypeChange) {
      onChunkTypeChange(selectedType);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim().length >= 3) {
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      <div className="flex flex-col gap-3">
        <div className="flex gap-2">
          <Input
            type="text"
            placeholder="Search manuals... (e.g., 'battery test procedure')"
            value={query}
            onChange={(e) => handleChange(e.target.value)}
            className="flex-1"
            disabled={isLoading}
          />
          <Button type="submit" disabled={isLoading || query.trim().length < 3}>
            <Search className="h-4 w-4 mr-2" />
            Search
          </Button>
        </div>
        <div className="flex items-center gap-3">
          <Select value={chunkType} onValueChange={handleChunkTypeChange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All types</SelectItem>
              <SelectItem value="text">Text</SelectItem>
              <SelectItem value="table">Table</SelectItem>
              <SelectItem value="figure">Figure</SelectItem>
              <SelectItem value="title">Title</SelectItem>
              <SelectItem value="marginalia">Marginalia</SelectItem>
            </SelectContent>
          </Select>
          {isAutoRefresh && query.trim().length >= 3 && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <RefreshCw className="h-3 w-3 animate-spin" />
              <span>Auto-refresh enabled</span>
            </div>
          )}
        </div>
      </div>
    </form>
  );
}

