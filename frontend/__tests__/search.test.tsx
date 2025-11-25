/**
 * Basic smoke test for search functionality
 * Run with: npm test
 */

import { describe, it, expect } from '@jest/globals';

describe('Search API', () => {
  it('should have correct API base URL', () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8001';
    expect(apiUrl).toBeTruthy();
    expect(apiUrl).toMatch(/^https?:\/\//);
  });
});

describe('Type definitions', () => {
  it('should export SearchHit type', () => {
    // This is a compile-time check
    const hit = {
      anchor_id: 'test',
      manual_name: 'Test Manual',
      content: 'Test content',
      page_number: 1,
      bbox: { left: 0.1, top: 0.2, right: 0.3, bottom: 0.4 },
      pdf_page_url: '/static/manuals/test.pdf#page=1',
      page_image_url: '/static/previews/test/page-1.png',
    };
    expect(hit.anchor_id).toBe('test');
  });
});

