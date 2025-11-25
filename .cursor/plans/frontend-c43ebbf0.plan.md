<!-- c43ebbf0-c95b-4efd-b389-04ac85ceeed6 6d7f51cf-bd66-43d5-a8b1-22d62f736362 -->
# Search Refinement Plan

## 1. Clean & enrich ingestion data

- **ingest-cleanup:** Extend `weaviate_ingest_manual.py` to strip ADE markup (tables, anchor tags, flowchart blocks), decode HTML entities, and capture nearby section headings (`section_title`). Re-ingest the manual so each chunk has clean prose + metadata.
- **dedupe-tags:** While ingesting, compute a normalized hash (e.g., SHA256 on lowercase text) stored in a new `content_hash` property to support deduplication and grouping later.

## 2. API-side improvements

- **api-schema:** Update `api/main.py` to include `section_title`, `chunk_type`, `content_hash`, and expose a `chunk_type` filter + optional `group_by_page` flag that collapses duplicate hashes per `(manual,page)` pair.
- **preview-context:** Extend `/search` responses with a `page_hits` structure (aggregated bboxes per page) so the UI can highlight multiple snippets in the same preview.

## 3. Frontend UX upgrades

- **search-flow:** Reintroduce debounced queries in `SearchBar`/`useManualSearch`, add a chunk-type dropdown, and surface an “Auto-refresh” indicator so queries run after typing stops.
- **result-card:** Show section titles beneath manual names, badge chunk types, hide preview buttons when `page_image_url` is missing, and provide a secondary “Open page” link pointing to the PDF page.
- **preview-modal:** Render multiple hit highlights on the same image, list the related snippets beneath the preview, and add next/previous controls when a page has several hits.

## 4. Dedup & grouping logic

- **ui-grouping:** In `frontend/app/page.tsx`, group hits by `(manual,page)` (or `content_hash`) using the API metadata, display only unique snippets initially, and let the user expand to see duplicates when desired.
- **tests-docs:** Update Playwright smoke tests plus `frontend/README.md` to cover the new filters, debounced flow, and grouping behavior.

### To-dos

- [ ] Verify LandingAI script updates with November 2025 settings
- [ ] Update ingest script to use optimal JSON format and Weaviate best practices
- [ ] Deep research Weaviate docs for schema optimization
- [ ] Store ADE bbox metadata as JSON for downstream APIs
- [ ] Build FastAPI search API returning highlighted PDF context