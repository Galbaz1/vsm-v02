# Manual Search Frontend

Modern Next.js frontend for semantic search over technical asset manuals with visual grounding.

## Tech Stack

- **Next.js 14** (App Router) with TypeScript
- **Tailwind CSS** for styling
- **shadcn/ui** for UI components
- **React Query** (@tanstack/react-query) for data fetching
- **Axios** for API calls

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- FastAPI backend running on `http://localhost:8001` (see main README)

### Installation

```bash
cd frontend
npm install
```

### Environment Setup

Copy `.env.example` to `.env.local` and adjust if needed:

```bash
cp .env.example .env.local
```

The default `NEXT_PUBLIC_API_BASE_URL` points to `http://localhost:8001` where the FastAPI backend runs.

### Development

Start the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout with providers
│   ├── page.tsx           # Main search page
│   └── providers.tsx       # React Query provider
├── components/            # React components
│   ├── ui/               # shadcn/ui components
│   ├── SearchBar.tsx     # Search input component
│   ├── ResultCard.tsx    # Search result card
│   └── PreviewPanel.tsx  # Preview dialog with bbox overlay
├── lib/                  # Utilities and hooks
│   ├── api.ts            # API client
│   ├── types.ts          # TypeScript types
│   ├── hooks/            # React hooks
│   └── utils/            # Utility functions
└── public/               # Static assets
```

## Features

- **Semantic Search**: Query manuals using natural language with debounced auto-refresh
- **Chunk Type Filtering**: Filter results by content type (text, table, figure, etc.)
- **Visual Grounding**: See exact PDF regions highlighted with bounding boxes
- **Multi-Highlight Preview**: View all hits on a page with color-coded bounding boxes
- **Smart Deduplication**: Group duplicate content by page or content hash
- **Section Titles**: Display section headings for better context
- **Preview Navigation**: Navigate between multiple hits on the same page
- **PDF Linking**: Direct links to specific pages in PDFs
- **Keyword Highlighting**: Query terms highlighted in results
- **Responsive Design**: Works on desktop and mobile

## Features Explained

### Debounced Search

The search automatically triggers 400ms after you stop typing (when query length >= 3). An "Auto-refresh enabled" indicator shows when this is active. You can also manually submit using the Search button.

### Chunk Type Filter

Use the dropdown to filter results by content type:
- **All types** - Show everything
- **Text** - Paragraphs and prose
- **Table** - Tabular data
- **Figure** - Images and diagrams
- **Title** - Headings and titles
- **Marginalia** - Side notes

### Grouping & Deduplication

Enable "Group by page" checkbox to:
- Collapse duplicate content hashes per page
- Show only unique snippets initially
- Expand to see duplicates when needed

Without grouping, duplicates are collapsed by content hash across all pages.

### Preview Modal

When multiple hits exist on the same page:
- All bounding boxes are highlighted (different colors)
- Active hit is highlighted in blue
- Navigate with Previous/Next buttons
- View all snippets in a scrollable list
- Click any snippet to jump to it

## API Integration

The frontend consumes the FastAPI backend at `/search` endpoint:

- **GET** `/search?query=<query>&limit=<limit>&chunk_type=<type>&group_by_page=<bool>`
- Returns: `{ query: string, hits: SearchHit[], page_hits?: Record<string, PageHit> }`

Each `SearchHit` includes:
- `content`: Cleaned text snippet (HTML/ADE markup removed)
- `page_number`: Page index (0-based)
- `bbox`: Bounding box coordinates (0-1 normalized)
- `chunk_type`: Content type (text, table, figure, etc.)
- `section_title`: Section heading if available
- `content_hash`: SHA256 hash for deduplication
- `pdf_page_url`: Link to PDF with page anchor
- `page_image_url`: Preview image URL

The `page_hits` structure aggregates all hits per page for the preview modal.

## Static File Proxying

The Next.js config includes rewrites to proxy `/static/*` requests to the FastAPI backend, so preview images and PDFs are served seamlessly.
