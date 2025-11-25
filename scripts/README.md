# Utility Scripts

This directory contains utility scripts for parsing, ingesting, and searching manual data.

**Note:** All scripts should be run from the project root directory to ensure relative paths (like `data/` and `static/`) work correctly.

📖 **See also:** [System Architecture - Scripts Reference](../docs/ARCHITECTURE.md#scripts-reference)

## Usage

### Parsing PDFs

Parse a PDF using LandingAI ADE.

```bash
python scripts/parse_with_landingai.py data/uk_firmware.pdf output_landingai.json
```

### Generating Previews

Generate PNG preview images for each page of a PDF.

```bash
# Specify paths (recommended)
python scripts/generate_previews.py data/uk_firmware.pdf static/previews/uk_firmware

# For techman.pdf
python scripts/generate_previews.py data/techman.pdf static/previews/techman
```

### Ingesting Data

#### Regular RAG (Text-based)

Ingest the parsed JSON output into Weaviate AssetManual collection.

```bash
python scripts/weaviate_ingest_manual.py output_landingai.json "UK Firmware Manual"
```

#### ColQwen RAG (Multimodal)

Ingest PDF pages as multi-vector embeddings into Weaviate PDFDocuments collection.

```bash
# Requires preview PNGs to exist first!
python scripts/colqwen_ingest.py "UK Firmware Manual"
python scripts/colqwen_ingest.py "Technical Manual"
```

**Note:** ColQwen ingestion downloads ~8GB model on first run.

### Search Testing

#### CLI Search (Regular RAG)

Perform a search query from the command line for testing.

```bash
python scripts/weaviate_search_manual.py "how do I reset the compressor alarm?"
```

#### Verify Ingestion

Check that both manuals are properly ingested.

```bash
python test_both_manuals.py
```

