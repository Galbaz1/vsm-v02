# Utility Scripts

This directory contains utility scripts for parsing, ingesting, and searching manual data.

**Note:** All scripts should be run from the project root directory to ensure relative paths (like `data/` and `static/`) work correctly.

## Usage

### Parsing PDFs

Parse a PDF using LandingAI ADE.

```bash
python scripts/parse_with_landingai.py data/uk_firmware.pdf output_landingai.json
```

### Generating Previews

Generate PNG preview images for each page of a PDF.

```bash
# Uses default paths
python scripts/generate_previews.py

# Or specify paths
python scripts/generate_previews.py data/uk_firmware.pdf static/previews/uk_firmware
```

### Ingesting Data

Ingest the parsed JSON output into Weaviate.

```bash
python scripts/weaviate_ingest_manual.py output_landingai.json "UK Firmware Manual"
```

### Manual Search (CLI)

Perform a search query from the command line for testing.

```bash
python scripts/weaviate_search_manual.py "how do I reset the compressor alarm?"
```
