# llms.txt Generator

Generate `llms.txt` documentation from any GitHub repository using DSPy.

## Usage

```bash
# Generate for DSPy (default)
python fetch_repo_llms_txt.py

# Any GitHub repo
python fetch_repo_llms_txt.py --repo langchain-ai/langchain
python fetch_repo_llms_txt.py --repo openai/openai-python --output openai_llms.txt

# With verbose output
python fetch_repo_llms_txt.py --repo stanfordnlp/dspy --verbose
```

## Requirements

- Python 3.8+
- DSPy (`pip install dspy-ai`)
- Ollama running locally with `gpt-oss:120b`
- Optional: `GITHUB_ACCESS_TOKEN` env var for higher API rate limits

## Output

Generated `llms.txt` includes:
- Project purpose
- Key concepts
- Architecture overview
- Important directories
- Entry points
- Development info
- Usage examples
