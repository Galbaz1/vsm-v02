#!/usr/bin/env python
"""
RAG Benchmark Evaluation Script

Evaluates retrieval quality using ground-truth benchmark questions.
Supports both local client-side evaluation and server-side via /benchmark/evaluate.

Usage:
    # Full benchmark via API (uses TechnicalJudge, saves reports)
    python scripts/run_benchmark.py --use-endpoint
    
    # Quick test (3 queries only)
    python scripts/run_benchmark.py --use-endpoint --max-queries 3
    
    # Override mode
    python scripts/run_benchmark.py --use-endpoint --mode cloud
    
    # Legacy client-side comparison (Regular RAG vs ColQwen)
    python scripts/run_benchmark.py --output results.json --tolerance 1
"""

import json
import sys
import argparse
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Document name mapping from benchmark to Weaviate collection
DOC_NAME_MAP = {
    # Old format (benchmark.json)
    "91001002 Techman": "Technical Manual",
    "92002703 Bedienung UK": "UK Firmware Manual",
    # New format (benchmarksv03.json)
    "techman.pdf": "Technical Manual",
    "uk_firmware.pdf": "UK Firmware Manual",
}

BASE_URL = "http://localhost:8001"


class BenchmarkResult:
    """Result for a single question."""
    def __init__(self, question_id: int, question: str, expected_page: int, expected_manual: str):
        self.question_id = question_id
        self.question = question
        self.expected_page = expected_page
        self.expected_manual = expected_manual
        self.regular_rag_rank: Optional[int] = None
        self.regular_rag_manual: Optional[str] = None
        self.regular_rag_results: List[Dict] = []
        self.colqwen_rank: Optional[int] = None
        self.colqwen_manual: Optional[str] = None
        self.colqwen_results: List[Dict] = []


def load_benchmark(benchmark_path: str) -> List[Dict]:
    """Load benchmark questions from JSON."""
    with open(benchmark_path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_regular_rag(query: str, limit: int = 5) -> List[Dict]:
    """Query the regular RAG endpoint (/search)."""
    try:
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"query": query, "limit": limit},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", [])
    except Exception as e:
        print(f"Error querying regular RAG: {e}")
        return []


def query_colqwen_rag(query: str, limit: int = 5) -> List[Dict]:
    """Query the ColQwen RAG endpoint (/agentic_search)."""
    try:
        resp = requests.get(
            f"{BASE_URL}/agentic_search",
            params={"query": query, "top_k": limit},
            timeout=30,
            stream=True
        )
        resp.raise_for_status()
        
        # Parse NDJSON stream to find colqwen_results
        colqwen_results = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "colqwen_results":
                    colqwen_results = obj.get("data", [])
                    break
            except json.JSONDecodeError:
                continue
        
        return colqwen_results
    except Exception as e:
        print(f"Error querying ColQwen RAG: {e}")
        return []


def find_page_in_results(
    results: List[Dict],
    expected_page: int,
    expected_manual: str,
    tolerance: int = 1
) -> Tuple[Optional[int], Optional[str]]:
    """
    Find the rank (1-indexed) of the first result matching expected page.
    Returns (rank, manual_name) or (None, None) if not found.
    """
    for rank, result in enumerate(results, start=1):
        page = result.get("page_number")
        manual = result.get("manual_name") or result.get("asset_manual")
        
        if page is None:
            continue
        
        # Check if page matches (with tolerance)
        if abs(page - expected_page) <= tolerance:
            # Also check manual matches
            if manual == expected_manual:
                return rank, manual
    
    return None, None


def calculate_mrr(results: List[BenchmarkResult], pipeline: str) -> float:
    """Calculate Mean Reciprocal Rank for a pipeline."""
    ranks = []
    for r in results:
        rank = r.regular_rag_rank if pipeline == "regular" else r.colqwen_rank
        if rank is not None:
            ranks.append(1.0 / rank)
        else:
            ranks.append(0.0)
    return sum(ranks) / len(ranks) if ranks else 0.0


def calculate_hit_at_k(results: List[BenchmarkResult], k: int, pipeline: str) -> float:
    """Calculate Hit@k percentage."""
    hits = 0
    for r in results:
        rank = r.regular_rag_rank if pipeline == "regular" else r.colqwen_rank
        if rank is not None and rank <= k:
            hits += 1
    return (hits / len(results)) * 100.0 if results else 0.0


def calculate_manual_accuracy(results: List[BenchmarkResult], pipeline: str) -> float:
    """Calculate percentage of queries where correct manual was retrieved."""
    correct = 0
    for r in results:
        manual = r.regular_rag_manual if pipeline == "regular" else r.colqwen_manual
        if manual == r.expected_manual:
            correct += 1
    return (correct / len(results)) * 100.0 if results else 0.0


def run_legacy_benchmark(benchmark_path: str, tolerance: int = 1, output_path: Optional[str] = None):
    """Run legacy client-side benchmark evaluation (Regular RAG vs ColQwen)."""
    print("=" * 70)
    print("RAG Benchmark Evaluation (Legacy Mode)")
    print("=" * 70)
    print()
    
    # Load benchmark
    benchmark = load_benchmark(benchmark_path)
    print(f"Loaded {len(benchmark)} benchmark questions")
    print()
    
    # Initialize results
    results: List[BenchmarkResult] = []
    
    # Process each question
    for idx, item in enumerate(benchmark):
        # Support both old (benchmark.json) and new (benchmarksv03.json) formats
        q_id = item.get("id", idx + 1)
        question = item.get("question") or item.get("query")
        
        # Handle both reference formats
        ref = item.get("reference", {})
        evidence = item.get("evidence", {})
        
        if evidence:
            # New format (benchmarksv03.json)
            expected_doc = evidence.get("document")
            locations = evidence.get("locations", [{}])
            expected_page = int(locations[0].get("page", 0)) if locations else 0
        else:
            # Old format (benchmark.json)
            expected_doc = ref.get("document")
            expected_page = ref.get("page")
        
        # Map document name
        expected_manual = DOC_NAME_MAP.get(expected_doc, expected_doc)
        
        result = BenchmarkResult(q_id, question, expected_page, expected_manual)
        
        print(f"Q{q_id}: {question[:60]}...")
        print(f"  Expected: {expected_manual}, Page {expected_page}")
        
        # Query Regular RAG
        regular_results = query_regular_rag(question, limit=5)
        result.regular_rag_results = regular_results
        rank, manual = find_page_in_results(regular_results, expected_page, expected_manual, tolerance)
        result.regular_rag_rank = rank
        result.regular_rag_manual = manual
        if rank:
            print(f"  Regular RAG: Found at rank {rank} ({manual})")
        else:
            print(f"  Regular RAG: Not found")
        
        # Query ColQwen RAG
        colqwen_results = query_colqwen_rag(question, limit=5)
        result.colqwen_results = colqwen_results
        rank, manual = find_page_in_results(colqwen_results, expected_page, expected_manual, tolerance)
        result.colqwen_rank = rank
        result.colqwen_manual = manual
        if rank:
            print(f"  ColQwen: Found at rank {rank} ({manual})")
        else:
            print(f"  ColQwen: Not found")
        
        print()
        results.append(result)
    
    # Calculate and print metrics
    print_legacy_metrics(results, output_path)


def print_legacy_metrics(results: List[BenchmarkResult], output_path: Optional[str] = None):
    """Print metrics for legacy benchmark."""
    print("=" * 70)
    print("Regular RAG Results")
    print("=" * 70)
    print()
    
    hit1 = calculate_hit_at_k(results, 1, "regular")
    hit3 = calculate_hit_at_k(results, 3, "regular")
    hit5 = calculate_hit_at_k(results, 5, "regular")
    mrr = calculate_mrr(results, "regular")
    manual_acc = calculate_manual_accuracy(results, "regular")
    
    print(f"Hit@1: {hit1:.1f}% | Hit@3: {hit3:.1f}% | Hit@5: {hit5:.1f}% | MRR: {mrr:.3f}")
    print()
    
    print("=" * 70)
    print("ColQwen Pipeline Results")
    print("=" * 70)
    print()
    
    hit1_col = calculate_hit_at_k(results, 1, "colqwen")
    hit3_col = calculate_hit_at_k(results, 3, "colqwen")
    hit5_col = calculate_hit_at_k(results, 5, "colqwen")
    mrr_col = calculate_mrr(results, "colqwen")
    manual_acc_col = calculate_manual_accuracy(results, "colqwen")
    
    print(f"Hit@1: {hit1_col:.1f}% | Hit@3: {hit3_col:.1f}% | Hit@5: {hit5_col:.1f}% | MRR: {mrr_col:.3f}")
    print()
    
    print("=" * 70)
    print("Comparison")
    print("=" * 70)
    print()
    print(f"{'Metric':<20} {'Regular RAG':<15} {'ColQwen':<15} {'Delta':<10}")
    print("-" * 60)
    print(f"{'Hit@1':<20} {hit1:>6.1f}%{'':<7} {hit1_col:>6.1f}%{'':<7} {hit1_col-hit1:>+6.1f}%")
    print(f"{'Hit@3':<20} {hit3:>6.1f}%{'':<7} {hit3_col:>6.1f}%{'':<7} {hit3_col-hit3:>+6.1f}%")
    print(f"{'MRR':<20} {mrr:>6.3f}{'':<7} {mrr_col:>6.3f}{'':<7} {mrr_col-mrr:>+6.3f}")
    print()
    
    # Save JSON output if requested
    if output_path:
        output_data = {
            "regular_rag": {"hit_at_1": hit1, "hit_at_3": hit3, "mrr": mrr},
            "colqwen": {"hit_at_1": hit1_col, "hit_at_3": hit3_col, "mrr": mrr_col},
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_path}")


def run_via_endpoint(
    benchmark_path: str,
    mode: Optional[str] = None,
    max_queries: Optional[int] = None,
    enable_tracing: bool = False,
):
    """
    Run benchmark via the /benchmark/evaluate endpoint.
    Uses TechnicalJudge for answer quality evaluation.
    """
    print("=" * 70)
    print("Benchmark Evaluation via API")
    print("=" * 70)
    print()
    
    payload = {
        "dataset_path": benchmark_path,
        "enable_tracing": enable_tracing,
        "save_report": True,
    }
    if mode:
        payload["mode"] = mode
    if max_queries:
        payload["max_queries"] = max_queries
    
    print(f"Sending request to {BASE_URL}/benchmark/evaluate...")
    print(f"  Dataset: {benchmark_path}")
    if max_queries:
        print(f"  Max queries: {max_queries}")
    if mode:
        print(f"  Mode override: {mode}")
    print()
    
    try:
        # Long timeout for full benchmark
        timeout = 60 * 10 if not max_queries else 60 * 2
        resp = requests.post(
            f"{BASE_URL}/benchmark/evaluate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Print summary
        print("=" * 70)
        print("Results Summary")
        print("=" * 70)
        print()
        
        metrics = data.get("metrics", {})
        summary = data.get("summary", {})
        
        print(f"Mode: {data.get('mode')}")
        print(f"Total queries: {data.get('total')}")
        print(f"Completed: {data.get('completed')}")
        print()
        
        print("Metrics:")
        print(f"  Average Judge Score: {metrics.get('avg_score', 0):.2f}")
        print(f"  Hit@1: {metrics.get('hit_at_1', 0):.1%}")
        print(f"  Hit@3: {metrics.get('hit_at_3', 0):.1%}")
        print(f"  MRR: {metrics.get('mrr', 0):.3f}")
        print(f"  Latency P50: {metrics.get('latency_p50_ms', 0):.0f}ms")
        print(f"  Latency P95: {metrics.get('latency_p95_ms', 0):.0f}ms")
        print(f"  Avg Iterations: {metrics.get('avg_iterations', 0):.1f}")
        print()
        
        # Tool distribution
        tool_dist = metrics.get("tool_distribution", {})
        if tool_dist:
            print("Tool Distribution:")
            for tool, count in sorted(tool_dist.items(), key=lambda x: -x[1]):
                print(f"  {tool}: {count}")
            print()
        
        # Report path
        if data.get("report_path"):
            print(f"Full report saved: {data['report_path']}")
        
        # Print per-query results (first 5)
        records = data.get("records", [])
        if records:
            print()
            print("=" * 70)
            print(f"Per-Query Results (showing {min(5, len(records))} of {len(records)})")
            print("=" * 70)
            for rec in records[:5]:
                score = rec.get("judge_score", 0)
                icon = "✓" if score >= 0.7 else "~" if score >= 0.4 else "✗"
                print(f"\n{icon} Q{rec.get('id')}: {rec.get('query', '')[:60]}...")
                print(f"  Score: {score:.2f} | Latency: {rec.get('latency_ms', 0):.0f}ms | Iterations: {rec.get('iterations', 0)}")
                if rec.get("error"):
                    print(f"  Error: {rec['error']}")
                if rec.get("judge_rationale"):
                    print(f"  Rationale: {rec['judge_rationale'][:100]}...")
        
    except requests.exceptions.Timeout:
        print("Request timed out. For large benchmarks, consider using --max-queries to test first.")
    except Exception as exc:
        print(f"Benchmark endpoint call failed: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark via API (recommended)
  python scripts/run_benchmark.py --use-endpoint
  
  # Quick test with 3 queries
  python scripts/run_benchmark.py --use-endpoint --max-queries 3
  
  # Run in specific mode
  python scripts/run_benchmark.py --use-endpoint --mode cloud
  
  # Legacy client-side comparison
  python scripts/run_benchmark.py --output results.json
"""
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmarks/benchmarksv03.json",
        help="Path to benchmark JSON file"
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output file (legacy mode only)"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=1,
        help="Page number tolerance for legacy mode (default: 1)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8001",
        help="API base URL (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--use-endpoint",
        action="store_true",
        help="Use /benchmark/evaluate endpoint (Phase 8)"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "cloud"],
        help="Override VSM_MODE for benchmark run"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Limit number of queries (useful for testing)"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable query tracing"
    )
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    if args.use_endpoint:
        run_via_endpoint(
            str(benchmark_path),
            mode=args.mode,
            max_queries=args.max_queries,
            enable_tracing=args.trace,
        )
    else:
        run_legacy_benchmark(str(benchmark_path), args.tolerance, args.output)


if __name__ == "__main__":
    main()
