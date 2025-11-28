"""
Benchmark Service for Phase 8.

Runs benchmark queries through the agent, evaluates with TechnicalJudge,
and computes aggregate metrics.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.services.agent import get_agent
from api.services.judge import evaluate_answer, JudgeResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkItem:
    """A single benchmark item from the dataset."""
    id: str
    query: str
    expected_answer: str
    expected_sources: Optional[List[Dict[str, Any]]] = None


@dataclass
class QueryResult:
    """Result of running a single benchmark query."""
    id: str
    query: str
    expected_answer: str
    model_answer: str
    sources: List[Dict[str, Any]]
    judge_score: float
    judge_rationale: str
    cited_pages: List[str]
    latency_ms: float
    iterations: int
    tools_used: List[str]
    trace_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BenchmarkMetrics:
    """Aggregate metrics for a benchmark run."""
    avg_score: float
    hit_at_1: float  # % of queries with correct page in top-1
    hit_at_3: float  # % of queries with correct page in top-3
    mrr: float  # Mean Reciprocal Rank
    latency_p50_ms: float
    latency_p95_ms: float
    total_queries: int
    completed: int
    failed: int
    avg_iterations: float
    tool_distribution: Dict[str, int]


@dataclass
class BenchmarkRunResult:
    """Complete result of a benchmark run."""
    mode: str
    dataset: str
    timestamp: str
    total: int
    completed: int
    metrics: BenchmarkMetrics
    records: List[QueryResult]
    summary: Dict[str, Any] = field(default_factory=dict)


def load_benchmark_dataset(path: str) -> List[BenchmarkItem]:
    """Load the benchmark dataset into typed items."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[BenchmarkItem] = []
    for idx, row in enumerate(data):
        item_id = str(row.get("id") or idx + 1)
        query = row.get("query") or row.get("question") or ""
        expected_answer = (
            row.get("expected_answer")
            or row.get("answer")
            or row.get("reference", {}).get("answer", "")
        )
        evidence = row.get("evidence", {})
        expected_sources = evidence.get("locations") if isinstance(evidence, dict) else None

        items.append(
            BenchmarkItem(
                id=item_id,
                query=query,
                expected_answer=expected_answer,
                expected_sources=expected_sources,
            )
        )

    logger.info(f"Loaded {len(items)} benchmark items from {dataset_path}")
    return items


async def run_single_query(
    item: BenchmarkItem,
    enable_tracing: bool = False,
) -> QueryResult:
    """
    Run a single benchmark query through the agent.
    
    Returns:
        QueryResult with model answer, sources, timing, etc.
    """
    agent = get_agent()
    
    start_time = time.perf_counter()
    model_answer = ""
    sources: List[Dict[str, Any]] = []
    tools_used: List[str] = []
    iterations = 0
    trace_id = None
    error = None
    
    try:
        async for event in agent.run(item.query):
            if isinstance(event, dict):
                event_type = event.get("type")
                
                if event_type == "decision":
                    payload = event.get("payload", {})
                    tool = payload.get("tool")  # Decision.to_frontend uses "tool" not "tool_name"
                    if tool:
                        tools_used.append(tool)
                    iterations += 1
                    
                elif event_type == "response":
                    # TextResponseTool yields Response with text
                    payload = event.get("payload", {})
                    response_text = payload.get("text", "")
                    # Only update if we got a non-trivial response
                    # (avoid overwriting good answer with empty or short default messages)
                    if response_text and len(response_text) > 50:
                        model_answer = response_text
                    elif not model_answer:  # Only set if we don't have one yet
                        model_answer = response_text
                    # Also capture sources from response
                    resp_sources = payload.get("sources", [])
                    for src in resp_sources:
                        page = src.get("page")
                        manual = src.get("manual")
                        if page:
                            sources.append({
                                "page": str(page),
                                "manual": manual or "unknown",
                            })
                
                elif event_type == "result":
                    payload = event.get("payload", {})
                    name = payload.get("name", "")
                    
                    # Capture sources from search results
                    if name in ("fast_vector_search", "hybrid_search", "colqwen_search", "search_results"):
                        objects = payload.get("objects", [])
                        for obj in objects:
                            if isinstance(obj, dict):
                                page = obj.get("page_number") or obj.get("page")
                                manual = obj.get("manual_name") or obj.get("asset_manual")
                                if page:
                                    sources.append({
                                        "page": str(page),
                                        "manual": manual or "unknown",
                                    })
                
                elif event_type == "trace":
                    trace_id = event.get("payload", {}).get("trace_id")
                    
    except Exception as e:
        error = str(e)
        logger.error(f"Query failed: {item.id} - {e}")
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Evaluate with judge
    judge_result = await evaluate_answer(
        query=item.query,
        answer=model_answer,
        expected_answer=item.expected_answer,
        sources=sources,
    )
    
    return QueryResult(
        id=item.id,
        query=item.query,
        expected_answer=item.expected_answer,
        model_answer=model_answer,
        sources=sources,
        judge_score=judge_result.score,
        judge_rationale=judge_result.rationale,
        cited_pages=judge_result.cited_pages,
        latency_ms=latency_ms,
        iterations=iterations,
        tools_used=tools_used,
        trace_id=trace_id,
        error=error,
    )


def calculate_hit_at_k(
    result: QueryResult,
    expected_sources: Optional[List[Dict[str, Any]]],
    k: int,
) -> bool:
    """Check if any expected page appears in top-k sources."""
    if not expected_sources:
        return False
    
    expected_pages = {str(s.get("page")) for s in expected_sources}
    top_k_pages = {str(s.get("page")) for s in result.sources[:k]}
    
    return bool(expected_pages & top_k_pages)


def calculate_mrr(
    result: QueryResult,
    expected_sources: Optional[List[Dict[str, Any]]],
) -> float:
    """Calculate reciprocal rank for a result."""
    if not expected_sources:
        return 0.0
    
    expected_pages = {str(s.get("page")) for s in expected_sources}
    
    for rank, source in enumerate(result.sources, 1):
        if str(source.get("page")) in expected_pages:
            return 1.0 / rank
    
    return 0.0


def compute_metrics(
    items: List[BenchmarkItem],
    results: List[QueryResult],
) -> BenchmarkMetrics:
    """Compute aggregate metrics from benchmark results."""
    if not results:
        return BenchmarkMetrics(
            avg_score=0.0,
            hit_at_1=0.0,
            hit_at_3=0.0,
            mrr=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            total_queries=len(items),
            completed=0,
            failed=0,
            avg_iterations=0.0,
            tool_distribution={},
        )
    
    # Build item lookup
    item_map = {item.id: item for item in items}
    
    # Filter successful results
    successful = [r for r in results if r.error is None]
    failed = len(results) - len(successful)
    
    # Scores
    scores = [r.judge_score for r in successful]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # Hit@k
    hit_1_count = sum(
        1 for r in successful
        if calculate_hit_at_k(r, item_map.get(r.id, BenchmarkItem("", "", "")).expected_sources, 1)
    )
    hit_3_count = sum(
        1 for r in successful
        if calculate_hit_at_k(r, item_map.get(r.id, BenchmarkItem("", "", "")).expected_sources, 3)
    )
    
    hit_at_1 = hit_1_count / len(successful) if successful else 0.0
    hit_at_3 = hit_3_count / len(successful) if successful else 0.0
    
    # MRR
    mrr_values = [
        calculate_mrr(r, item_map.get(r.id, BenchmarkItem("", "", "")).expected_sources)
        for r in successful
    ]
    mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0
    
    # Latency percentiles
    latencies = sorted(r.latency_ms for r in successful)
    if latencies:
        p50_idx = int(len(latencies) * 0.5)
        p95_idx = int(len(latencies) * 0.95)
        latency_p50 = latencies[min(p50_idx, len(latencies) - 1)]
        latency_p95 = latencies[min(p95_idx, len(latencies) - 1)]
    else:
        latency_p50 = latency_p95 = 0.0
    
    # Iterations
    iterations = [r.iterations for r in successful]
    avg_iterations = sum(iterations) / len(iterations) if iterations else 0.0
    
    # Tool distribution
    tool_counts: Dict[str, int] = {}
    for r in successful:
        for tool in r.tools_used:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    return BenchmarkMetrics(
        avg_score=avg_score,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        mrr=mrr,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        total_queries=len(items),
        completed=len(successful),
        failed=failed,
        avg_iterations=avg_iterations,
        tool_distribution=tool_counts,
    )


async def run_benchmark(
    dataset_path: str,
    mode: str,
    enable_tracing: bool = False,
    max_queries: Optional[int] = None,
) -> BenchmarkRunResult:
    """
    Run the full benchmark evaluation.
    
    Args:
        dataset_path: Path to benchmark JSON file
        mode: VSM_MODE value (for reporting)
        enable_tracing: Whether to save query traces
        max_queries: Optional limit on queries to run
        
    Returns:
        BenchmarkRunResult with all metrics and records
    """
    items = load_benchmark_dataset(dataset_path)

    if max_queries:
        items = items[:max_queries]
    
    logger.info(f"Running benchmark: {len(items)} queries in {mode} mode")
    
    results: List[QueryResult] = []
    
    for i, item in enumerate(items):
        logger.info(f"[{i+1}/{len(items)}] Running: {item.query[:50]}...")
        result = await run_single_query(item, enable_tracing)
        results.append(result)
        logger.info(f"  Score: {result.judge_score:.2f}, Latency: {result.latency_ms:.0f}ms")
    
    metrics = compute_metrics(items, results)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    return BenchmarkRunResult(
        mode=mode,
        dataset=str(dataset_path),
        timestamp=timestamp,
        total=len(items),
        completed=metrics.completed,
        metrics=metrics,
        records=results,
        summary={
            "avg_score": f"{metrics.avg_score:.2f}",
            "hit_at_3": f"{metrics.hit_at_3:.1%}",
            "mrr": f"{metrics.mrr:.3f}",
            "latency_p50": f"{metrics.latency_p50_ms:.0f}ms",
            "latency_p95": f"{metrics.latency_p95_ms:.0f}ms",
        },
    )


def save_benchmark_report(result: BenchmarkRunResult) -> Path:
    """Save benchmark report to logs/benchmarks/."""
    logs_dir = Path("logs/benchmarks")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{result.timestamp}-{result.mode}.json"
    filepath = logs_dir / filename
    
    # Convert to serializable dict
    data = {
        "mode": result.mode,
        "dataset": result.dataset,
        "timestamp": result.timestamp,
        "total": result.total,
        "completed": result.completed,
        "summary": result.summary,
        "metrics": asdict(result.metrics),
        "records": [asdict(r) for r in result.records],
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved benchmark report to {filepath}")
    return filepath


# Legacy stub for backwards compatibility
async def run_benchmark_stub(
    dataset_path: str,
    mode: str,
    enable_tracing: bool = False,
) -> BenchmarkRunResult:
    """Legacy stub - now runs the real benchmark."""
    return await run_benchmark(dataset_path, mode, enable_tracing, max_queries=3)


def serialize_run_result(result: BenchmarkRunResult) -> Dict[str, Any]:
    """Serialize a run result to a JSON-friendly dict."""
    return {
        "mode": result.mode,
        "dataset": result.dataset,
        "timestamp": result.timestamp,
        "total": result.total,
        "completed": result.completed,
        "summary": result.summary,
        "metrics": asdict(result.metrics),
        "records": [asdict(r) for r in result.records],
    }
