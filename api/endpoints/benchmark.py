"""
Benchmark API Endpoint.

POST /benchmark/evaluate - Run benchmark and return results.
POST /benchmark/evaluate-single - Run benchmark for a single query.
GET /benchmark/questions - Get benchmark questions for UI suggestions.
GET /benchmark/reports - List saved benchmark reports.
"""

from pathlib import Path
from typing import Optional, List
import json

from fastapi import APIRouter, Body, Query, HTTPException

from api.services.benchmark import (
    run_benchmark,
    run_single_benchmark,
    save_benchmark_report,
    serialize_run_result,
)
from api.core.config import get_settings

router = APIRouter()


@router.get("/benchmark/questions")
async def get_benchmark_questions(
    limit: int = Query(5, ge=1, le=50, description="Max questions to return"),
    dataset_path: str = Query("data/benchmarks/benchmarksv03.json"),
) -> List[dict]:
    """
    Get benchmark questions for UI suggestions.
    
    Returns questions with id, category, query, and expected answer.
    """
    filepath = Path(dataset_path)
    if not filepath.exists():
        raise HTTPException(404, f"Dataset not found: {dataset_path}")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    questions = []
    for i, item in enumerate(data[:limit]):
        questions.append({
            "id": str(i + 1),
            "category": item.get("category", ""),
            "query": item.get("query", ""),
            "expected_answer": item.get("answer", ""),
        })
    
    return questions


@router.post("/benchmark/evaluate-single")
async def evaluate_single_query(
    query: str = Body(..., description="The query to evaluate"),
    expected_answer: str = Body(..., description="Expected answer for scoring"),
    query_id: Optional[str] = Body(None, description="Optional query ID"),
    enable_tracing: bool = Body(True),
    mode: Optional[str] = Body(None),
):
    """
    Run benchmark evaluation for a single query.
    
    Executes the query through the agent, evaluates with TechnicalJudge,
    and returns the result immediately.
    
    Args:
        query: The search query to evaluate
        expected_answer: Ground truth answer for scoring
        query_id: Optional identifier for the query
        enable_tracing: Save query trace for debugging
        mode: Override VSM_MODE (default: use current setting)
        
    Returns:
        Single benchmark record with score, answer, sources, etc.
    """
    settings = get_settings()
    run_mode = mode or settings.vsm_mode
    
    result = await run_single_benchmark(
        query=query,
        expected_answer=expected_answer,
        query_id=query_id,
        mode=run_mode,
        enable_tracing=enable_tracing,
    )
    
    return result


@router.post("/benchmark/evaluate")
async def evaluate_benchmark(
    dataset_path: str = Body("data/benchmarks/benchmarksv03.json"),
    enable_tracing: bool = Body(False),
    mode: Optional[str] = Body(None),
    max_queries: Optional[int] = Body(None, description="Limit queries for quick tests"),
    save_report: bool = Body(True, description="Save report to logs/benchmarks/"),
):
    """
    Run benchmark evaluation.

    Executes queries through the agent, evaluates with TechnicalJudge,
    and returns aggregate metrics.
    
    Args:
        dataset_path: Path to benchmark JSON file
        enable_tracing: Save query traces for debugging
        mode: Override VSM_MODE (default: use current setting)
        max_queries: Limit number of queries (for testing)
        save_report: Save full report to logs/benchmarks/
        
    Returns:
        Benchmark results with metrics and per-query records
    """
    settings = get_settings()
    run_mode = mode or settings.vsm_mode

    result = await run_benchmark(
        dataset_path=dataset_path,
        mode=run_mode,
        enable_tracing=enable_tracing,
        max_queries=max_queries,
    )
    
    # Persist report
    report_path = None
    if save_report:
        report_path = save_benchmark_report(result)

    response = serialize_run_result(result)
    if report_path:
        response["report_path"] = str(report_path)
    
    return response


@router.get("/benchmark/reports")
async def list_reports(
    limit: int = Query(10, ge=1, le=100),
) -> List[dict]:
    """
    List saved benchmark reports.
    
    Returns most recent reports with summary info.
    """
    logs_dir = Path("logs/benchmarks")
    if not logs_dir.exists():
        return []
    
    reports = []
    for filepath in sorted(logs_dir.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            reports.append({
                "filename": filepath.name,
                "mode": data.get("mode"),
                "timestamp": data.get("timestamp"),
                "total": data.get("total"),
                "completed": data.get("completed"),
                "summary": data.get("summary"),
            })
        except Exception:
            continue
    
    return reports


@router.get("/benchmark/reports/{filename}")
async def get_report(filename: str):
    """
    Get a specific benchmark report by filename.
    """
    filepath = Path("logs/benchmarks") / filename
    if not filepath.exists():
        return {"error": f"Report not found: {filename}"}
    
    with open(filepath, "r") as f:
        return json.load(f)
