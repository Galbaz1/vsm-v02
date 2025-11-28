"""
Technical Judge Service for Benchmark Evaluation.

Uses DSPy to evaluate model answers against expected answers.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

import dspy

from api.prompts.signatures import TechnicalJudgeSignature
from api.core.dspy_config import get_dspy_lm

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from judging a single benchmark item."""
    score: float  # 0.0 - 1.0
    rationale: str
    cited_pages: List[str]
    
    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "rationale": self.rationale,
            "cited_pages": self.cited_pages,
        }


class TechnicalJudge(dspy.Module):
    """
    DSPy module for evaluating answer quality.
    
    Grades a model-generated answer against the expected answer,
    considering factual accuracy, completeness, and source citations.
    """
    
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(TechnicalJudgeSignature)
    
    def forward(
        self,
        query: str,
        answer: str,
        expected_answer: str,
        sources: str,
    ) -> dspy.Prediction:
        """
        Evaluate answer quality.
        
        Args:
            query: The benchmark query/question
            answer: Model-generated answer text
            expected_answer: Expected/reference answer text
            sources: JSON list of cited sources
            
        Returns:
            Prediction with score, rationale, cited_pages
        """
        return self.judge(
            query=query,
            answer=answer,
            expected_answer=expected_answer,
            sources=sources,
        )


# Singleton instance
_judge: Optional[TechnicalJudge] = None


def get_judge() -> TechnicalJudge:
    """Get or create the TechnicalJudge singleton."""
    global _judge
    if _judge is None:
        # Ensure DSPy is configured
        get_dspy_lm()
        _judge = TechnicalJudge()
    return _judge


async def evaluate_answer(
    query: str,
    answer: str,
    expected_answer: str,
    sources: Optional[List[dict]] = None,
) -> JudgeResult:
    """
    Evaluate a model answer against expected answer.
    
    Args:
        query: The benchmark query
        answer: Model-generated answer
        expected_answer: Ground truth answer
        sources: List of source dicts with page/manual info
        
    Returns:
        JudgeResult with score, rationale, and cited pages
    """
    import asyncio
    import dspy
    from api.core.dspy_config import get_dspy_lm
    
    # Capture DSPy LM for thread propagation
    lm = get_dspy_lm()
    
    judge = get_judge()
    sources_json = json.dumps(sources or [])
    
    # Run DSPy module in thread pool with settings context propagation
    def run_judge():
        with dspy.settings.context(lm=lm):
            return judge(
                query=query,
                answer=answer,
                expected_answer=expected_answer,
                sources=sources_json,
            )
    
    try:
        result = await asyncio.to_thread(run_judge)
        
        # Parse score (handle string or float)
        score = result.score
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0.5  # Default if parsing fails
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        # Parse cited pages
        cited_pages = []
        if result.cited_pages:
            try:
                cited_pages = json.loads(result.cited_pages)
                if not isinstance(cited_pages, list):
                    cited_pages = [str(cited_pages)]
            except json.JSONDecodeError:
                cited_pages = [result.cited_pages]
        
        return JudgeResult(
            score=score,
            rationale=result.rationale or "",
            cited_pages=cited_pages,
        )
        
    except Exception as e:
        logger.error(f"Judge evaluation failed: {e}")
        return JudgeResult(
            score=0.0,
            rationale=f"Evaluation error: {str(e)}",
            cited_pages=[],
        )

