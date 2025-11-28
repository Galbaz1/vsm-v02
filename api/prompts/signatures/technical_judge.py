"""
Technical Judge Signature for benchmarking.

Scores a model answer against an expected answer and sources.
"""

import dspy


class TechnicalJudgeSignature(dspy.Signature):
    """
    Evaluate answer quality for a benchmark item.

    Inputs:
        query: Original user query
        answer: Model-produced answer text
        expected_answer: Ground-truth or reference answer
        sources: JSON list of cited sources with page/manual info
    Outputs:
        score: Numeric quality score (0.0 - 1.0)
        rationale: Short explanation for the score
        cited_pages: JSON list of pages/manuals considered correct
    """

    # Inputs
    query: str = dspy.InputField(desc="The benchmark query/question")
    answer: str = dspy.InputField(desc="Model-generated answer text")
    expected_answer: str = dspy.InputField(desc="Expected/reference answer text")
    sources: str = dspy.InputField(desc="JSON list of cited sources [{page, manual, relevance}]")

    # Outputs
    score: float = dspy.OutputField(desc="Quality score between 0.0 and 1.0")
    rationale: str = dspy.OutputField(desc="Why this score was assigned")
    cited_pages: str = dspy.OutputField(desc="JSON list of accepted pages/manuals")
