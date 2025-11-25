"""
Agent schemas - Result, Error, Response types.

Adapted from Weaviate Elysia framework for VSM's dual-pipeline RAG.

Original: https://github.com/weaviate/elysia/blob/main/elysia/objects.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class PayloadType(str, Enum):
    """Types of payloads that can be streamed to frontend."""
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"
    RESPONSE = "response"
    DECISION = "decision"
    COMPLETE = "complete"


@dataclass
class Result:
    """
    Result object yielded from tools to store retrieved data.
    
    This gets automatically added to the Environment when yielded.
    
    Attributes:
        objects: List of retrieved objects (each is a dict)
        metadata: Metadata about the retrieval (query, timing, etc.)
        name: Name for indexing in environment (e.g., collection name)
        llm_message: Optional message to show LLM about this result
        display: Whether to send to frontend
    
    Example:
        yield Result(
            objects=[
                {"content": "...", "page_number": 12, "bbox": "[[0,0],[100,100]]"},
                {"content": "...", "page_number": 15, "bbox": "[[50,50],[150,150]]"},
            ],
            metadata={
                "query": "voltage specifications",
                "count": 2,
                "time_ms": 450,
                "collection": "AssetManual"
            },
            name="AssetManual",
            llm_message="Found 2 relevant chunks about voltage specifications."
        )
    """
    objects: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    name: str = "default"
    llm_message: Optional[str] = None
    display: bool = True
    
    def __post_init__(self):
        self.metadata = self.metadata or {}
    
    def __len__(self) -> int:
        return len(self.objects)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": PayloadType.RESULT.value,
            "objects": self.objects,
            "metadata": self.metadata,
            "name": self.name,
            "llm_message": self.llm_message,
            "count": len(self.objects),
        }
    
    def to_frontend(self, query_id: str) -> Dict[str, Any]:
        """Format for streaming to frontend."""
        if not self.display:
            return {}
        
        return {
            "type": PayloadType.RESULT.value,
            "query_id": query_id,
            "payload": {
                "objects": self.objects,
                "metadata": self.metadata,
                "name": self.name,
                "count": len(self.objects),
            }
        }


@dataclass
class Error:
    """
    Error object for self-healing - informs LLM without crashing.
    
    When yielded, errors are stored in TreeData and shown to the LLM
    so it can decide whether to retry or try a different approach.
    
    Attributes:
        message: Human-readable error description
        recoverable: Whether the LLM should try to recover
        suggestion: Hint for how to recover
        error_type: Category of error for routing
    
    Example:
        yield Error(
            message="No results found for query 'xyzzy specifications'",
            recoverable=True,
            suggestion="Try broadening the search terms or using different keywords"
        )
    """
    message: str
    recoverable: bool = True
    suggestion: Optional[str] = None
    error_type: str = "search_error"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": PayloadType.ERROR.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion,
            "error_type": self.error_type,
        }
    
    def to_frontend(self, query_id: str) -> Dict[str, Any]:
        """Format for streaming to frontend."""
        return {
            "type": PayloadType.ERROR.value,
            "query_id": query_id,
            "payload": {
                "message": self.message,
                "recoverable": self.recoverable,
                "suggestion": self.suggestion,
            }
        }


@dataclass
class Response:
    """
    Text response to show to the user.
    
    Used for final answers, explanations, or intermediate messages.
    
    Attributes:
        text: The response text
        sources: Optional list of source references
        display: Whether to show on frontend
    
    Example:
        yield Response(
            text="Based on the Technical Manual, the maximum voltage is 24V DC.",
            sources=[{"page": 12, "manual": "Technical Manual"}]
        )
    """
    text: str
    sources: Optional[List[Dict[str, Any]]] = None
    display: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": PayloadType.RESPONSE.value,
            "text": self.text,
            "sources": self.sources or [],
        }
    
    def to_frontend(self, query_id: str) -> Dict[str, Any]:
        """Format for streaming to frontend."""
        if not self.display:
            return {}
        
        return {
            "type": PayloadType.RESPONSE.value,
            "query_id": query_id,
            "payload": {
                "text": self.text,
                "sources": self.sources or [],
            }
        }


@dataclass
class Status:
    """
    Status update for real-time progress indication.
    
    Example:
        yield Status("Searching AssetManual collection...")
    """
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": PayloadType.STATUS.value,
            "message": self.message,
        }
    
    def to_frontend(self, query_id: str) -> Dict[str, Any]:
        return {
            "type": PayloadType.STATUS.value,
            "query_id": query_id,
            "payload": {"message": self.message}
        }


@dataclass
class Decision:
    """
    Decision made by the agent about which tool to use.
    
    Attributes:
        tool_name: Name of the chosen tool
        inputs: Inputs to pass to the tool
        reasoning: Why this tool was chosen
        should_end: Whether to end after this tool
        impossible: Whether the task is impossible
    """
    tool_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    should_end: bool = False
    impossible: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": PayloadType.DECISION.value,
            "tool_name": self.tool_name,
            "inputs": self.inputs,
            "reasoning": self.reasoning,
            "should_end": self.should_end,
            "impossible": self.impossible,
        }
    
    def to_frontend(self, query_id: str) -> Dict[str, Any]:
        return {
            "type": PayloadType.DECISION.value,
            "query_id": query_id,
            "payload": {
                "tool": self.tool_name,
                "reasoning": self.reasoning,
            }
        }


@dataclass
class Complete:
    """Signal that the agent has completed processing."""
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": PayloadType.COMPLETE.value}
    
    def to_frontend(self, query_id: str) -> Dict[str, Any]:
        return {
            "type": PayloadType.COMPLETE.value,
            "query_id": query_id,
            "payload": {}
        }


# Type alias for any yielded object from tools
ToolOutput = Result | Error | Response | Status | Decision | Complete

