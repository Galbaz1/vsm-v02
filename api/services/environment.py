"""
Environment and TreeData - Adapted from Weaviate Elysia framework.

This module provides centralized state management for the agentic RAG system,
inspired by Elysia's patterns but adapted for VSM's dual-pipeline architecture.

Original: https://github.com/weaviate/elysia/blob/main/elysia/tree/objects.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from copy import deepcopy
import json


class Environment:
    """
    Persistent store of all retrieved objects and tool outputs.
    
    Structure:
    {
        "tool_name": {
            "result_name": [
                {
                    "metadata": dict,
                    "objects": list[dict],
                },
                ...
            ]
        }
    }
    
    Example:
    {
        "fast_vector_search": {
            "AssetManual": [
                {
                    "metadata": {"query": "voltage", "count": 5, "time_ms": 450},
                    "objects": [
                        {"content": "...", "page_number": 12, "bbox": "...", "_REF_ID": "fast_vector_search_AssetManual_0_0"},
                        ...
                    ]
                }
            ]
        },
        "colqwen_search": {
            "PDFDocuments": [
                {
                    "metadata": {"query": "wiring diagram", "count": 3, "time_ms": 3200},
                    "objects": [
                        {"page_number": 45, "image_path": "...", "score": 0.89, "_REF_ID": "colqwen_search_PDFDocuments_0_0"},
                        ...
                    ]
                }
            ]
        }
    }
    """
    
    def __init__(
        self,
        environment: Optional[Dict[str, Dict[str, Any]]] = None,
        hidden_environment: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the environment.
        
        Args:
            environment: Pre-existing environment data (for loading saved state)
            hidden_environment: Data not shown to LLM but accessible to tools
        """
        self.environment = environment or {}
        self.hidden_environment = hidden_environment or {}
    
    def is_empty(self) -> bool:
        """Check if environment has any retrieved data."""
        for tool_key in self.environment:
            for result_key in self.environment[tool_key]:
                if len(self.environment[tool_key][result_key]) > 0:
                    return False
        return True
    
    def add(self, tool_name: str, result: "Result") -> None:
        """
        Add a Result to the environment.
        
        Args:
            tool_name: Name of the tool that produced this result
            result: Result object to add
        """
        if tool_name not in self.environment:
            self.environment[tool_name] = {}
        
        name = result.name or "default"
        objects = result.objects
        metadata = result.metadata or {}
        
        self.add_objects(tool_name, name, objects, metadata)
    
    def add_objects(
        self,
        tool_name: str,
        name: str,
        objects: List[Dict],
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add objects directly to the environment.
        
        Args:
            tool_name: Name of the tool
            name: Name/category for this result set
            objects: List of object dictionaries
            metadata: Optional metadata about the retrieval
        """
        if tool_name not in self.environment:
            self.environment[tool_name] = {}
        
        if name not in self.environment[tool_name]:
            self.environment[tool_name][name] = []
        
        if len(objects) > 0:
            # Assign _REF_IDs to objects without them
            for i, obj in enumerate(objects):
                if "_REF_ID" not in obj:
                    idx = len(self.environment[tool_name][name])
                    obj["_REF_ID"] = f"{tool_name}_{name}_{idx}_{i}"
            
            self.environment[tool_name][name].append({
                "metadata": metadata or {},
                "objects": objects,
            })
    
    def find(
        self,
        tool_name: str,
        name: Optional[str] = None,
        index: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Find data in the environment.
        
        Args:
            tool_name: Name of the tool
            name: Optional result name to filter by
            index: Optional index within the result list
            
        Returns:
            The requested data, or None if not found
        """
        if tool_name not in self.environment:
            return None
        
        if name is None:
            return self.environment[tool_name]
        
        if name not in self.environment[tool_name]:
            return None
        
        data = self.environment[tool_name][name]
        
        if index is not None:
            return data[index] if index < len(data) else None
        return data
    
    def remove(
        self,
        tool_name: str,
        name: str,
        index: Optional[int] = None,
    ) -> None:
        """Remove data from the environment."""
        if tool_name in self.environment:
            if name in self.environment[tool_name]:
                if index is None:
                    self.environment[tool_name][name] = []
                else:
                    self.environment[tool_name][name].pop(index)
    
    def get_all_objects(self) -> List[Dict]:
        """Get all objects from all tools flattened into a single list."""
        all_objects = []
        for tool_name in self.environment:
            for name in self.environment[tool_name]:
                for entry in self.environment[tool_name][name]:
                    all_objects.extend(entry.get("objects", []))
        return all_objects
    
    def estimate_tokens(self) -> int:
        """Estimate token count of environment content (rough: chars / 4)."""
        return len(json.dumps(self.environment)) // 4
    
    def to_llm_context(self, max_tokens: int = 10000) -> str:
        """
        Format environment for LLM consumption.
        
        Args:
            max_tokens: Maximum tokens to include (truncates if exceeded)
            
        Returns:
            Formatted string representation
        """
        if self.is_empty():
            return "No data has been retrieved yet."
        
        lines = ["Retrieved data:"]
        
        for tool_name in self.environment:
            for name in self.environment[tool_name]:
                entries = self.environment[tool_name][name]
                for entry in entries:
                    metadata = entry.get("metadata", {})
                    objects = entry.get("objects", [])
                    
                    lines.append(f"\n[{tool_name} → {name}]")
                    if metadata:
                        lines.append(f"  Query: {metadata.get('query', 'N/A')}")
                        lines.append(f"  Count: {len(objects)}")
                    
                    for obj in objects[:5]:  # Limit to 5 objects per entry
                        preview = str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj)
                        lines.append(f"  - {preview}")
                    
                    if len(objects) > 5:
                        lines.append(f"  ... and {len(objects) - 5} more")
        
        content = "\n".join(lines)
        
        # Truncate if too long
        if len(content) // 4 > max_tokens:
            content = content[:max_tokens * 4] + "\n... (truncated)"
        
        return content
    
    def to_json(self) -> Dict:
        """Serialize environment to JSON-compatible dict."""
        return {
            "environment": deepcopy(self.environment),
            "hidden_environment": deepcopy(self.hidden_environment),
        }
    
    @classmethod
    def from_json(cls, data: Dict) -> "Environment":
        """Deserialize environment from JSON."""
        return cls(
            environment=data.get("environment", {}),
            hidden_environment=data.get("hidden_environment", {}),
        )


@dataclass
class TreeData:
    """
    Central state object passed to all tools.
    
    Contains everything a tool needs to make decisions and access context.
    """
    
    # User's current query
    user_prompt: str = ""
    
    # Persistent environment with retrieved data
    environment: Environment = field(default_factory=Environment)
    
    # Conversation history
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Track which tasks have been completed
    tasks_completed: Dict[str, List[float]] = field(default_factory=dict)
    
    # Errors that have occurred (for self-healing)
    errors: List[str] = field(default_factory=list)
    
    # Current iteration count
    num_iterations: int = 0
    
    # Maximum iterations before forced stop
    max_iterations: int = 10
    
    # Available Weaviate collections
    collection_names: List[str] = field(default_factory=lambda: ["AssetManual", "PDFDocuments"])
    
    # Current datetime for temporal context
    datetime_ref: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set datetime reference after init."""
        now = datetime.now()
        self.datetime_ref = {
            "current_datetime": now.isoformat(),
            "current_day": now.strftime("%A"),
            "current_time": now.strftime("%I:%M %p"),
        }
    
    def add_conversation_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        if content:
            # If last message is same role, append
            if self.conversation_history and self.conversation_history[-1]["role"] == role:
                self.conversation_history[-1]["content"] += " " + content
            else:
                self.conversation_history.append({"role": role, "content": content})
    
    def record_task(self, task_name: str, duration_ms: float) -> None:
        """Record that a task was completed with its duration."""
        if task_name not in self.tasks_completed:
            self.tasks_completed[task_name] = []
        self.tasks_completed[task_name].append(duration_ms)
    
    def add_error(self, error_message: str) -> None:
        """Add an error for self-healing context."""
        self.errors.append(error_message)
    
    def clear_errors(self) -> None:
        """Clear errors after successful recovery."""
        self.errors = []
    
    def iteration_status(self) -> str:
        """Get formatted iteration status for LLM."""
        status = f"{self.num_iterations + 1}/{self.max_iterations}"
        if self.num_iterations >= self.max_iterations - 1:
            status += " (FINAL - must complete or end)"
        return status
    
    def to_json(self) -> Dict:
        """Serialize TreeData to JSON."""
        return {
            "user_prompt": self.user_prompt,
            "environment": self.environment.to_json(),
            "conversation_history": self.conversation_history,
            "tasks_completed": self.tasks_completed,
            "errors": self.errors,
            "num_iterations": self.num_iterations,
            "max_iterations": self.max_iterations,
            "collection_names": self.collection_names,
            "datetime_ref": self.datetime_ref,
        }
    
    @classmethod
    def from_json(cls, data: Dict) -> "TreeData":
        """Deserialize TreeData from JSON."""
        tree_data = cls(
            user_prompt=data.get("user_prompt", ""),
            environment=Environment.from_json(data.get("environment", {})),
            conversation_history=data.get("conversation_history", []),
            tasks_completed=data.get("tasks_completed", {}),
            errors=data.get("errors", []),
            num_iterations=data.get("num_iterations", 0),
            max_iterations=data.get("max_iterations", 10),
            collection_names=data.get("collection_names", []),
        )
        tree_data.datetime_ref = data.get("datetime_ref", {})
        return tree_data


# Import Result here to avoid circular imports
# This is defined in api/schemas/agent.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from api.schemas.agent import Result

