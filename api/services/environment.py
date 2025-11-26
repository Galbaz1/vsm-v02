"""
Environment and TreeData - Adapted from Weaviate Elysia framework.

This module provides centralized state management for the agentic RAG system,
inspired by Elysia's patterns but adapted for VSM's dual-pipeline architecture.

Original: https://github.com/weaviate/elysia/blob/main/elysia/tree/objects.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from copy import deepcopy
import json

if TYPE_CHECKING:
    from api.knowledge import Atlas


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
    
    # Track which tasks have been completed (rich context for LLM)
    # Structure: [{"prompt": str, "task": [{"task": str, "iteration": int, ...}]}]
    tasks_completed: List[Dict[str, Any]] = field(default_factory=list)
    
    # Errors that have occurred (for self-healing), keyed by tool name
    errors: Dict[str, List[str]] = field(default_factory=dict)
    
    # Current tool being executed (for error context filtering)
    current_tool: Optional[str] = None
    
    # Current iteration count
    num_iterations: int = 0
    
    # Maximum iterations before forced stop
    max_iterations: int = 10
    
    # Available Weaviate collections
    collection_names: List[str] = field(default_factory=lambda: ["AssetManual", "PDFDocuments"])
    
    # Current datetime for temporal context
    datetime_ref: Dict[str, str] = field(default_factory=dict)
    
    # Domain knowledge for agent guidance (optional)
    atlas: Optional["Atlas"] = None
    
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
    
    def _update_task_field(self, task_dict: Dict, key: str, value: Any) -> None:
        """Helper to update a task field, appending if it already exists."""
        if value is None:
            return
        if key in task_dict:
            # Append to existing value based on type
            if isinstance(value, str):
                task_dict[key] += "\n" + value
            elif isinstance(value, (int, float)):
                task_dict[key] += value
            elif isinstance(value, list):
                task_dict[key].extend(value)
            elif isinstance(value, dict):
                task_dict[key].update(value)
            elif isinstance(value, bool):
                task_dict[key] = value
        else:
            task_dict[key] = value
    
    def update_tasks_completed(
        self,
        prompt: str,
        task: str,
        num_iterations: int,
        **kwargs,
    ) -> None:
        """
        Update tasks completed with rich context for LLM consumption.
        
        Handles three cases:
        1. New prompt - create new entry
        2. Existing prompt, new task - append to task list
        3. Existing prompt, existing task, new iteration - append new task entry
        4. Existing prompt, existing task, same iteration - update with kwargs
        
        Args:
            prompt: The user prompt this task is for
            task: The tool/task name
            num_iterations: Current iteration number
            **kwargs: Additional fields (reasoning, inputs, llm_message, action, error, duration_ms)
        
        Pattern from Elysia: elysia/tree/objects.py:685-742
        """
        prompt_idx = -1
        task_idx = -1
        iteration_found = False
        
        # Search for existing prompt and task
        for i, task_prompt in enumerate(self.tasks_completed):
            if task_prompt["prompt"] == prompt:
                prompt_idx = i
                for j, task_entry in enumerate(task_prompt["task"]):
                    if task_entry["task"] == task:
                        task_idx = j
                        # Only set iteration_found if THIS task has the iteration
                        if task_entry.get("iteration") == num_iterations:
                            iteration_found = True
                break  # Stop once we find the matching prompt
        
        # Case 1: New prompt - create new entry
        if prompt_idx == -1:
            self.tasks_completed.append({
                "prompt": prompt,
                "task": [{
                    "task": task,
                    "iteration": num_iterations,
                }]
            })
            for key, value in kwargs.items():
                self._update_task_field(self.tasks_completed[-1]["task"][0], key, value)
            return
        
        # Case 2: Existing prompt, new task - append to task list
        if task_idx == -1:
            self.tasks_completed[prompt_idx]["task"].append({
                "task": task,
                "iteration": num_iterations,
            })
            for key, value in kwargs.items():
                self._update_task_field(self.tasks_completed[prompt_idx]["task"][-1], key, value)
            return
        
        # Case 3: Existing prompt, existing task, new iteration - append new task entry
        if not iteration_found:
            self.tasks_completed[prompt_idx]["task"].append({
                "task": task,
                "iteration": num_iterations,
            })
            for key, value in kwargs.items():
                self._update_task_field(self.tasks_completed[prompt_idx]["task"][-1], key, value)
            return
        
        # Case 4: Existing prompt, existing task, same iteration - update with kwargs
        for key, value in kwargs.items():
            self._update_task_field(self.tasks_completed[prompt_idx]["task"][task_idx], key, value)
    
    def set_current_tool(self, task: str) -> None:
        """Set the current tool being executed (for error context)."""
        self.current_tool = task
    
    def get_errors(self) -> List[str]:
        """
        Get errors relevant to current context.
        
        If current_tool is set, returns only errors for that tool.
        Otherwise returns all errors.
        """
        if self.current_tool is None or self.current_tool not in self.errors:
            # Return all errors flattened
            all_errors = []
            for tool_errors in self.errors.values():
                all_errors.extend(tool_errors)
            return all_errors
        return self.errors.get(self.current_tool, [])
    
    def add_error(self, error_message: str) -> None:
        """Add an error for self-healing context."""
        tool = self.current_tool or "_global"
        if tool not in self.errors:
            self.errors[tool] = []
        self.errors[tool].append(error_message)
    
    def clear_errors(self, tool: Optional[str] = None) -> None:
        """
        Clear errors after successful recovery.
        
        Args:
            tool: If provided, clear only errors for this tool. Otherwise clear all.
        """
        if tool is not None:
            if tool in self.errors:
                self.errors[tool] = []
        else:
            self.errors = {}
    
    def tasks_completed_string(self) -> str:
        """
        Output a nicely formatted string of tasks completed for LLM consumption.
        
        This formats the tasks with XML-like tags for structure, showing:
        - Which prompts have been processed
        - What tasks were executed for each prompt
        - Whether they succeeded or failed
        - Reasoning and outputs from each task
        
        Pattern from Elysia: elysia/tree/objects.py:759-798
        
        Returns:
            Formatted string for LLM context injection
        """
        if not self.tasks_completed:
            return "No tasks have been completed yet."
        
        out = ""
        for j, task_prompt in enumerate(self.tasks_completed):
            out += f"<prompt_{j+1}>\n"
            out += f"Prompt: {task_prompt['prompt']}\n"
            
            for i, task in enumerate(task_prompt.get("task", [])):
                out += f"<task_{i+1}>\n"
                
                # Show task name with action indicator
                if task.get("action", True):
                    out += (
                        f"Chosen action: {task['task']} "
                        "(this does not mean it has been completed, "
                        "only that it was chosen, use the environment to judge if a task is completed)\n"
                    )
                else:
                    out += f"Chosen subcategory: {task['task']} (this action has not been completed, this is only a subcategory)\n"
                
                # Show success/error status
                if task.get("error", False):
                    out += (
                        " (UNSUCCESSFUL) There was an error during this tool call. "
                        "See the error messages for details. This action did not complete.\n"
                    )
                else:
                    out += " (SUCCESSFUL)\n"
                    # Show additional fields
                    for key in task:
                        if key not in ("task", "action", "error"):
                            out += f"{key.capitalize()}: {task[key]}\n"
                
                out += f"</task_{i+1}>\n"
            out += f"</prompt_{j+1}>\n"
        
        return out
    
    def iteration_status(self) -> str:
        """Get formatted iteration status for LLM."""
        status = f"{self.num_iterations + 1}/{self.max_iterations}"
        if self.num_iterations >= self.max_iterations - 1:
            status += " (FINAL - must complete or end)"
        return status
    
    def to_json(self) -> Dict:
        """Serialize TreeData to JSON."""
        data = {
            "user_prompt": self.user_prompt,
            "environment": self.environment.to_json(),
            "conversation_history": self.conversation_history,
            "tasks_completed": self.tasks_completed,
            "errors": self.errors,
            "current_tool": self.current_tool,
            "num_iterations": self.num_iterations,
            "max_iterations": self.max_iterations,
            "collection_names": self.collection_names,
            "datetime_ref": self.datetime_ref,
        }
        if self.atlas is not None:
            data["atlas"] = self.atlas.model_dump()
        return data
    
    @classmethod
    def from_json(cls, data: Dict) -> "TreeData":
        """Deserialize TreeData from JSON."""
        # Handle atlas deserialization
        atlas = None
        if "atlas" in data and data["atlas"] is not None:
            from api.knowledge import Atlas
            atlas = Atlas.model_validate(data["atlas"])
        
        # Handle backwards compatibility for tasks_completed (old format was Dict)
        tasks_completed = data.get("tasks_completed", [])
        if isinstance(tasks_completed, dict):
            # Convert old format to new format
            tasks_completed = []
        
        # Handle backwards compatibility for errors (old format was List)
        errors = data.get("errors", {})
        if isinstance(errors, list):
            # Convert old format to new format
            errors = {"_global": errors} if errors else {}
        
        tree_data = cls(
            user_prompt=data.get("user_prompt", ""),
            environment=Environment.from_json(data.get("environment", {})),
            conversation_history=data.get("conversation_history", []),
            tasks_completed=tasks_completed,
            errors=errors,
            num_iterations=data.get("num_iterations", 0),
            max_iterations=data.get("max_iterations", 10),
            collection_names=data.get("collection_names", []),
            atlas=atlas,
        )
        tree_data.current_tool = data.get("current_tool")
        tree_data.datetime_ref = data.get("datetime_ref", {})
        return tree_data


# Import Result here to avoid circular imports
# This is defined in api/schemas/agent.py
if TYPE_CHECKING:
    from api.schemas.agent import Result

