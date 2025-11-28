"""
Chat endpoint with conversation memory.

Supports multi-turn conversations with the agent, persisting
state between messages within a session.
"""

import json
import uuid
from typing import Optional
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from api.services.agent import get_agent
from api.services.environment import TreeData, Environment

router = APIRouter()

# In-memory session store (for production, use Redis or database)
_sessions: dict[str, TreeData] = {}


def get_or_create_session(session_id: Optional[str]) -> tuple[str, TreeData]:
    """Get existing session or create new one."""
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    
    # Create new session
    new_id = str(uuid.uuid4())[:8]
    tree_data = TreeData()
    _sessions[new_id] = tree_data
    return new_id, tree_data


@router.post("/chat")
async def chat(
    message: str = Body(..., description="User's message"),
    session_id: Optional[str] = Body(None, description="Session ID for conversation continuity"),
):
    """
    Chat with the agent, maintaining conversation history.
    
    The agent remembers previous messages and retrieved data within a session.
    
    Request body:
    ```json
    {
        "message": "What are the jumper settings for 24V?",
        "session_id": "abc123"  // Optional, creates new session if not provided
    }
    ```
    
    Response format (NDJSON):
    ```
    {"type": "session", "session_id": "abc123", "payload": {"is_new": true}}
    {"type": "status", "query_id": "...", "payload": {"message": "..."}}
    {"type": "decision", "query_id": "...", "payload": {"tool": "...", "reasoning": "..."}}
    {"type": "result", "query_id": "...", "payload": {"objects": [...], "name": "..."}}
    {"type": "response", "query_id": "...", "payload": {"text": "...", "sources": [...]}}
    {"type": "complete", "query_id": "...", "payload": {}}
    ```
    """
    
    async def generate():
        # Get or create session
        sid, tree_data = get_or_create_session(session_id)
        is_new = session_id is None or session_id not in _sessions
        
        # Emit session info first
        yield json.dumps({
            "type": "session",
            "session_id": sid,
            "payload": {
                "is_new": is_new,
                "message_count": len(tree_data.conversation_history),
            }
        }) + "\n"
        
        # Add user message to history
        tree_data.add_conversation_message("user", message)
        
        # Reset iteration counter for new turn (but keep environment/history)
        tree_data.num_iterations = 0
        tree_data.user_prompt = message
        
        # Run agent with existing tree_data
        agent = get_agent()
        
        async for output in agent.run_with_state(message, tree_data):
            if output:
                yield json.dumps(output) + "\n"
        
        # Save updated session
        _sessions[sid] = tree_data
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/chat/sessions")
async def list_sessions():
    """List all active chat sessions with summary info."""
    sessions = []
    for sid, tree_data in _sessions.items():
        sessions.append({
            "session_id": sid,
            "message_count": len(tree_data.conversation_history),
            "last_query": tree_data.user_prompt or None,
            "has_data": not tree_data.environment.is_empty(),
        })
    return {"sessions": sessions}


@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full session state including conversation history."""
    if session_id not in _sessions:
        return {"error": f"Session not found: {session_id}"}
    
    tree_data = _sessions[session_id]
    return {
        "session_id": session_id,
        "conversation_history": tree_data.conversation_history,
        "environment_summary": tree_data.environment.to_llm_context(max_tokens=2000),
        "tasks_completed_count": len(tree_data.tasks_completed),
    }


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": True, "session_id": session_id}
    return {"deleted": False, "error": f"Session not found: {session_id}"}


@router.post("/chat/sessions")
async def create_session():
    """Create a new empty chat session."""
    new_id = str(uuid.uuid4())[:8]
    _sessions[new_id] = TreeData()
    return {"session_id": new_id, "created": True}

