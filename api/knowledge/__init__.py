"""
Knowledge Module - Atlas and Domain Knowledge for VSM.

This module provides the Atlas class for injecting domain-specific knowledge
into DSPy-powered agents, following the Elysia pattern.

Original pattern: docs/elysia-source-code-for-reference-only/elysia/tree/objects.py (lines 354-375)
"""

from datetime import datetime
from pydantic import BaseModel, Field


def datetime_reference() -> dict:
    """
    Generate current datetime context for decision-making.
    
    Returns:
        dict with current_datetime, current_day_of_week, current_time_of_day
    """
    date = datetime.now()
    return {
        "current_datetime": date.isoformat(),
        "current_day_of_week": date.strftime("%A"),
        "current_time_of_day": date.strftime("%I:%M %p"),
    }


class Atlas(BaseModel):
    """
    Domain knowledge container for agent guidance.
    
    Atlas provides context-aware guidance to DSPy agents, including:
    - Writing style preferences
    - Domain-specific descriptions
    - Goal definitions
    - Temporal context
    
    This is injected into DSPy signatures as an InputField with description:
    "Your guide to how you should proceed as an agent in this task."
    
    Example:
        >>> from api.knowledge import Atlas
        >>> atlas = Atlas(
        ...     style="Technical and precise",
        ...     agent_description="You are a security system expert",
        ...     end_goal="Answer installation questions accurately"
        ... )
        >>> print(atlas.style)
        Technical and precise
    """
    
    style: str = Field(
        default="No style provided.",
        description="The writing style of the agent. "
        "This is the way the agent writes, and the tone of the language it uses.",
    )
    agent_description: str = Field(
        default="No description provided.",
        description="The description of the process you are following. This is pre-defined by the user. "
        "This could be anything - this is the theme of the program you are a part of.",
    )
    end_goal: str = Field(
        default="No end goal provided.",
        description="A short description of your overall goal. "
        "Use this to determine if you have completed your task. "
        "However, you can still choose to end actions early, "
        "if you believe the task is not possible to be completed with what you have available.",
    )
    datetime_reference: dict = Field(
        default_factory=datetime_reference,
        description="Current context information (e.g., date, time) for decision-making",
    )


__all__ = ["Atlas", "datetime_reference"]

