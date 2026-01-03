"""
Autonomous Learning Agent package.

This package contains:
- State definitions
- Prompt templates
- Research agent graph components
"""

from .state_scope import AgentState, AgentInputState
from .prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
)

__all__ = [
    "AgentState",
    "AgentInputState",
    "clarify_with_user_instructions",
    "transform_messages_into_research_topic_prompt",
]
