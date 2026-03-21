# AgenticRAG/graph/state.py
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    route_decision: str
    context: str
    generation: str
    critic_score: float
    retry_count: int