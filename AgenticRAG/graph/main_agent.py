import sys
import os

# 현재 작업 디렉토리를 파이썬 경로에 추가
sys.path.append(os.getcwd())

from langgraph.graph import StateGraph, END
from AgenticRAG.graph.state import AgentState

# 각 노드 함수 불러오기
from AgenticRAG.nodes.router import slm_router_node
from AgenticRAG.nodes.text_rag import text_rag_node
from AgenticRAG.nodes.vision_rag import vision_rag_node
from AgenticRAG.nodes.critic import critic_node

def route_to_rag(state: AgentState) -> str:
    """
    라우터의 결정(route_decision)에 따라 다음 노드의 이름을 반환하는 조건부 엣지 함수
    """
    decision = state.get("route_decision", "TEXT")
    if decision == "VISION":
        return "vision_rag"
    return "text_rag"

def route_after_critic(state: AgentState) -> str:
    """
    Critic의 점수와 재시도 횟수에 따라 종료할지 다시 검색할지 결정
    """
    score = state.get("critic_score", 0.0)
    retries = state.get("retry_count", 0)
    
    if score >= 0.8 or retries >= 2:
        return END
    return "router" # 품질이 떨어지면 아예 처음부터 다시 경로를 고민하도록 라우터로 보냄

# ==========================================
# 🚀 Agentic RAG 그래프 조립
# ==========================================
workflow = StateGraph(AgentState)

# 1. 노드(Node) 추가
workflow.add_node("router", slm_router_node)
workflow.add_node("text_rag", text_rag_node)
workflow.add_node("vision_rag", vision_rag_node)
workflow.add_node("critic", critic_node)

# 2. 엣지(Edge) 및 흐름 연결
workflow.set_entry_point("router")

# 라우터 -> Text 또는 Vision 분기 (조건부 엣지)
workflow.add_conditional_edges(
    "router",
    route_to_rag,
    {
        "text_rag": "text_rag",
        "vision_rag": "vision_rag"
    }
)

# Text/Vision RAG가 끝나면 모두 Critic으로 모임
workflow.add_edge("text_rag", "critic")
workflow.add_edge("vision_rag", "critic")

# Critic 평가 후 종료 또는 재시도 루프
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        END: END,
        "router": "router"
    }
)

# 3. 그래프 컴파일
app = workflow.compile()

# ==========================================
# 테스트 실행 (직접 이 파일을 실행했을 때)
# ==========================================
if __name__ == "__main__":
    test_question = "조기졸업하려면 어떤 조건이 필요한지 장학금 기준이랑 같이 알려줘."
    print("===========================================")
    print("▶️ 호서대 멀티모달 Agentic RAG 시스템 가동")
    print("===========================================")
    
    initial_state = {
        "question": test_question,
        "retry_count": 0
    }
    
    # 그래프 실행
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"\n✅ [Node: {key}] 완료")
            
    print("\n🏁 최종 시스템 처리 완료!")