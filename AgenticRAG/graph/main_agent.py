import sys
import os
import time

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
    
    # 목표 점수(0.8) 도달 시 또는 최대 재시도(2회) 초과 시 종료
    if score >= 0.8 or retries >= 2:
        return END
    return "router" # 품질이 떨어지면 아예 처음부터 다시 경로를 고민하도록 라우터로 보냄

# ==========================================
# 🚀 TV-RAG (Text-Vision) 그래프 조립
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
# 테스트 실행부 (정밀 출처 출력 모드)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 [TV-RAG] 호서대 멀티모달 Agentic RAG 시스템 가동")
    print("="*70 + "\n")

    # 🧪 [실험 시나리오 설정] 
    # 시나리오 A: "조기졸업하려면 어떤 조건이 필요한지 장학금 기준이랑 같이 알려줘." (VISION 유도)
    # 시나리오 B: "휴학 신청을 하려면 어떤 서류가 필요하고 절차가 어떻게 돼?" (TEXT 유도)
    
    test_question = "차세대디스플레이 서포터즈 모집에 지원하면 어떤 혜택이 있어?"
    
    initial_state = {
        "question": test_question,
        "retry_count": 0
    }

    print(f"📥 [USER INPUT]: {test_question}")
    print("-" * 70)

    # LangGraph 실행 및 결과 스트리밍 시작
    for output in app.stream(initial_state):
        for node_name, state_value in output.items():
            
            # 1️⃣ RAG 노드(TEXT/VISION) 실행 결과 가로채기
            if node_name in ["text_rag", "vision_rag"]:
                generation = state_value.get("generation", "")
                context = state_value.get("context", [])
                
                if generation:
                    print("\n" + "✨" * 35)
                    print(f"📝 [{node_name.upper()} 노드 생성 답변]")
                    print("✨" * 35)
                    print(generation)
                    print("-" * 70)
                    
                    # 💡 정밀 출처(Source) 출력 로직
                    if context:
                        print("📍 [SYSTEM TRACE: 분석에 활용된 원본 데이터]")
                        for i, src in enumerate(context):
                            print(f"   ({i+1}) {src}")
                    print("="*70 + "\n")
            
            # 2️⃣ Critic 노드 평가 결과 출력
            if node_name == "critic":
                score = state_value.get("critic_score", 0.0)
                retry = state_value.get("retry_count", 0)
                
                print(f"⚖️ [Node: CRITIC] 답변 품질 검증 결과")
                print(f"   - 품질 점수: {score} / 1.0")
                print(f"   - 현재 재시도 횟수: {retry}")
                
                if score < 0.8:
                    print("⚠️ [품질 미달] 정보의 구체성이 부족하여 파이프라인을 재가동합니다. (Looping...)\n")
                else:
                    print("✅ [검증 통과] 신뢰할 수 있는 답변으로 판명되어 프로세스를 종료합니다.\n")

    print("🏁 [TV-RAG] 시스템 모든 처리가 완료되었습니다.")