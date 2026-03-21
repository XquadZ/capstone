from typing import Dict, TypedDict, Literal
from langgraph.graph import StateGraph, END

# 1. 상태(State) 정의: 에이전트가 노드를 돌며 들고 다닐 '바구니'입니다.
class AgentState(TypedDict):
    question: str
    route_decision: str  # 'text' 또는 'vision'
    context: str         # 검색된 문서 내용
    generation: str      # 생성된 답변
    critic_score: float  # 평가 점수 (0.0 ~ 1.0)
    retry_count: int     # 재시도 횟수

# 2. 노드(Node) 정의: 각 단계에서 실행될 함수들입니다.
def router_node(state: AgentState) -> Dict:
    """질문을 분석하여 텍스트 RAG를 탈지 비전 RAG를 탈지 결정합니다."""
    print("--- [NODE: Router] 질문 난이도 및 문서 타입 분석 중 ---")
    question = state["question"]
    
    # TODO: Phase 2에서 여기에 4090 로컬 모델(Llama-3 라우터)이 들어갑니다.
    # 지금은 간단한 키워드 룰베이스로 더미 처리합니다.
    if "표" in question or "양식" in question or "절차도" in question:
        decision = "vision"
    else:
        decision = "text"
        
    print(f"    -> 라우팅 결정: {decision.upper()} RAG")
    return {"route_decision": decision, "retry_count": state.get("retry_count", 0)}

def text_rag_node(state: AgentState) -> Dict:
    """기존 rag_pipeline.py 가 실행될 자리입니다."""
    print("--- [NODE: Text RAG] 텍스트 기반 검색 및 생성 중 ---")
    # TODO: 여기에 기존 텍스트 검색 모듈 연동
    return {"context": "텍스트로 찾은 학칙 규정...", "generation": "텍스트 기반 답변입니다."}

def vision_rag_node(state: AgentState) -> Dict:
    """기존 vision_processor.py 가 실행될 자리입니다."""
    print("--- [NODE: Vision RAG] 멀티모달(colpali) 표/양식 분석 중 ---")
    # TODO: 여기에 기존 비전 검색 모듈 연동
    return {"context": "이미지에서 추출한 표 데이터...", "generation": "비전 기반의 정확한 답변입니다."}

def critic_node(state: AgentState) -> Dict:
    print("--- [NODE: Critic] 답변 품질 평가 중 ---")
    
    # 현재 재시도 횟수를 가져옵니다. (없으면 0)
    current_retry = state.get("retry_count", 0)
    
    if state["route_decision"] == "text":
        score = 0.5
        print(f"    -> [결과] 텍스트 유실 감지 (Score: 0.5, Retry: {current_retry})")
    else:
        score = 0.9
        print(f"    -> [결과] 비전 품질 우수 (Score: 0.9, Retry: {current_retry})")
        
    # 중요: 변경된 점수와 함께 'retry_count'를 1 증가시켜서 반환합니다.
    return {"critic_score": score, "retry_count": current_retry + 1}

# 3. 조건부 엣지(Conditional Edge) 함수 정의
def route_after_analysis(state: AgentState) -> Literal["text_rag_node", "vision_rag_node"]:
    """Router 결과에 따라 다음 노드를 결정합니다."""
    if state["route_decision"] == "vision":
        return "vision_rag_node"
    return "text_rag_node"

def check_hallucination(state: AgentState) -> Literal["vision_rag_node", "__end__"]:
    # 이제 state["retry_count"]가 정상적으로 증가된 상태입니다.
    if state["critic_score"] < 0.8 and state["retry_count"] <= 2: # 최대 2번까지 재시도
        print(f"--- [EDGE] 품질 미달 ({state['retry_count']}회차). Vision RAG로 재시도합니다. ---")
        return "vision_rag_node"
    
    print("--- [EDGE] 종료 조건을 만족하여 결과물을 출력합니다. ---")
    return "__end__"

# 4. 그래프(Graph) 조립
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("router_node", router_node)
workflow.add_node("text_rag_node", text_rag_node)
workflow.add_node("vision_rag_node", vision_rag_node)
workflow.add_node("critic_node", critic_node)

# 시작점 설정
workflow.set_entry_point("router_node")

# 엣지 연결
workflow.add_conditional_edges("router_node", route_after_analysis)
workflow.add_edge("text_rag_node", "critic_node")
workflow.add_edge("vision_rag_node", "critic_node")
workflow.add_conditional_edges("critic_node", check_hallucination)

# 그래프 컴파일
app = workflow.compile()

# 5. 실행 테스트 (메인 함수)
if __name__ == "__main__":
    print("\n[테스트 1] 일반 질문 (Text RAG 통과 예상)")
    inputs1 = {"question": "도서관 운영 시간이 어떻게 되나요?", "retry_count": 0}
    for output in app.stream(inputs1):
        pass
    
    print("\n" + "="*50 + "\n")
    
    print("[테스트 2] 복잡한 질문 (Text 실패 -> Vision 재시도 루프 예상)")
    inputs2 = {"question": "장학금 지급 기준 표를 설명해주세요.", "retry_count": 0}
    for output in app.stream(inputs2):
        pass