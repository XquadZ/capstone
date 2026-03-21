import os
import sys
import json
from datetime import datetime
from typing import Dict, Literal

# ==========================================
# 1. 프로젝트 경로 설정 (순환 참조 및 Import 에러 방지)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# LangGraph 및 내부 모듈 임포트
from langgraph.graph import StateGraph, END
from AgenticRAG.graph.state import AgentState
from AgenticRAG.nodes.text_rag import text_rag_node
from AgenticRAG.nodes.vision_rag import vision_rag_node
from AgenticRAG.nodes.critic import critic_node

# ==========================================
# 2. 라우터 & 엣지 제어 함수
# ==========================================
def router_node(state: AgentState) -> Dict:
    print("--- [NODE: Router] 질문 난이도 및 문서 타입 분석 중 ---")
    question = state["question"]
    
    # 임시 룰베이스 라우터 (나중에 DPO 학습된 4090 로컬 모델로 교체할 부분)
    if "표" in question or "양식" in question or "기준" in question or "별표" in question:
        decision = "vision"
    else:
        decision = "text"
        
    print(f"    -> 라우팅 결정: {decision.upper()} RAG")
    return {"route_decision": decision, "retry_count": state.get("retry_count", 0)}

def route_after_analysis(state: AgentState) -> Literal["text_rag_node", "vision_rag_node"]:
    if state["route_decision"] == "vision":
        return "vision_rag_node"
    return "text_rag_node"

def check_hallucination(state: AgentState) -> Literal["vision_rag_node", "__end__"]:
    # Critic 점수가 0.8 미만이면 Vision으로 재시도 (최대 2번)
    if state["critic_score"] < 0.8 and state["retry_count"] <= 2:
        print(f"\n--- [EDGE] 품질 미달 ({state['retry_count']}회차). Vision RAG로 경로를 수정하여 재시도합니다. ---")
        return "vision_rag_node"
    
    print("\n--- [EDGE] 만족스러운 답변 도출. 프로세스를 종료합니다. ---")
    return "__end__"

# ==========================================
# 3. LangGraph 조립
# ==========================================
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("router_node", router_node)
workflow.add_node("text_rag_node", text_rag_node)
workflow.add_node("vision_rag_node", vision_rag_node)
workflow.add_node("critic_node", critic_node)

# 흐름(엣지) 연결
workflow.set_entry_point("router_node")
workflow.add_conditional_edges("router_node", route_after_analysis)
workflow.add_edge("text_rag_node", "critic_node")
workflow.add_edge("vision_rag_node", "critic_node")
workflow.add_conditional_edges("critic_node", check_hallucination)

app = workflow.compile()

# ==========================================
# 4. DPO 학습용 경험 데이터(Experience) 수집기
# ==========================================
def save_experience_data(final_state: dict):
    """에이전트의 최종 결과를 jsonl 형태로 저장하여 나중에 강화학습에 사용합니다."""
    # 저장할 폴더가 없으면 자동 생성
    save_dir = os.path.join(project_root, "AgenticRAG", "rl_training")
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, "collected_experience.jsonl")
    
    # 파일에 기록할 알짜배기 데이터만 추출
    data = {
        "timestamp": datetime.now().isoformat(),
        "question": final_state.get("question", ""),
        "route_taken": final_state.get("route_decision", ""),
        "retry_count": final_state.get("retry_count", 0),
        "critic_score": final_state.get("critic_score", 0.0),
        "generation": final_state.get("generation", "")
    }
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
    print(f"💾 [데이터 수집 완료] rl_training/collected_experience.jsonl 에 저장되었습니다.")
    print(f"   (기록된 점수: {data['critic_score']} / 최종 경로: {data['route_taken']})")

# ==========================================
# 5. 메인 실행 루프
# ==========================================
if __name__ == "__main__":
    print("\n🚀 [실전 테스트] 질문을 던집니다...")
    
    # 💡 팁: 나중에는 이 부분을 for문으로 바꿔서 질문 100개를 한 번에 돌리면 됩니다.
    inputs = {
        "question": "휴학 신청 절차와 필요한 서류를 알려줘.", 
        "retry_count": 0
    }
    
    final_state = None
    
    try:
        # app.stream을 통해 노드를 지날 때마다 상태(output)를 받아옵니다.
        for output in app.stream(inputs):
            for key, value in output.items():
                # 콘솔 가독성을 위해 노드 실행 완료 메시지 출력
                # print(f"\n[시스템 로그] 노드 '{key}' 실행 완료") 
                final_state = value # 계속 덮어씌워서 마지막 노드(Critic)의 상태를 확보
        
        # 그래프 실행이 끝난 후, 확보된 최종 상태를 저장합니다.
        if final_state:
            save_experience_data(final_state)
            
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")