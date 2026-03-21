# AgenticRAG/graph/main_agent.py
import os
import sys
from typing import Dict, TypedDict, Literal
from langgraph.graph import StateGraph, END

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 노드 파일들 임포트
from AgenticRAG.nodes.text_rag import text_rag_node
from AgenticRAG.nodes.vision_rag import vision_rag_node
from AgenticRAG.nodes.critic import critic_node
from AgenticRAG.graph.state import AgentState

# Router는 일단 간단히 내부에 둡니다. (추후 로컬 모델로 교체)
def router_node(state: AgentState) -> Dict:
    decision = "vision" if any(k in state["question"] for k in ["표", "양식", "기준"]) else "text"
    return {"route_decision": decision, "retry_count": state.get("retry_count", 0)}

def route_after_analysis(state: AgentState) -> Literal["text_rag_node", "vision_rag_node"]:
    return "vision_rag_node" if state["route_decision"] == "vision" else "text_rag_node"

def check_hallucination(state: AgentState) -> Literal["vision_rag_node", "__end__"]:
    if state["critic_score"] < 0.8 and state["retry_count"] <= 2:
        return "vision_rag_node"
    return "__end__"

# 그래프 조립 (기존과 동일)
workflow = StateGraph(AgentState)
workflow.add_node("router_node", router_node)
workflow.add_node("text_rag_node", text_rag_node)
workflow.add_node("vision_rag_node", vision_rag_node)
workflow.add_node("critic_node", critic_node)

workflow.set_entry_point("router_node")
workflow.add_conditional_edges("router_node", route_after_analysis)
workflow.add_edge("text_rag_node", "critic_node")
workflow.add_edge("vision_rag_node", "critic_node")
workflow.add_conditional_edges("critic_node", check_hallucination)

app = workflow.compile()

# main_agent.py 하단 수정
if __name__ == "__main__":
    print("\n🚀 [실전 테스트] 질문을 던집니다...")
    
    # 테스트 질문 입력
    inputs = {
        "question": "휴학 신청 절차와 필요한 서류를 알려줘.", # 실제 학칙에 있을법한 질문
        "retry_count": 0
    }
    
    # 에이전트 실행 (노드별로 결과 출력)
    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"\n[시스템 로그] 노드 '{key}' 실행 완료")
                # 답변이 생성되었다면 출력
                if "generation" in value and value["generation"]:
                    print(f"--- [최종 결과물] ---\n{value['generation']}\n--------------------")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")