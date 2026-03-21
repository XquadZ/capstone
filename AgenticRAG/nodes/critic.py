# AgenticRAG/nodes/critic.py
from typing import Dict
from AgenticRAG.graph.state import AgentState
from ai_engine.rag_pipeline_rules import client, LLM_MODEL_NAME # 기존 OpenAI 클라이언트 재사용

def critic_node(state: AgentState) -> Dict:
    print("--- [NODE: Critic] LLM 기반 답변 품질 평가 중 ---")
    
    question = state["question"]
    generation = state["generation"]
    current_retry = state.get("retry_count", 0)

    # 4090 서버의 LLM을 사용하여 셀프 채점
    prompt = f"""
    당신은 학칙 전문 평가관입니다. 아래 답변이 질문에 대해 충분한 정보를 제공하는지, 특히 표나 수치 정보가 누락되지 않았는지 평가하세요.
    
    질문: {question}
    답변: {generation}
    
    [평가 기준]
    - 답변이 완벽하고 근거가 명확하면: 1.0
    - 답변이 모호하거나 '찾을 수 없다'고 하거나 정보 유실이 의심되면: 0.5
    
    다른 설명 없이 오직 숫자(점수)만 출력하세요. 예: 0.5
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
        score = float(response.choices[0].message.content.strip())
    except:
        score = 0.5 # 에러 시 보수적으로 재시도 유도

    print(f"    -> [LLM 채점 점수] {score} (Retry: {current_retry})")
    return {"critic_score": score, "retry_count": current_retry + 1}