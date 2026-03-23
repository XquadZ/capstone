import os
import sys

# 💡 경로 문제 해결을 위한 절대 경로 추가 (안전장치)
sys.path.append(os.getcwd())

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from AgenticRAG.graph.state import AgentState

# ==========================================
# 1. SLM 라우터 전역(Global) 로드 
# ==========================================
base_model_id = "google/gemma-2-2b-it"
adapter_path = "hoseo_router_gemma_2b_sft" # SFT 학습 완료된 어댑터 경로

print("🧠 [Router Node] 지능형 SLM 라우터 부팅 중 (Gemma-2B SFT)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    attn_implementation="eager"
)

# 어댑터가 존재할 경우 로드, 없으면 베이스 모델 사용 (예외 처리)
try:
    if os.path.exists(adapter_path):
        router_model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"✅ [Router Node] SFT 어댑터 로드 완료: {adapter_path}")
    else:
        router_model = base_model
        print("⚠️ [Router Node] 어댑터를 찾지 못해 베이스 모델로 구동합니다.")
except Exception as e:
    print(f"❌ [Router Node] 모델 로드 실패: {e}")
    router_model = base_model

router_model.eval()
print("✅ [Router Node] 라우터 준비 완료.")

# ==========================================
# 2. 라우터 노드 실행 함수
# ==========================================
def slm_router_node(state: AgentState) -> dict:
    """
    사용자 질문을 분석하여 TEXT 또는 VISION 모드를 결정합니다.
    """
    question = state["question"]
    print(f"\n" + "🚦" * 20)
    print(f"🚦 [Router] 질의 분석 중: '{question}'")
    
    # SAIFE X 가이드라인에 따른 프롬프트 구성 [cite: 81]
    system_instruction = (
        "당신은 호서대학교 학사행정 전문 라우터입니다. "
        "질문의 의도가 표(Table), 수치 비교, 시각적 확인이 필요한 '비정형 데이터 분석'인 경우 'VISION', "
        "일반적인 절차, 서류, 단순 안내 등 '텍스트 기반 지식'인 경우 'TEXT'라고 판단하세요. "
        "다른 설명 없이 단어 하나만 대답하세요."
    )
    
    messages = [
        {"role": "user", "content": f"{system_instruction}\n\n사용자 질문: {question}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(router_model.device)
    
    with torch.no_grad():
        output_tokens = router_model.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=False, 
            use_cache=True
        )
        input_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True).strip().upper()
    
    # ✨ 출처 및 판정 근거 로직
    decision = "TEXT"
    if "VISION" in generated_text:
        decision = "VISION"
        reason = "표 분석 및 정밀 수치 추출이 필요한 시각적 질문으로 판명됨"
    else:
        reason = "일반적인 정보 안내 및 절차 확인 질문으로 판명됨"
        
    print(f"🎯 [Router] 판정 결과: {decision}")
    print(f"🧐 [Router] 판정 근거: {reason}")
    print("🚦" * 20 + "\n")
    
    # State 업데이트 (추후 분석을 위해 판정 근거도 context에 임시 저장 가능)
    return {
        "route_decision": decision,
        "context": [f"Router Decision: {decision} ({reason})"]
    }

# 단독 테스트용 코드
if __name__ == "__main__":
    test_state = {"question": "조기졸업 평점 기준이 표에 어떻게 나와있어?", "retry_count": 0}
    slm_router_node(test_state)