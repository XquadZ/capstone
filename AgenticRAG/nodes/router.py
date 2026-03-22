import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from AgenticRAG.graph.state import AgentState

# ==========================================
# 1. SLM 라우터 전역(Global) 로드 
# (서버가 켜질 때 1번만 로드하여 지연 시간 최소화)
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
router_model = PeftModel.from_pretrained(base_model, adapter_path)
router_model.eval()
print("✅ [Router Node] 라우터 부팅 완료. 대기 중.")

# ==========================================
# 2. 라우터 노드 실행 함수
# ==========================================
def slm_router_node(state: AgentState) -> dict:
    """
    사용자 질문을 분석하여 TEXT 또는 VISION 모드를 결정합니다.
    """
    question = state["question"]
    print(f"\n🚦 [Router] 질의 분석 중: '{question}'")
    
    # 학습 때와 100% 동일한 프롬프트(페르소나) 주입
    system_instruction = "당신은 질문의 난이도와 정보 결손을 파악하여 최적의 검색 경로를 결정하는 [AI 라우터]입니다. 다른 말은 생략하고 오직 'TEXT' 또는 'VISION' 으로만 대답하세요."
    
    messages = [
        {"role": "user", "content": f"{system_instruction}\n\n사용자 질문: {question}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    with torch.no_grad():
        output_tokens = router_model.generate(
            **inputs, 
            max_new_tokens=5,  # 딱 한 단어만 뱉으면 되므로 최소화
            do_sample=False, 
            use_cache=True
        )
        input_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True).strip().upper()
    
    # 모델 출력 파싱 및 안전 장치 (Fallback)
    decision = "TEXT" # 기본값
    if "VISION" in generated_text or "비전" in generated_text:
        decision = "VISION"
        
    print(f"🎯 [Router] 판정 결과: {decision} RAG 실행")
    
    # State 업데이트
    return {"route_decision": decision}