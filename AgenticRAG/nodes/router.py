import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
# 중요:
# 현재 v4 어댑터는 이 base model과 size mismatch가 날 수 있습니다.
# 그래도 우선 경로는 정확하게 잡고, 실패 원인을 명확히 로그로 확인합니다.
BASE_MODEL_ID = "google/gemma-2b-it"
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "experience", "exp1", "gemma_router_lora_v4")

print("🧠 [Router Node] 지능형 SLM 라우터 부팅 중...")
print(f"   - PROJECT_ROOT : {PROJECT_ROOT}")
print(f"   - BASE_MODEL_ID: {BASE_MODEL_ID}")
print(f"   - ADAPTER_PATH : {ADAPTER_PATH}")

print("🔤 [Router Node] 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("📦 [Router Node] 베이스 모델 로드 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)

router_model = base_model
ROUTER_BACKEND = "BASE"

try:
    if os.path.exists(ADAPTER_PATH):
        print("🧩 [Router Node] LoRA 어댑터 로드 시도 중...")
        router_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        ROUTER_BACKEND = "LORA"
        print(f"✅ [Router Node] SFT 어댑터 로드 완료: {ADAPTER_PATH}")
    else:
        print(f"⚠️ [Router Node] 어댑터 경로가 존재하지 않습니다: {ADAPTER_PATH}")
        print("⚠️ [Router Node] 베이스 모델로 구동합니다.")
except Exception as e:
    print(f"❌ [Router Node] 모델 로드 실패: {e}")
    print("⚠️ [Router Node] LoRA 로드 실패로 베이스 모델 fallback 사용")
    router_model = base_model
    ROUTER_BACKEND = "BASE"

router_model.eval()
print(f"✅ [Router Node] 라우터 준비 완료. (backend={ROUTER_BACKEND})")


# ==========================================
# 2. 라우터 노드 실행 함수
# ==========================================
def slm_router_node(state: AgentState) -> dict:
    """
    사용자 질문을 분석하여 TEXT 또는 VISION 모드를 결정합니다.
    """
    question = state["question"]
    print("\n" + "🚦" * 20)
    print(f"🚦 [Router] 질의 분석 중: '{question}'")
    print(f"🚦 [Router] backend={ROUTER_BACKEND}")

    system_instruction = (
        "당신은 호서대학교 학사행정 전문 라우터입니다. "
        "질문의 의도가 표(Table), 수치 비교, 시각적 확인, 첨부파일/이미지/도표/구조도/시간표/명단 확인이 필요한 경우 'VISION', "
        "일반적인 절차, 서류, 단순 안내, 개념 설명, 텍스트 본문만으로 충분한 질문인 경우 'TEXT'라고 판단하세요. "
        "다른 설명 없이 단어 하나만 대답하세요."
    )

    messages = [
        {
            "role": "user",
            "content": f"{system_instruction}\n\n사용자 질문: {question}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(router_model.device)

    with torch.no_grad():
        output_tokens = router_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
        )
        input_length = inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(
            output_tokens[0][input_length:],
            skip_special_tokens=True
        ).strip().upper()

    decision = "TEXT"
    reason = "일반적인 정보 안내 및 절차 확인 질문으로 판명됨"

    if "VISION" in generated_text:
        decision = "VISION"
        reason = "표/이미지/첨부/시각 자료 기반 확인이 필요한 질문으로 판명됨"
    elif "TEXT" in generated_text:
        decision = "TEXT"

    print(f"🧾 [Router Raw Output] {generated_text}")
    print(f"🎯 [Router] 판정 결과: {decision}")
    print(f"🧐 [Router] 판정 근거: {reason}")
    print("🚦" * 20 + "\n")

    return {
        "route_decision": decision,
        "context": [f"Router Decision: {decision} ({reason})"],
    }


if __name__ == "__main__":
    test_questions = [
        "조기졸업 평점 기준이 표에 어떻게 나와있어?",
        "휴학 신청 절차가 어떻게 돼?",
        "첨부 파일의 구조도를 보면 취업박람회 일정이 어떻게 돼?",
        "아래 시각 자료에 따르면 봉사장학생 지원 자격이 뭐야?"
    ]

    for q in test_questions:
        test_state = {"question": q, "retry_count": 0}
        slm_router_node(test_state)
