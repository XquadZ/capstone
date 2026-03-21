import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

print("🚀 [디버그 1단계] 라이브러리 및 환경 로드")
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
model_id = "google/gemma-2-2b-it"

print("🚀 [디버그 2단계] 모델 및 토크나이저 로드 (여기서 멈추면 GPU 메모리 할당 문제)")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

try:
    # 💡 의심 포인트 1 해결: device_map을 "auto" 대신 "cuda:0"으로 명확히 고정
    # 💡 의심 포인트 2 해결: 윈도우에서 자주 멈추는 SDPA 대신 "eager" 어텐션 사용
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"":"cuda:0"}, 
        torch_dtype=torch.float16,
        attn_implementation="eager", 
        token=hf_token
    )
    model.enable_input_require_grads()
    print("   ✅ 모델 로드 성공!")
except Exception as e:
    print(f"   ❌ 모델 로드 실패: {e}")
    exit()

print("🚀 [디버그 3단계] LoRA 적용 (여기서 멈추면 PEFT 라이브러리 문제)")
peft_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
print("   ✅ LoRA 적용 성공!")

print("🚀 [디버그 4단계] 더미 데이터 생성 및 GPU 전송")
inputs = tokenizer("딥러닝 프리징 원인 분석 테스트입니다.", return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
inputs["labels"] = inputs["input_ids"].clone()
print("   ✅ GPU 데이터 전송 성공!")

print("🚀 [디버그 5단계] 모델 연산(Forward) 시도 - 🚨 가장 유력한 프리징 후보 구간 🚨")
try:
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"   ✅ 연산 성공! (계산된 Loss: {loss.item():.4f})")
except Exception as e:
    print(f"   ❌ 연산 실패: {e}")
    exit()

print("🚀 [디버그 6단계] 역전파(Backward) 시도 - 🚨 두 번째 유력한 후보 구간 🚨")
try:
    loss.backward()
    print("   ✅ 역전파 (그래디언트 계산) 성공!")
except Exception as e:
    print(f"   ❌ 역전파 실패: {e}")
    exit()

print("\n🎉🎉🎉 [결론] 모델과 GPU 코어 연산에는 아무 문제가 없습니다! 🎉🎉🎉")
print("만약 여기까지 완벽하게 출력되었다면, 문제는 DPOTrainer의 데이터 전처리나 '참조 모델(Reference Model)' 복사 과정에 있는 것입니다.")