import os
# Triton 에러 방지용 환경변수
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# 1. 설정 (SFT 모델 & SFT 테스트 데이터)
# ==========================================
base_model_id = "google/gemma-2-2b-it"
adapter_path = "hoseo_router_gemma_2b_sft"  # ✨ SFT 모델 경로
test_data_path = "test_dataset_sft.jsonl"   # ✨ SFT 학습 때 빼둔 평가 데이터

print("📦 SFT 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager" 
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

test_dataset = load_dataset("json", data_files=test_data_path, split="train")

matrix = {
    "True_TEXT_Pred_TEXT": 0,
    "True_TEXT_Pred_VISION": 0,
    "True_VISION_Pred_VISION": 0,
    "True_VISION_Pred_TEXT": 0
}

# ==========================================
# 2. 평가 진행
# ==========================================
print("\n🔍 SFT 라우터 최종 성능 검증 시작 (Vision 예측력 집중 평가)...")

for example in tqdm(test_dataset, desc="평가 진행도"):
    # SFT 데이터의 'messages'에서 user 질문과 model 정답 분리
    user_msgs = [m for m in example['messages'] if m['role'] == 'user']
    model_msgs = [m for m in example['messages'] if m['role'] == 'model']
    
    if not user_msgs or not model_msgs: continue
    
    # 실제 정답 추출
    actual_label = "TEXT" if "TEXT" in model_msgs[0]['content'].upper() else "VISION"
    
    # 평가를 위해 정답을 뺀 프롬프트(사용자 질문만) 생성
    prompt = tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        input_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True).strip().upper()
    
    # 예측 판별
    pred_label = "UNKNOWN"
    if "TEXT" in generated_text or "텍스트" in generated_text:
        pred_label = "TEXT"
    elif "VISION" in generated_text or "비전" in generated_text:
        pred_label = "VISION"
        
    # 혼동 행렬 기록
    if actual_label == "TEXT":
        if pred_label == "TEXT": matrix["True_TEXT_Pred_TEXT"] += 1
        else: matrix["True_TEXT_Pred_VISION"] += 1
    elif actual_label == "VISION":
        if pred_label == "VISION": matrix["True_VISION_Pred_VISION"] += 1
        else: matrix["True_VISION_Pred_TEXT"] += 1

# ==========================================
# 3. 결과 출력
# ==========================================
total_text = matrix['True_TEXT_Pred_TEXT'] + matrix['True_TEXT_Pred_VISION']
total_vision = matrix['True_VISION_Pred_TEXT'] + matrix['True_VISION_Pred_VISION']
total_samples = total_text + total_vision

correct = matrix['True_TEXT_Pred_TEXT'] + matrix['True_VISION_Pred_VISION']
accuracy = (correct / total_samples * 100) if total_samples > 0 else 0

text_acc = (matrix['True_TEXT_Pred_TEXT'] / total_text * 100) if total_text > 0 else 0
vision_acc = (matrix['True_VISION_Pred_VISION'] / total_vision * 100) if total_vision > 0 else 0

print("\n" + "="*50)
print("📊 [캡스톤 논문 삽입용] SFT 라우터 혼동 행렬 (Confusion Matrix)")
print("="*50)
print(f"               | 예측: TEXT (Text RAG) | 예측: VISION (Vision RAG)")
print("-" * 60)
print(f"실제: TEXT     | {matrix['True_TEXT_Pred_TEXT']:^21} | {matrix['True_TEXT_Pred_VISION']:^23}")
print(f"실제: VISION   | {matrix['True_VISION_Pred_TEXT']:^21} | {matrix['True_VISION_Pred_VISION']:^23}")
print("="*50)
print(f"🚀 전체 정확도 (Accuracy): {accuracy:.1f}% ({correct}/{total_samples})")
print(f"📌 TEXT 질의 방어율: {text_acc:.1f}%")
print(f"🎯 VISION 질의 예측률 (노벨티 핵심): {vision_acc:.1f}%")
print("="*50)