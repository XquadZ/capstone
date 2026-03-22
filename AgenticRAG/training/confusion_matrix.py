import os

# ✨ [강력 조치] PyTorch 컴파일러(Triton) 사용을 OS 단에서 원천 봉쇄
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch._dynamo
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

torch._dynamo.config.suppress_errors = True

# ==========================================
# 1. 경로 및 환경 설정
# ==========================================
base_model_id = "google/gemma-2-2b-it"
adapter_path = "hoseo_router_gemma_2b"  # 9 Epoch 학습 완료된 모델 (가장 똑똑한 버전)
test_data_path = "test_dataset.jsonl"   

print("📦 토크나이저 및 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="eager" 
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# ==========================================
# 2. 데이터셋 로드 및 혼동 행렬 세팅
# ==========================================
print("\n🔍 지시어(Instruction) 강제 주입 후 최종 평가 시작...")
test_dataset = load_dataset("json", data_files=test_data_path, split="train")

matrix = {
    "True_TEXT_Pred_TEXT": 0,
    "True_TEXT_Pred_VISION": 0,
    "True_VISION_Pred_VISION": 0,
    "True_VISION_Pred_TEXT": 0
}

# 💡 모델의 라우터 자아를 깨우는 마법의 지시어
magic_instruction = "\n이 질문에 대해 텍스트 RAG와 비전 RAG 중 어떤 모드를 실행할까요?"

# ==========================================
# 3. 추론 및 정확도 측정 루프
# ==========================================
for i, example in enumerate(tqdm(test_dataset, desc="평가 진행도")):
    raw_prompt = example['prompt']
    
    # 🚨 [핵심 수정] 프롬프트에 지시어가 없으면 강제로 붙여줍니다!
    if magic_instruction.strip() not in raw_prompt:
        raw_prompt = raw_prompt.rstrip() + magic_instruction
        
    chosen = example['chosen'].upper().replace("[", "").replace("]", "").replace(" RAG", "").strip()
    actual_label = "TEXT" if "TEXT" in chosen else "VISION"
    
    messages = [{"role": "user", "content": raw_prompt}]
    templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(templated_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            use_cache=True 
        )
        
        input_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True).strip()
    
    # 예측 판별 (단답형 대비)
    pred_label = "UNKNOWN"
    if any(keyword in generated_text.lower() for keyword in ["text", "텍스트"]):
        pred_label = "TEXT"
    elif any(keyword in generated_text.lower() for keyword in ["vision", "비전", "비젼"]):
        pred_label = "VISION"
        
    # 혼동 행렬 기록
    if actual_label == "TEXT":
        if pred_label == "TEXT": matrix["True_TEXT_Pred_TEXT"] += 1
        else: matrix["True_TEXT_Pred_VISION"] += 1
    elif actual_label == "VISION":
        if pred_label == "VISION": matrix["True_VISION_Pred_VISION"] += 1
        else: matrix["True_VISION_Pred_TEXT"] += 1

# ==========================================
# 4. 최종 결과 출력
# ==========================================
total_text = matrix['True_TEXT_Pred_TEXT'] + matrix['True_TEXT_Pred_VISION']
total_vision = matrix['True_VISION_Pred_TEXT'] + matrix['True_VISION_Pred_VISION']
total_samples = total_text + total_vision

correct = matrix['True_TEXT_Pred_TEXT'] + matrix['True_VISION_Pred_VISION']
accuracy = (correct / total_samples * 100) if total_samples > 0 else 0

text_acc = (matrix['True_TEXT_Pred_TEXT'] / total_text * 100) if total_text > 0 else 0
vision_acc = (matrix['True_VISION_Pred_VISION'] / total_vision * 100) if total_vision > 0 else 0

print("\n" + "="*50)
print(f"🎯 최종 모델 성능 및 혼동 행렬 (Confusion Matrix)")
print("="*50)
print(f"               | 예측: TEXT (Text RAG) | 예측: VISION (Vision RAG)")
print("-" * 60)
print(f"실제: TEXT     | {matrix['True_TEXT_Pred_TEXT']:^21} | {matrix['True_TEXT_Pred_VISION']:^23}")
print(f"실제: VISION   | {matrix['True_VISION_Pred_TEXT']:^21} | {matrix['True_VISION_Pred_VISION']:^23}")
print("="*50)
print(f"🚀 전체 정확도 (Accuracy): {accuracy:.2f}% ({correct}/{total_samples})")
print(f"📌 TEXT 데이터 분류 정확도: {text_acc:.1f}%")
print(f"📌 VISION 데이터 분류 정확도: {vision_acc:.1f}%")