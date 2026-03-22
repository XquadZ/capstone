import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 설정
# ==========================================
base_model_id = "google/gemma-2-2b-it"
adapter_path = "hoseo_router_gemma_2b_sft"  
test_data_path = "test_dataset_sft.jsonl"   

print("📦 SFT 모델 및 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager" 
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 데이터 로드 및 50개만 추출
dataset = load_dataset("json", data_files=test_data_path, split="train")
subset_size = min(50, len(dataset))
test_subset = dataset.select(range(subset_size))

print("\n" + "="*70)
print(f"🧐 [실제 대답 까보기] 총 {subset_size}개 데이터 테스트 시작!")
print("="*70)

# ==========================================
# 2. 추론 및 출력 루프
# ==========================================
match_count = 0

for i, example in enumerate(test_subset):
    user_msgs = [m for m in example['messages'] if m['role'] == 'user']
    model_msgs = [m for m in example['messages'] if m['role'] == 'model']
    
    if not user_msgs or not model_msgs: continue
    
    query_text = user_msgs[0]['content']
    actual_label = "TEXT" if "TEXT" in model_msgs[0]['content'].upper() else "VISION"
    
    # 모델에 넣을 프롬프트 포장
    prompt = tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    with torch.no_grad():
        # max_new_tokens=15로 여유있게 주어 모델이 헛소리를 덧붙이는지 감시합니다.
        output_tokens = model.generate(**inputs, max_new_tokens=15, do_sample=False, use_cache=True)
        input_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True).strip()
    
    # 결과 비교
    pred_label = "UNKNOWN"
    if "TEXT" in generated_text.upper() or "텍스트" in generated_text:
        pred_label = "TEXT"
    elif "VISION" in generated_text.upper() or "비전" in generated_text:
        pred_label = "VISION"
        
    is_match = actual_label == pred_label
    if is_match: match_count += 1
    
    icon = "✅" if is_match else "❌"
    
    # 보기 좋게 출력
    # 페르소나 지시어 부분은 너무 기니까 빼고 실제 질문만 잘라서 보여줍니다.
    display_query = query_text.split("이 질문에 대해 텍스트 RAG와")[0].replace("당신은 호서대학교 RAG 시스템의 [AI 라우터]입니다. 다른 부가 설명은 절대 하지 말고 오직 'TEXT' 또는 'VISION'으로만 대답하세요.\n\n", "").strip()
    
    print(f"[{i+1:02d}] {icon} 실제: {actual_label:6s} | 예측: {generated_text}")
    print(f"     🗣️ 질문: {display_query}")
    print("-" * 70)

print(f"\n🎯 50개 중 일치 개수: {match_count}개 ({match_count/subset_size*100:.1f}%)")
print("==================================================")