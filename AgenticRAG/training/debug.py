import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch._dynamo
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

torch._dynamo.config.suppress_errors = True

# ==========================================
# 1. 모델 로드
# ==========================================
base_model_id = "google/gemma-2-2b-it"
adapter_path = "hoseo_router_gemma_2b"  
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
# 2. VISION 정답 데이터만 필터링해서 테스트
# ==========================================
print("\n🧐 [정답이 VISION인 데이터] 모델의 실제 답변 까보기")
test_dataset = load_dataset("json", data_files=test_data_path, split="train")

# 정답이 VISION인 데이터만 수집
vision_samples = [ex for ex in test_dataset if "VISION" in ex['chosen'].upper()][:5]

if not vision_samples:
    print("⚠️ 테스트 데이터셋에 VISION 정답이 하나도 없습니다!")
else:
    for i, ex in enumerate(vision_samples):
        print("\n" + "-"*50)
        print(f"[{i+1}] 실제 정답: VISION")
        print(f"질문: {ex['prompt']}")
        
        # 모델 대화 형식 포장
        messages = [{"role": "user", "content": ex['prompt']}]
        templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(templated_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=30,  # 모델이 뱉는 말 끝까지 보기 위해 30으로 설정
                do_sample=False, 
                use_cache=True
            )
            
            input_length = inputs['input_ids'].shape[1]
            generated_text = tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True).strip()
            
        print(f"\n🤖 모델 대답: >> {generated_text} <<")
print("-" * 50)