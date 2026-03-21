import torch
import torch._dynamo
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 1. 경로 설정
base_model_id = "google/gemma-2-2b-it"
adapter_path = "hoseo_router_gemma_2b"  # 학습 완료된 폴더
test_data_path = "test_dataset.jsonl"   # 어제 저장된 테스트셋
torch._dynamo.config.suppress_errors = True

print("🔍 최종 테스트 데이터 평가 시작...")

# 2. 모델 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# 3. 데이터셋 로드
test_dataset = load_dataset("json", data_files=test_data_path, split="train")

correct = 0
total = len(test_dataset)

# 4. 평가 루프
for example in tqdm(test_dataset):
    prompt = example['prompt']
    chosen = example['chosen'] # 정답 라우팅 결과 (예: [Vision RAG])
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # 모델의 응답에 정답 키워드가 포함되어 있는지 확인
    if chosen.strip() in response:
        correct += 1

# 5. 결과 출력
accuracy = (correct / total) * 100
print(f"\n🎯 최종 테스트 결과")
print(f"✅ 총 개수: {total}")
print(f"✅ 맞은 개수: {correct}")
print(f"🚀 최종 정확도: {accuracy:.2f}%")