import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "google/gemma-2b-it"
LORA_ADAPTER_DIR = "experience/exp1/gemma_router_lora_stratified" 
TEST_DATA_PATH = "evaluation/datasets/sft_splits/test.jsonl"

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model.eval()

    test_data = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            test_data.append(json.loads(line))
            if i == 9: break # 딱 10개만 확인해봅시다.

    print("\n🔍 [모델의 실제 생성 텍스트 확인]")
    for i, item in enumerate(test_data):
        prompt = f"<bos><start_of_turn>user\n{item['instruction']}\n{item['input']}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, temperature=0.0)
            
        # 모델의 날것 그대로의 대답
        raw_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        print(f"[{i+1}] 정답: {item['output']} | 모델 대답: {raw_response}")

if __name__ == "__main__":
    main()