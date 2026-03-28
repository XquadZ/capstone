import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# ⚙️ 1. 설정
# ==========================================
BASE_MODEL_ID = "google/gemma-2b-it"
# 💡 대망의 V4 모델 경로로 업데이트 완료!
LORA_ADAPTER_DIR = "experience/exp1/gemma_router_lora_v4" 
TEST_DATA_PATH = "evaluation/datasets/sft_splits/test.jsonl"

def main():
    print("🚀 [Step 6] 의도 기반 라우팅을 마스터한 V4 라우터의 최종 수능(Test) 평가를 시작합니다!")

    # ==========================================
    # 🧠 2. 모델 및 토크나이저 로드
    # ==========================================
    print("🤖 베이스 모델에 V4 LoRA 어댑터를 합체 중입니다...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model.eval()

    # ==========================================
    # 📝 3. 테스트 데이터 로드
    # ==========================================
    test_data = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
            
    print(f"📥 수능 시험지(Test Data) {len(test_data)}문제 로드 완료.")

    # ==========================================
    # 🎯 4. 추론(Inference) 실행
    # ==========================================
    y_true = []
    y_pred = []

    print("🔥 V4 모델이 문제를 풀고 있습니다... (이번엔 다를 겁니다!)")
    for item in tqdm(test_data):
        ground_truth = item['output']
        y_true.append(ground_truth)

        prompt = f"<bos><start_of_turn>user\n{item['instruction']}\n{item['input']}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0)
            
        full_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
        
        # 첫 번째 단어 낚아채기
        prediction_word = full_response.split()[0] if full_response.split() else "TEXT"

        # 최종 라벨링 확정
        if "VISION" in prediction_word:
            prediction = "VISION"
        else:
            prediction = "TEXT" 

        y_pred.append(prediction)

    # ==========================================
    # 📊 5. 결과 분석 및 시각화
    # ==========================================
    print("\n" + "="*60)
    print("🏆 [V4 모델 최종 성적표]")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=["TEXT", "VISION"]))

    cm = confusion_matrix(y_true, y_pred, labels=["TEXT", "VISION"])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["TEXT", "VISION"], 
                yticklabels=["TEXT", "VISION"])
    plt.title('Gemma Router V4 Confusion Matrix (Intent-based)')
    plt.xlabel('Predicted Route')
    plt.ylabel('True Route')
    
    save_path = "experience/exp1/router_v4_confusion_matrix.png"
    plt.savefig(save_path)
    print(f"📸 혼동 행렬 그래프가 '{save_path}'에 저장되었습니다. (논문 첨부용으로 완벽합니다!)")

if __name__ == "__main__":
    main()