import os
import json
import random
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ 1. 설정
# ==========================================
# 💡 방금 생성한 완벽한 5:5 밸런스 데이터셋 경로!
INPUT_PATH = "evaluation/datasets/final_intent_balanced_dataset.json" 
OUTPUT_DIR = "evaluation/datasets/sft_splits"

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 입력 파일이 없습니다: {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    print(f"📥 총 {len(qa_data)}개의 완벽한 5:5 데이터를 SFT 포맷으로 변환합니다...")

    sft_dataset = []
    labels = []

    for item in qa_data:
        q = item.get("question", "")
        route = item.get("route", "TEXT").upper()

        sft_data_point = {
            "instruction": "주어진 질문에 답변하기 위해 텍스트 기반 검색(TEXT)과 시각적 문서 검색(VISION) 중 어느 것이 더 적합한지 판단하여 하나만 출력하세요.",
            "input": f"질문: {q}",
            "output": route
        }
        
        sft_dataset.append(sft_data_point)
        labels.append(route)

    # ---------------------------------------------------------
    # ✂️ 2. 8:1:1 계층적 분할 (Stratified Split)
    # ---------------------------------------------------------
    print("\n✂️ 증강/삭제 로직 없이 아주 깔끔하게 Train(80%), Valid(10%), Test(10%)로 쪼갭니다...")
    
    # 1차 분할: Train(80%) vs Temp(20%) - 5:5 비율 유지
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        sft_dataset, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # 2차 분할: Temp(20%)를 반으로 쪼개서 Valid(10%) vs Test(10%) - 5:5 비율 유지
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # 모델이 순서를 외우지 못하게 Train 데이터만 화려하게 섞어줌
    random.shuffle(train_data) 

    # ---------------------------------------------------------
    # 💾 3. 파일 저장
    # ---------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    paths = {
        "train": os.path.join(OUTPUT_DIR, "train.jsonl"),
        "val": os.path.join(OUTPUT_DIR, "val.jsonl"),
        "test": os.path.join(OUTPUT_DIR, "test.jsonl")
    }

    def save_jsonl(data_list, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_data, paths["train"])
    save_jsonl(val_data, paths["val"])
    save_jsonl(test_data, paths["test"])

    # 결과 출력용 헬퍼 함수
    def count_labels(data_list):
        t = sum(1 for x in data_list if x['output'] == 'TEXT')
        v = sum(1 for x in data_list if x['output'] == 'VISION')
        return t, v

    print("\n" + "="*60)
    print("🎯 [SFT 데이터 8:1:1 순정 분할 완료]")
    
    t_train, v_train = count_labels(train_data)
    print(f"📊 [순정 Train] (80%): 총 {len(train_data)}개 (TEXT: {t_train}, VISION: {v_train}) -> 완벽한 5:5 밸런스 유지! 🔥")
    
    t_val, v_val = count_labels(val_data)
    print(f"📊 [순정 Valid] (10%): 총 {len(val_data)}개 (TEXT: {t_val}, VISION: {v_val})")
    
    t_test, v_test = count_labels(test_data)
    print(f"📊 [순정 Test]  (10%): 총 {len(test_data)}개 (TEXT: {t_test}, VISION: {v_test})")
    print("="*60)
    print("🔥 속이 다 시원하네요! 이제 이 깔끔한 train.jsonl 파일로 Gemma 학습(Step 5)에 들어갑니다.")

if __name__ == "__main__":
    main()