import pandas as pd
import json

# 1. 5:5로 잘 맞춰진 기존 데이터 로드
dpo_data_path = "AgenticRAG/rl_training/dpo_dataset_balanced_final.jsonl"
df = pd.read_json(dpo_data_path, lines=True)

sft_data = []

# 2. SFT (대화형) 포맷으로 변환
for index, row in df.iterrows():
    # prompt에는 이미 페르소나와 지시어가 포함되어 있습니다.
    user_text = row['prompt']
    
    # DPO의 'chosen'이 SFT에서는 '유일한 정답(Target)'이 됩니다.
    # 안전장치: 혹시 모를 공백이나 특수문자 제거 후 딱 TEXT / VISION만 남김
    target_text = "TEXT" if "TEXT" in row['chosen'].upper() else "VISION"
    
    # Gemma가 완벽하게 이해하는 messages 리스트 포맷
    message = {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "model", "content": target_text}
        ]
    }
    sft_data.append(message)

# 3. SFT 전용 데이터셋으로 저장
sft_save_path = "AgenticRAG/rl_training/sft_dataset.jsonl"
with open(sft_save_path, "w", encoding="utf-8") as f:
    for item in sft_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n" + "="*50)
print(f"✅ SFT(지도학습)용 데이터 변환 완료!")
print(f"✅ 총 데이터 개수: {len(sft_data)}개")
print(f"💾 저장 경로: {sft_save_path}")
print("="*50)