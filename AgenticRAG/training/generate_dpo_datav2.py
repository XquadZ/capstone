import pandas as pd

# 1. GPT-4o-mini가 공들여 만든 원본 데이터 로드
dataset_path = "AgenticRAG/rl_training/dpo_dataset.jsonl"
df = pd.read_json(dataset_path, lines=True)

print(f"📦 원본 데이터: 총 {len(df)}개")

# 2. 텍스트/비전 분리
text_df = df[df['chosen'].str.contains('TEXT', case=False, na=False)]
vision_df = df[df['chosen'].str.contains('VISION', case=False, na=False)]

print(f"📊 실제 판정 비율 - TEXT: {len(text_df)}개, VISION: {len(vision_df)}개")

# 3. 5:5 황금 밸런싱 (Oversampling)
# VISION 데이터가 부족하므로, TEXT 개수만큼 복사해서 뻥튀기합니다.
vision_oversampled = vision_df.sample(n=len(text_df), replace=True, random_state=42)
balanced_df = pd.concat([text_df, vision_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. 🧠 페르소나 및 지시어 강제 주입
system_persona = "당신은 질문의 난이도와 정보 결손을 파악하여 최적의 검색 경로를 결정하는 [AI 라우터]입니다. 다른 말은 생략하고 오직 'TEXT' 또는 'VISION'으로만 대답하세요.\n\n"
instruction = "\n이 질문에 대해 텍스트 RAG와 비전 RAG 중 어떤 모드를 실행할까요?"

def inject_persona(row):
    original_prompt = row['prompt']
    
    # 혹시 기존에 지시어가 붙어있다면 떼어냄
    original_prompt = original_prompt.replace("이 질문에 대해 텍스트 RAG와 비전 RAG 중 어떤 모드를 실행할까요?", "").strip()
    
    # [페르소나] + [질문] + [단호한 지시어]
    row['prompt'] = system_persona + original_prompt + instruction
    
    # Chosen/Rejected 단답형 고정 (챗봇 말투 원천 봉쇄)
    row['chosen'] = "TEXT" if "TEXT" in row['chosen'].upper() else "VISION"
    row['rejected'] = "VISION" if "TEXT" in row['chosen'].upper() else "TEXT"
    
    return row

final_df = balanced_df.apply(inject_persona, axis=1)

# 5. 새로운 파일로 저장
new_path = "AgenticRAG/rl_training/dpo_dataset_balanced_final.jsonl"
final_df.to_json(new_path, orient='records', lines=True, force_ascii=False)

print("\n==================================================")
print(f"✅ 오리지널 로직 보존 + 5:5 밸런싱 + 페르소나 장착 완료!")
print(f"✅ 최종 학습 데이터: {len(final_df)}개 (TEXT {len(text_df)}개 : VISION {len(text_df)}개)")
print("==================================================")