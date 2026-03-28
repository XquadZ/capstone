import os
import json
import random
import time
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# ⚙️ 1. 설정 및 Ahoseo API 클라이언트
# ==========================================
INPUT_PATH = "evaluation/datasets/filtered_golden_dataset_2000_pure.json"
OUTPUT_PATH = "evaluation/datasets/final_vision_intent_dataset_v3_llm.json"

client = OpenAI(
    base_url="https://api.ahoseo.com/v1", 
    api_key=os.getenv("AHOSEO_API_KEY", "YOUR_API_KEY_HERE") 
)
MODEL_NAME = "gpt-4o-mini" 

# ==========================================
# 🧠 2. LLM 질문 변환 함수 (최종 진화형 프롬프트)
# ==========================================
def rewrite_to_vision_query(item):
    original_q = item.get("question", "")
    
    # 💡 형님의 아이디어 적용: 20개의 예시를 주고 고르게 만들기!
    system_prompt = """당신은 질문 수정 전문가입니다.
원본 질문의 의미는 100% 유지하되, 질문자가 시각 자료(표, 그림 등)를 보며 질문하는 것처럼 문장을 자연스럽게 재구성하세요.

아래 [시각적 의도 표현 20선] 중 질문의 맥락과 가장 잘 어울리는 것을 하나 고르세요. 
그리고 고른 표현을 문장의 앞, 중간, 뒤 중 가장 자연스러운 위치에 부드럽게 녹여내세요. 절대 로봇처럼 기계적으로 이어 붙이지 마세요!

[시각적 의도 표현 20선]
1. 첨부된 표를 참고해서 알려줘. 
2. 공지사항 내 이미지를 보면, 
3. 문서에 포함된 도표를 바탕으로 
4. 아래 시각 자료에 따르면 
5. 본문의 표 내용 기준으로, 
6. 제공된 그래프를 확인해 보면 
7. 사진 자료를 참고했을 때 
8. 첨부 파일의 구조도를 보면 
9. 화면 캡처 이미지를 바탕으로 
10. 문서 하단의 요약 표를 보고 대답해 줘. 
11. 여기 표에 나와있는 내용 중에서, 
12. 그림 자료를 보니까 궁금한 게 생겼는데, 
13. 안내문의 표를 보면서 확인하고 싶은데, 
14. 같이 첨부된 이미지를 기준으로 할 때, 
15. 표에 정리된 항목을 보면, 
16. (관련 표 참고해서 답변 부탁해)
17. - 이미지 자료 기준
18. (해당 내용이 정리된 표를 보고 알려줘)
19. 본문의 도표를 바탕으로 확인해 줄래?
20. 시각 자료에 나와 있는 내용으로 대답해.

[예시]
원본: 2024년 장학금 신청 기간은 언제야?
수정: 안내문의 표를 보면서 확인하고 싶은데, 2024년 장학금 신청 기간이 정확히 언제야?

수행 결과는 오직 '수정된 질문'만 출력하세요."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"원본 질문: {original_q}"}
            ],
            temperature=0.7 # 약간의 유연성을 줘서 자연스럽게 섞도록 유도
        )
        new_q = response.choices[0].message.content.strip()
        
        if new_q.startswith('"') and new_q.endswith('"'):
            new_q = new_q[1:-1]
            
        item["question"] = new_q
        item["route"] = "VISION" 
        
        return item
    except Exception as e:
        # 최후의 보루: 에러 나면 파이썬으로 20개 중 하나 무작위로 예쁘게 붙이기
        fallback_prefix = "공지된 표를 참고했을 때, "
        item["question"] = fallback_prefix + original_q
        item["route"] = "VISION"
        return item

# ==========================================
# 🚀 3. 메인 파싱 및 50:50 할당
# ==========================================
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 입력 파일이 없습니다: {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    total_data = len(qa_data)
    target_vision_count = total_data // 2 

    current_vision = [d for d in qa_data if d.get("route", "").upper() == "VISION"]
    current_text = [d for d in qa_data if d.get("route", "").upper() == "TEXT"]

    print(f"📥 총 {total_data}개 데이터 로드 완료.")
    print(f"   - 현재 VISION: {len(current_vision)}개")
    print(f"   - 현재 TEXT  : {len(current_text)}개")
    
    needed_more_vision = target_vision_count - len(current_vision)
    print(f"🎯 50:50 밸런스를 위해 TEXT 중 {needed_more_vision}개를 VISION으로 변환합니다!")

    random.seed(42)
    random.shuffle(current_text)

    text_to_become_vision = current_text[:needed_more_vision]
    keep_as_text = current_text[needed_more_vision:]

    items_to_rewrite = current_vision + text_to_become_vision

    print(f"\n🔥 총 {len(items_to_rewrite)}개의 질문에 LLM을 사용하여 시각적 의도를 주입합니다... (10개 단위 자동 저장!)")
    
    rewritten_vision_items = []
    
    for i, item in enumerate(tqdm(items_to_rewrite, desc="LLM 변환 중"), 1):
        result_item = rewrite_to_vision_query(item)
        rewritten_vision_items.append(result_item)
        time.sleep(0.05) 

        # 10개 단위로 파일 덮어쓰기 (백업용)
        if i % 10 == 0:
            current_dataset = keep_as_text + rewritten_vision_items + items_to_rewrite[i:]
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(current_dataset, f, ensure_ascii=False, indent=4)

    # 순정 TEXT 데이터 라벨 확정
    for item in keep_as_text:
        item["route"] = "TEXT"
        
    final_dataset = keep_as_text + rewritten_vision_items
    random.shuffle(final_dataset) 

    # 최종 완료본 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    t_count = sum(1 for x in final_dataset if x['route'] == 'TEXT')
    v_count = sum(1 for x in final_dataset if x['route'] == 'VISION')

    print("\n" + "="*60)
    print("🎯 [명시적 의도(Intent) 기반 50:50 황금 데이터셋 완성]")
    print(f"📊 총 {len(final_dataset)}개 데이터 (TEXT: {t_count}개 | VISION: {v_count}개)")
    print(f"💾 최종 저장 경로: {OUTPUT_PATH}")
    print("="*60)
    
    print("\n👀 [✨ 대망의 LLM 변환 샘플 확인 ✨]")
    sample_items = random.sample(rewritten_vision_items, min(5, len(rewritten_vision_items)))
    for i, item in enumerate(sample_items):
        print(f"[{i+1}] {item['question']}")

if __name__ == "__main__":
    main()