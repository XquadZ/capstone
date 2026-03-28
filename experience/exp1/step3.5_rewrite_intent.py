import os
import json
import random
from tqdm import tqdm

# ==========================================
# ⚙️ 1. 설정 및 경로
# ==========================================
INPUT_PATH = "evaluation/datasets/filtered_golden_dataset_2000_pure.json"
# 💡 최종적으로 학습에 사용할 완벽한 데이터셋 파일명
OUTPUT_PATH = "evaluation/datasets/final_intent_balanced_dataset.json"

# ==========================================
# 🎨 2. 시각적 의도 주입 함수 (20가지 패턴 랜덤)
# ==========================================
def inject_visual_intent(item):
    original_q = item.get("question", "").strip()
    
    # 형님이 고르신 고품질 문구 20선
    visual_cues = [
        "첨부된 표를 참고해서 알려줘. ", "공지사항 내 이미지를 보면, ", "문서에 포함된 도표를 바탕으로 ",
        "아래 시각 자료에 따르면 ", "본문의 표 내용 기준으로, ", "제공된 그래프를 확인해 보면 ",
        "사진 자료를 참고했을 때 ", "첨부 파일의 구조도를 보면 ", "화면 캡처 이미지를 바탕으로 ",
        "문서 하단의 요약 표를 보고 대답해 줘. ", "여기 표에 나와있는 내용 중에서, ", 
        "그림 자료를 보니까 궁금한 게 생겼는데, ", "안내문의 표를 보면서 확인하고 싶은데, ", 
        "같이 첨부된 이미지를 기준으로 할 때, ", "표에 정리된 항목을 보면, ",
        " (관련 표 참고해서 답변 부탁해)", " - 이미지 자료 기준", 
        " (해당 내용이 정리된 표를 보고 알려줘)", " 본문의 도표를 바탕으로 확인해 줄래?", 
        " 시각 자료에 나와 있는 내용으로 대답해."
    ]
    
    selected_cue = random.choice(visual_cues)
    
    # 💡 문구의 특성(접두사/접미사)에 따라 자연스럽게 결합
    if selected_cue.startswith(" ") or selected_cue.startswith("("):
        # 접미사(Suffix) 형태: 질문 끝에 붙임
        if original_q.endswith("?"):
            new_q = original_q[:-1] + selected_cue + "?"
        else:
            new_q = original_q + selected_cue
    else:
        # 접두사(Prefix) 형태: 질문 앞에 붙임
        new_q = selected_cue + original_q

    item["question"] = new_q
    item["route"] = "VISION" # 라벨 강제 확정
    return item

# ==========================================
# 🚀 3. 메인 프로세스
# ==========================================
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 입력 파일이 없습니다: {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # 1. 기존 라벨별로 일단 분리
    current_vision = [d for d in qa_data if d.get("route", "").upper() == "VISION"]
    current_text = [d for d in qa_data if d.get("route", "").upper() == "TEXT"]

    total_count = len(qa_data)
    target_vision_count = total_count // 2 # 🎯 정확히 50% 타겟

    print(f"📥 데이터 로드 완료 (총 {total_count}개)")
    print(f"   - 기존 VISION: {len(current_vision)}개")
    print(f"   - 기존 TEXT: {len(current_text)}개")

    # 2. 5:5를 맞추기 위해 TEXT에서 데려올 개수 계산
    needed_from_text = target_vision_count - len(current_vision)
    
    random.seed(42) # 결과 재현을 위해 시드 고정
    random.shuffle(current_text)

    # VISION으로 개조할 TEXT 그룹 / 그대로 둘 TEXT 그룹 분리
    text_to_convert = current_text[:needed_from_text]
    keep_as_text = current_text[needed_from_text:]

    # 3. VISION 데이터 생성 (기존 VISION + 개조된 TEXT)
    print(f"\n🔥 총 {target_vision_count}개의 질문에 시각적 의도를 주입합니다...")
    
    final_vision_items = []
    
    # 기존 VISION 데이터들도 질문 내용 보강 (의도 명확화)
    for item in tqdm(current_vision, desc="기존 VISION 보강 중"):
        final_vision_items.append(inject_visual_intent(item))
        
    # TEXT에서 넘어온 데이터들 VISION으로 개조
    for item in tqdm(text_to_convert, desc="TEXT -> VISION 개조 중"):
        final_vision_items.append(inject_visual_intent(item))

    # 4. 순정 TEXT 데이터 라벨링 (내용 수정 없음)
    for item in keep_as_text:
        item["route"] = "TEXT"

    # 5. 합치기 및 무작위 섞기
    final_dataset = keep_as_text + final_vision_items
    random.shuffle(final_dataset)

    # 6. 최종 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    # 결과 요약
    v_final = sum(1 for x in final_dataset if x['route'] == 'VISION')
    t_final = sum(1 for x in final_dataset if x['route'] == 'TEXT')

    print("\n" + "="*60)
    print(f"✅ 작업 완료! 파일이 생성되었습니다.")
    print(f"📊 최종 비율 -> VISION: {v_final}개 | TEXT: {t_final}개 (완벽한 5:5 밸런스)")
    print(f"💾 경로: {OUTPUT_PATH}")
    print("="*60)

    # 샘플 출력
    print("\n👀 [학습용 VISION 질문 샘플 5개]")
    samples = random.sample(final_vision_items, 5)
    for i, s in enumerate(samples):
        print(f"[{i+1}] {s['question']}")

if __name__ == "__main__":
    main()