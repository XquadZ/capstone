import os
import json

# 📂 파일 경로 설정 (캡처 화면에 맞춰서 파일명 수정 완료!)
NOTICE_PATH = "evaluation/datasets/ragas_testset_300.json"        # 기존 공지사항 데이터
RULES_PATH = "evaluation/datasets/vision_ragas_testset_300.json"  # 방금 밤새 만든 찐 학칙 데이터
FINAL_PATH = "evaluation/datasets/final_qa_testset.json"          # 내일 쓸 최종 무기

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("🔄 1. 데이터 로드 및 병합 시작...")
    notice_data = load_data(NOTICE_PATH)
    rules_data = load_data(RULES_PATH)
    
    all_raw_data = notice_data + rules_data
    print(f"📥 로드된 총 데이터 개수: {len(all_raw_data)}개 (기존: {len(notice_data)} + 신규학칙: {len(rules_data)})")

    final_valid_data = []
    error_count = 0

    print("🕵️‍♂️ 2. 무결성 전수 검사 (question, ground_truth, context_used 누락 확인)...")
    
    for item in all_raw_data:
        # 공백 제거 후 데이터 유효성 검사
        q = item.get("question", "")
        if isinstance(q, str): q = q.strip()
        
        gt = item.get("ground_truth", "")
        if isinstance(gt, str): gt = gt.strip()
            
        ctx = item.get("context_used", "")
        if isinstance(ctx, str): ctx = ctx.strip()
        
        # 💡 세 가지 핵심 키값이 모두 정상적으로 존재해야만 통과!
        if q and gt and ctx:
            final_valid_data.append({
                "id": len(final_valid_data), # 💡 내일 LangGraph 추적을 위해 깔끔하게 새 ID 부여
                "question": q,
                "ground_truth": gt,
                "context_used": ctx
            })
        else:
            error_count += 1

    print(f"🧹 불량 데이터(키값 누락/공백) 제거 건수: {error_count}건")

    # 3. 최종 저장
    os.makedirs(os.path.dirname(FINAL_PATH), exist_ok=True)
    with open(FINAL_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_valid_data, f, ensure_ascii=False, indent=4)

    print("=" * 60)
    print(f"🎉 [Phase 1 완전 종료] 최종 무결성 검증 완료!")
    print(f"💾 완벽하게 정제된 {len(final_valid_data)}개의 데이터가 저장되었습니다: {FINAL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()