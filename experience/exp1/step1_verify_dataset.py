import json
import os

INPUT_PATH = "evaluation/datasets/notice_qa_2000_target.json"
OUTPUT_PATH = "evaluation/datasets/notice_qa_2000_verified.json"

def verify_data():
    if not os.path.exists(INPUT_PATH):
        print("❌ 파일을 찾을 수 없습니다.")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🧐 총 {len(data)}개의 데이터를 검증합니다...")
    
    verified_list = []
    error_count = 0

    for idx, item in enumerate(data):
        # 필수 키 체크
        required_keys = ["question", "ground_truth", "question_type"]
        if all(key in item for key in required_keys) and len(item["question"]) > 5:
            verified_list.append(item)
        else:
            error_count += 1

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(verified_list, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    print(f"✅ 검증 완료!")
    print(f"📊 정상 데이터: {len(verified_list)}개")
    print(f"❌ 불량 데이터: {error_count}개 (자동 폐기)")
    print(f"💾 저장 경로: {OUTPUT_PATH}")
    print("-" * 50)

if __name__ == "__main__":
    verify_data()