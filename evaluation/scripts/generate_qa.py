import os
import random
import base64
import json
import time
from openai import OpenAI

client = OpenAI()

INPUT_DIR = "data/byaldi_input"
OUTPUT_DIR = "evaluation/datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qa_ground_truth_v1.json")
TARGET_COUNT = 100  # 목표 개수

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_available_images(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def generate_qa_for_image(image_filename):
    image_path = os.path.join(INPUT_DIR, image_filename)
    base64_image = encode_image(image_path)
    
    # 💡 요청하신 프롬프트 원문 그대로 적용!
    prompt = """
    당신은 대학교 공지사항 이미지를 분석하여 검색 시스템 평가용(QA) 데이터셋을 만드는 AI입니다.
    제공된 이미지를 분석하여, 대학생이 실제로 검색창에 입력할 법한 자연스러운 질문과 그에 대한 정답을 작성해주세요.
    반드시 아래의 난이도 기준을 따라 3개의 질문(easy, medium, hard)을 생성해야 합니다.

    1. easy: 포스터의 큰 제목이나 핵심 주제 (예: 무엇에 대한 공지인가요?)
    2. medium: 본문 텍스트 내의 구체적 정보 (예: 모집 기간, 지원 대상, 신청 장소 등)
    3. hard: 구석의 작은 글씨, 문의처 번호, 비고 사항, 혹은 QR코드의 목적 등 세밀한 관찰이 필요한 정보

    결과는 반드시 아래 JSON 구조로 출력하세요:
    {
        "qa_list": [
            {"difficulty": "easy", "question": "...", "answer": "..."},
            {"difficulty": "medium", "question": "...", "answer": "..."},
            {"difficulty": "hard", "question": "...", "answer": "..."}
        ]
    }
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ API 에러 발생 ({image_filename}): {e}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 100개가 채워질 때까지 무한 루프
    while True:
        # 1. 파일에서 기존 성공 데이터 읽어오기 (이어하기)
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                try:
                    final_dataset = json.load(f)
                except:
                    final_dataset = []
        else:
            final_dataset = []

        current_success_count = len(final_dataset)
        print(f"\n📊 현재 성공 개수: {current_success_count} / {TARGET_COUNT}")

        # 100개 도달 시 루프 완전 종료
        if current_success_count >= TARGET_COUNT:
            print("🎉 목표한 100개 QA 세트가 모두 완성되었습니다!")
            break

        # 2. 앞 5자리 고유번호 기준 중복 방지 로직
        used_prefixes = {item['doc_id'] for item in final_dataset}
        all_images = get_available_images(INPUT_DIR)
        
        # 아직 사용되지 않은 '새로운 5자리 번호'를 가진 이미지만 남김
        remaining_images = []
        for img in all_images:
            prefix = img[:5]
            if prefix.isdigit() and prefix not in used_prefixes:
                remaining_images.append(img)

        if not remaining_images:
            print("❌ 더 이상 처리할 고유 번호 이미지가 없습니다.")
            break

        # 3. 남은 후보군 중 완전히 랜덤하게 1개 뽑기
        target_img = random.choice(remaining_images)
        print(f"📸 분석 시도 중: {target_img}")
        
        result = generate_qa_for_image(target_img)
        
        # 4. 결과 저장 또는 실패 처리
        if result and "qa_list" in result:
            final_dataset.append({
                "doc_id": target_img[:5],
                "image_file": target_img,
                "qas": result["qa_list"]
            })
            
            # 성공 즉시 파일 덮어쓰기 (강제 종료 대비)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_dataset, f, ensure_ascii=False, indent=4)
            
            print(f"✅ 성공! ({len(final_dataset)}/{TARGET_COUNT})")
            
            # 🚀 한 번 성공할 때마다 여유 있게 3초 휴식
            time.sleep(3) 
        else:
            print(f"❌ 실패. 카운트하지 않고 다른 문서를 골라 다시 시도합니다.")
            
            # Rate Limit 등 API 에러 시 안전하게 5초 쉬고 루프 재시작
            time.sleep(5) 

    print(f"🏁 작업 종료. 확인 경로: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()