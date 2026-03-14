import json
import os
from openai import OpenAI

# 1. 설정
import os
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
INPUT_PATH = "evaluation/datasets/qa_ground_truth_v1.json"
OUTPUT_PATH = "evaluation/datasets/qa_active_v1.json"

def generate_active_queries(doc_info):
    """
    기존의 easy, medium, hard 문답을 믹싱하여 
    학생들이 실제로 사용할법한 능동적 검색어 3개를 생성합니다.
    """
    prompt = f"""
    너는 대학교 커뮤니티(에브리타임 등)에서 학생들이 어떻게 검색하는지 잘 아는 전문가야.
    아래 [공지사항 요약 정보]를 바탕으로, 학생들이 실제 검색창에 입력할 법한 '리얼한 검색어' 3개를 생성해줘.

    [규칙]
    1. "무엇에 대한 건가요?" 같은 문장형보다는 키워드 중심이나 구어체 질문으로 만들 것.
    2. 학생의 상황(예: 알바 구함, 장학금 필요, 담당자 연락처 필요 등)이 투영될 것.
    3. 정답을 그대로 노출하기보다 그 정보를 얻기 위한 '의도'를 질문에 담을 것.
    4. 출력은 반드시 JSON 배열 ["질문1", "질문2", "질문3"] 형식으로만 대답할 것.

    [공지사항 요약 정보]
    - 핵심 주제: {doc_info['qas'][0]['answer']}
    - 세부 조건: {doc_info['qas'][1]['answer']}
    - 기타 정보: {doc_info['qas'][2]['answer']}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 비용 절감을 위해 mini 권장
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic search queries for university students."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        # JSON 문자열을 리스트로 파싱
        res_json = json.loads(response.choices[0].message.content)
        return res_json.get("queries") or list(res_json.values())[0]
    except Exception as e:
        print(f"Error generating queries for {doc_info['doc_id']}: {e}")
        return ["검색어 생성 실패 1", "검색어 생성 실패 2", "검색어 생성 실패 3"]

# 2. 실행 로직
def main():
    if not os.path.exists(INPUT_PATH):
        print("기존 QA 데이터셋 파일이 없습니다.")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    active_dataset = []
    
    print(f"총 {len(data)}개의 문서에 대해 능동형 쿼리 생성을 시작합니다...")

    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] doc_id: {item['doc_id']} 처리 중...")
        
        # 새로운 쿼리 생성
        new_queries = generate_active_queries(item)
        
        # 데이터 구조 재구성
        new_item = {
            "doc_id": item['doc_id'],
            "image_file": item['image_file'],
            "user_queries": new_queries,
            "original_context": {
                "topic": item['qas'][0]['answer'],
                "details": item['qas'][1]['answer'],
                "extra": item['qas'][2]['answer']
            }
        }
        active_dataset.append(new_item)

    # 3. 결과 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(active_dataset, f, ensure_ascii=False, indent=4)
    
    print(f"완료! 저장 위치: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()