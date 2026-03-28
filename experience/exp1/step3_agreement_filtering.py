import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# ⚙️ 1. 설정 및 API 클라이언트 (Ahoseo API)
# ==========================================
SAIFEX_API_KEY = os.getenv("SAIFEX_API_KEY")

if not SAIFEX_API_KEY:
    raise ValueError("❌ 환경변수에 'SAIFEX_API_KEY'가 없습니다. Ahoseo API 키를 세팅해주세요!")

# Ahoseo(Saifex) 서버 설정 
client = OpenAI(
    api_key=SAIFEX_API_KEY, 
    base_url="https://ahoseo.saifex.ai/v1"
)
MODEL_NAME = "gpt-4o-mini" # 

# 입력/출력 경로 설정
INPUT_PATH = "evaluation/datasets/notice_qa_2000_compared_fixed.json" 
OUTPUT_PATH = "evaluation/datasets/filtered_golden_dataset_2000.json"
SAVE_INTERVAL = 20

# ==========================================
# ⚖️ 2. 심판관 판정 로직
# ==========================================
def evaluate_and_route(question, ground_truth, text_ans, vision_ans):
    system_prompt = """당신은 하이브리드 RAG 시스템의 '데이터 검수 및 라우팅 심판관'입니다.
주어진 질문과 실제 정답(Ground Truth), 그리고 두 가지 시스템의 답변(Text RAG, Vision RAG)을 엄격하게 비교 평가하세요.

[판단 기준]
1. REJECT (폐기): 두 시스템 모두 실제 정답과 전혀 다른 오답을 냈거나, 질문의 의도를 파악하지 못한 경우.
2. TEXT: Text RAG 답변이 실제 정답과 내용이 일치하고 충분히 훌륭한 경우.
3. VISION: Text RAG는 오답을 내거나 정보를 누락했는데, Vision RAG는 시각적 정보나 전체 맥락을 통해 실제 정답을 훨씬 더 정확하게 맞춘 경우.

반드시 아래 JSON 포맷으로만 응답하세요.
{
    "route": "TEXT" | "VISION" | "REJECT",
    "reason": "판정 이유 요약"
}"""

    user_prompt = f"""[질문]: {question}
[실제 정답(Ground Truth)]: {ground_truth}
[Text RAG 답변]: {text_ans}
[Vision RAG 답변]: {vision_ans}"""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # SyntaxError가 발생했던 지점을 깨끗하게 수정했습니다.
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content.strip())
            route = result.get("route", "REJECT").upper()
            return route, result.get("reason", "")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return "REJECT", f"API Error: {e}"

# ==========================================
# 🚀 3. 메인 실행 루프
# ==========================================
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 입력 파일이 없습니다: {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    start_index = 0
    stats = {"TEXT": 0, "VISION": 0, "REJECT": 0}

    # 이어하기 로직
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                results = json.load(f)
            start_index = len(results)
            print(f"🔄 [이어하기] 기존 {start_index}개부터 재개합니다.")
            for item in results:
                if item["route"] in stats: stats[item["route"]] += 1
        except:
            results = []

    print(f"🧐 총 {len(data)}개 데이터 심사 시작...")
    pbar = tqdm(total=len(data), initial=start_index, desc="라벨링 중")

    for i in range(start_index, len(data)):
        item = data[i]
        route, reason = evaluate_and_route(
            item.get("question", ""),
            item.get("ground_truth", ""),
            item.get("text_rag_answer", ""),
            item.get("vision_rag_answer", "")
        )

        item["route"] = route
        item["judge_reason"] = reason
        results.append(item)
        stats[route] += 1
        pbar.update(1)

        if len(results) % SAVE_INTERVAL == 0:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Ahoseo 서버 안정성을 위해 0.3초 매너 타임
        time.sleep(0.3)

    # 최종 결과 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 🌟 순수 골든 데이터셋(TEXT/VISION)만 따로 저장
    pure_data = [item for item in results if item["route"] in ["TEXT", "VISION"]]
    pure_path = OUTPUT_PATH.replace(".json", "_pure.json")
    with open(pure_path, 'w', encoding='utf-8') as f:
        json.dump(pure_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료! TEXT: {stats['TEXT']}, VISION: {stats['VISION']}, REJECT: {stats['REJECT']}")
    print(f"💎 최종 황금 데이터셋: {pure_path}")

if __name__ == "__main__":
    main()