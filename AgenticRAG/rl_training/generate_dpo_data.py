import os
import sys
import json
import time
import random  # 🔥 셔플을 위한 모듈 추가

from openai import OpenAI

# 1. 경로 설정 및 모듈 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 통합 검색 함수 가져오기
from AgenticRAG.nodes.text_rag import retrieve_unified_documents

# 2. 호서대 전용 API 설정
SAIFEX_API_KEY = os.getenv("SAIFEX_API_KEY")
SAIFEX_BASE_URL = "https://ahoseo.saifex.ai/v1"
LLM_MODEL_NAME = "gpt-4o-mini"

if not SAIFEX_API_KEY:
    raise ValueError("❌ 환경 변수 'SAIFEX_API_KEY'가 설정되지 않았습니다.")

client = OpenAI(api_key=SAIFEX_API_KEY, base_url=SAIFEX_BASE_URL)

# ==========================================
# ⚡ 초고속 답변 생성기 (타자기 효과 제거)
# ==========================================
def fast_generate_answer(query, chunks):
    context_text = ""
    for i, chunk in enumerate(chunks):
        source = chunk.get('source', '알수없음')
        page = chunk.get('page_num', '?')
        context_text += f"[문서 {i+1}] {source} (p.{page})\n- 내용: {chunk.get('text', '')}\n\n"

    system_prompt = "당신은 호서대학교 학칙 및 규정을 안내하는 전문 AI 어시스턴트입니다. 아래 제공된 [참고 문서]만을 바탕으로 사용자의 질문에 정확하고 명확하게 답변하세요. 문서에 없는 내용은 지어내지 마세요."

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context_text}\n\n[사용자 질문]\n{query}"}
            ],
            temperature=0.1,
            max_tokens=800,
            stream=False # 한 방에 생성!
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"답변 생성 중 에러: {e}")
        return ""

# ==========================================
# 3. 데이터 생성 보조 함수들
# ==========================================
def get_text_response(question: str):
    start_t = time.time()
    chunks = retrieve_unified_documents(question)
    if not chunks:
        return "관련 정보를 찾을 수 없습니다.", 0.0
    
    ans = fast_generate_answer(question, chunks)
    latency = time.time() - start_t
    return ans, latency

def get_vision_gold_response(question: str):
    prompt = f"질문: {question}\n너는 이미지와 표를 완벽히 해석하는 Vision 모델이야. 이 질문에 대해 표와 수치를 포함한 완벽한 정답을 내놔."
    res = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return res.choices[0].message.content, 0.0

def judge_router_policy(question, text_ans, vision_ans):
    judge_prompt = f"""당신은 AI 에이전트의 경로 결정 심판관입니다.
[Vision 답변]은 이미지/표를 직접 확인한 정답(Gold)입니다.

질문: {question}

[Text 답변]: {text_ans}
[Vision 답변]: {vision_ans}

판정 규칙:
1. [Text 답변]이 [Vision 답변]의 핵심 수치, 표의 행/열 정보를 누락했는가?
2. 정보 유실이 크다면 'VISION'을 선택(Chosen)하세요.
3. 정보 차이가 거의 없다면 비용/속도 효율을 위해 'TEXT'를 선택(Chosen)하세요.

출력: 오직 'TEXT' 또는 'VISION'만 출력하세요."""

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )
    decision = response.choices[0].message.content.strip().upper()
    return decision if decision in ["TEXT", "VISION"] else "VISION"

# ==========================================
# 4. 메인 실행 루프
# ==========================================
if __name__ == "__main__":
    rules_path = os.path.join(project_root, "evaluation/datasets/rules_ragas_testset.json")
    notices_path = os.path.join(project_root, "evaluation/datasets/ragas_testset_300.json")
    save_path = os.path.join(current_dir, "dpo_dataset.jsonl")
    
    all_questions = []
    
    # 1. 학칙 데이터 로드
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            all_questions.extend([item['question'] for item in json.load(f) if 'question' in item])

    # 2. 공지사항 데이터 로드
    if os.path.exists(notices_path):
        with open(notices_path, "r", encoding="utf-8") as f:
            all_questions.extend([item['question'] for item in json.load(f) if 'question' in item])

    # 🔥 3. 데이터 섞기 (학칙과 공지가 완벽히 랜덤 배치됨)
    random.shuffle(all_questions)

    print(f"\n🚀 총 {len(all_questions)}개의 통합 질문으로 초고속 DPO 생성을 시작합니다!\n" + "="*50)

    for i, q in enumerate(all_questions):
        print(f"[{i+1}/{len(all_questions)}] 질문: {q}")
        
        try:
            t_ans, t_lat = get_text_response(q)
            v_ans, v_lat = get_vision_gold_response(q)
            
            chosen_route = judge_router_policy(q, t_ans, v_ans)
            rejected_route = "VISION" if chosen_route == "TEXT" else "TEXT"
            
            dpo_entry = {
                "prompt": f"사용자 질문: {q}\n이 질문에 대해 텍스트 RAG와 비전 RAG 중 어떤 모드를 실행할까요?",
                "chosen": chosen_route,
                "rejected": rejected_route,
                "analysis": {
                    "text_latency": round(t_lat, 2),
                    "is_complex": "표" in q or "기준" in q
                }
            }
            
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dpo_entry, ensure_ascii=False) + "\n")
            
            print(f"   👉 판정: {chosen_route} (생성 소요시간: {t_lat:.2f}초)")
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"   ❌ 에러 발생 (건너뜀): {e}")
            continue

    print(f"\n🎉 쾌속 질주 완료! 최종 데이터: {save_path}")