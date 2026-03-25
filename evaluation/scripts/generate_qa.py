import os
import json
import random
import time
from openai import OpenAI

# ==========================================
# ⚙️ 1. 무적의 API 돌려막기 세팅 (Failover)
# ==========================================
# 환경변수 OPENAI_API_KEYS에 "sk-A...,sk-B..." 처럼 쉼표로 여러 개 넣어도 되고, 
# 기존처럼 OPENAI_API_KEY 하나만 둬도 알아서 작동합니다.
raw_keys = os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", ""))
if not raw_keys:
    raise ValueError("❌ 환경변수 'OPENAI_API_KEY'를 찾을 수 없습니다. 시스템 설정을 확인하세요.")

API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
current_key_idx = 0

def get_client():
    """현재 활성화된 API 키로 클라이언트를 반환합니다."""
    return OpenAI(api_key=API_KEYS[current_key_idx])

def rotate_key():
    """API 제한(Rate Limit)에 걸리면 다음 키로 교체합니다."""
    global current_key_idx
    if len(API_KEYS) > 1:
        current_key_idx = (current_key_idx + 1) % len(API_KEYS)
        print(f"\n🔄 [API 스위칭] 다음 API 키로 교체하여 공격을 재개합니다. (Key Index: {current_key_idx})")
    else:
        print("\n⏳ [API 휴식] 키가 1개뿐이라 잠시 10초간 숨을 고릅니다...")
        time.sleep(10)

# ==========================================
# 🧠 2. 데이터 생성 프롬프트 (VISION 강제 할당)
# ==========================================
def generate_ragas_testset(chunks_data):
    if isinstance(chunks_data, dict):
        chunks_data = chunks_data.get("chunks", chunks_data)
        
    combined_text = "\n\n".join([f"[청크 {i+1}] {chunk.get('chunk_text', '')}" for i, chunk in enumerate(chunks_data)])
    
    system_prompt = (
        "너는 호서대학교 컴퓨터공학부에 재학 중인 20대 대학생 페르소나를 완벽하게 연기하며, "
        "RAG 시스템 평가를 위한 고품질 Q&A 데이터셋을 생성하는 전문가다."
    )
    
    user_prompt = f"""
다음 제공된 호서대학교 관련 문서를 읽고, 에브리타임(대학생 커뮤니티)이나 학과 단톡방, 조교에게 개인 카톡으로 물어볼 법한 극히 현실적인 질문과 그에 대한 모범 답안 쌍을 딱 3개만 생성해.

[제약사항]
1. 금지어: "이 공지사항에 따르면", "문서에서", "이 프로그램은", "이 행사는" 등 지시 대명사와 기계적인 표현 절대 금지.
2. ★핵심 명사(Entity) 절대 보존★: [프로그램명, 장학금명, 행사명, 부서명] 등 고유명사는 반드시 질문에 포함시켜라.
3. 페르소나 및 말투: 고유명사는 포함하되, 문장 전체는 20대 대학생처럼 친근한 구어체, 줄임말, 툭 던지는 말투를 사용해라.
4. 독립성 보장: 질문 한 줄만 뚝 떼어놓고 봐도 어떤 공지에 대해 묻는지 다른 사람이 100% 알아들을 수 있어야 한다.
5. ★정답(Ground Truth) 팩트 조작 절대 금지★: 문서의 팩트(연도, 학과 제한, 일정 등)를 훼손하지 않고 있는 그대로 대답해야 한다.
6. ★연도(Year) 필수 포함★: 질문에는 해당 문서의 기준 연도(예: "2023년", "24학년도" 등)를 반드시 포함시켜라.
7. ★질문 유형 (반드시 아래 3가지 유형을 각각 1개씩 생성할 것)★:
   - 일반 추론형: 문서의 논리나 텍스트 내용을 묻는 질문.
   - 다중 문맥형: 최소 2개 이상의 청크에 있는 텍스트 정보가 결합되어야만 대답할 수 있는 질문.
   - ★시각 구조형 (매우 중요)★: 만약 문서 내용 중에 '표(Table)', '일정표', '모집 인원 표', '포스터 내용' 등이 포함되어 있다면, 반드시 행(Row)과 열(Column)의 데이터를 교차해서 비교해야만 풀 수 있는 복잡한 질문을 만들어라.

[문서 내용]
{combined_text}

[출력 형식]
반드시 아래 JSON 포맷을 엄격하게 준수하여 출력해. (반드시 "qa_pairs"라는 단일 키를 가진 객체로 출력할 것)
{{
  "qa_pairs": [
    {{
      "question_type": "시각 구조형",
      "question": "고유명사와 '연도'가 반드시 포함된 현실적인 구어체 질문",
      "ground_truth": "문서에 근거한 명확하고 구체적인 정답",
      "context_used": "답변을 도출하는 데 사용된 청크 번호"
    }}
  ]
}}
"""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        parsed_data = json.loads(content)
        return parsed_data.get("qa_pairs", [])

    except Exception as e:
        print(f"\n❌ GPT API 호출 중 오류 발생: {e}")
        # Rate Limit(429)이나 서버 에러(50x) 발생 시 키를 교체하거나 대기
        if "429" in str(e) or "Rate limit" in str(e) or "50" in str(e):
            rotate_key()
        return []

# ==========================================
# 🚀 3. 메인 무한 루프 (이어하기 & 실시간 저장)
# ==========================================
if __name__ == "__main__":
    chunk_dir = "data/processed/chunks"
    save_dir = "evaluation/datasets"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "notice_qa_2000_target.json")
    TARGET_COUNT = 2000  

    if not os.path.exists(chunk_dir):
        print(f"❌ '{chunk_dir}' 경로를 찾을 수 없습니다.")
        exit()

    all_files = [f for f in os.listdir(chunk_dir) if f.endswith(".json")]
    if not all_files:
        print(f"❌ '{chunk_dir}' 경로에 JSON 파일이 없습니다.")
        exit()

    # 🌟 [좀비 로직 1] 기존 파일 이어하기
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                final_dataset = json.load(f)
            print(f"♻️  [이어하기] 기존 파일을 찾았습니다! {len(final_dataset)}개부터 다시 달립니다.")
        except json.JSONDecodeError:
            print("⚠️ 기존 파일이 손상되어 0개부터 새로 시작합니다.")
            final_dataset = []
    else:
        final_dataset = []
        print("🚀 [새출발] 2,000개 무한 질주를 시작합니다.")

    used_files = set() 
    
    print("="*60)
    print(f"🎯 목표: {TARGET_COUNT}개 | 현재: {len(final_dataset)}개")
    print("="*60)

    # 🌟 [좀비 로직 2] 목표 달성할 때까지 무한 루프
    while len(final_dataset) < TARGET_COUNT:
        
        available_files = [f for f in all_files if f not in used_files]
        if not available_files:
            print("\n🔄 모든 문서를 1회독 완료! 다시 섞어서 2회독 들어갑니다.")
            used_files.clear()
            available_files = all_files
        
        filename = random.choice(available_files)
        used_files.add(filename)
        file_path = os.path.join(chunk_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            print(f"🔍 [{len(final_dataset)}/{TARGET_COUNT}] 분석 중: {filename[:20]}... ", end="", flush=True)
            
            qa_pairs = generate_ragas_testset(data)
            
            if qa_pairs:
                for pair in qa_pairs:
                    if len(final_dataset) < TARGET_COUNT:
                        pair["source_file"] = filename
                        final_dataset.append(pair)
                
                # 🌟 [좀비 로직 3] 실시간 저장 (컴퓨터 꺼져도 무적)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(final_dataset, f, ensure_ascii=False, indent=2)
                
                print(f"✅ {len(qa_pairs)}문제 획득!")
            else:
                print("⚠️ 획득 실패 (다음 파일로 이동)")
                
        except Exception as e:
            print(f"❌ 파일 읽기 오류: {e}")
            
        # API 과부하 방지를 위한 미세 딜레이
        time.sleep(0.3) 

    print("="*60)
    print(f"🎉 대장정 완료! 총 {len(final_dataset)}개의 무적 데이터가 '{save_path}'에 저장되었습니다.")
    print("="*60)