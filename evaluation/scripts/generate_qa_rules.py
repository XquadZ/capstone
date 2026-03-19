import os
import sys
import io
import json
import random
import time
from openai import OpenAI

# ==========================================
# 🛡️ 0. 윈도우 터미널 인코딩 에러 방지
# ==========================================
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==========================================
# ⚙️ 1. SAIFEX API 설정
# ==========================================
SAIFEX_API_KEY = os.environ.get("SAIFEX_API_KEY")
if not SAIFEX_API_KEY:
    raise ValueError("❌ 환경변수 'SAIFEX_API_KEY'를 찾을 수 없습니다. 터미널 환경변수를 확인하세요.")

client = OpenAI(
    api_key=SAIFEX_API_KEY,
    base_url="https://ahoseo.saifex.ai/v1",
    default_headers={"Content-Type": "application/json; charset=utf-8"}
)

def generate_ragas_testset(chunk_block):
    """
    여러 개의 학칙 청크 묶음을 입력받아,
    Ragas 평가용 진화형 가상 QA 데이터셋 3개를 생성합니다.
    """
    # LLM이 참고할 텍스트 구성 (메타데이터 포함)
    combined_text = "\n\n".join([
        f"[청크 {i+1} | 출처: {chunk.get('source', '알수없음')} | 페이지: {chunk.get('page_num', 0)}] \n{chunk.get('text', '')}" 
        for i, chunk in enumerate(chunk_block)
    ])
    
    system_prompt = (
        "너는 호서대학교 컴퓨터공학부에 재학 중인 20대 대학생 페르소나를 완벽하게 연기하며, "
        "RAG 시스템 평가를 위한 고품질 Q&A 데이터셋을 생성하는 전문가다."
    )
    
    user_prompt = f"""
다음 제공된 호서대학교 학칙/규정 문서를 읽고, 에브리타임이나 학과 단톡방, 혹은 학사지원팀 조교에게 물어볼 법한 극히 현실적인 질문과 그에 대한 모범 답안 쌍을 딱 3개만 생성해.

[제약사항]
1. 금지어: "이 규정에 따르면", "문서에서", "이 조항은" 등 지시 대명사와 기계적인 표현 절대 금지.
2. ★핵심 명사(Entity) 절대 보존★: [조기졸업, 장학금, 복수전공, 출석 인정] 등 규정의 핵심 고유명사는 반드시 질문에 포함시켜라.
3. 페르소나 및 말투: 고유명사는 포함하되, 문장 전체는 20대 대학생처럼 친근한 구어체, 줄임말, 툭 던지는 말투를 사용해라.
4. 독립성 보장: 질문 한 줄만 뚝 떼어놓고 봐도 어떤 규정에 대해 묻는지 100% 알아들을 수 있어야 한다.
5. 상황 부여: 질문 시 본인이 '컴퓨터공학부 소속이며 2026년도에 재학 중'이라는 페르소나를 자연스럽게 섞어라. (모든 질문에 억지로 넣을 필요는 없음)
6. ★정답(Ground Truth) 팩트 조작 절대 금지★: 질문자가 특정 상황을 가정하더라도, 답변(Ground Truth)은 제공된 규정의 팩트(기준 학점, 기간 등)를 훼손하지 않고 있는 그대로 객관적으로 작성해야 한다.
7. 질문 유형 (반드시 아래 3가지 유형을 각각 1개씩 생성할 것):
   - 추론형 (Reasoning): 학칙의 예외 사항이나 적용 이유를 묻는 질문.
   - 조건부 (Conditioning): 본인의 학과나 학년 등 특정 조건 하에서 이 학칙이 어떻게 적용되는지 묻는 질문.
   - 다중 문맥 (Multi-context): 최소 2개 이상의 청크에 있는 정보(예: A조항과 B조항)가 결합되어야만 대답할 수 있는 질문.

[문서 내용]
{combined_text}

[출력 형식]
반드시 아래 JSON 포맷을 엄격하게 준수하여 출력해. (마크다운 백틱(```) 없이 순수 JSON만 출력할 것)
{{
  "qa_pairs": [
    {{
      "question_type": "추론형",
      "question": "고유명사가 포함되고 지시대명사가 없는 현실적인 구어체 질문",
      "ground_truth": "학칙에 근거한 명확하고 구체적인 정답 (출처 조항 포함)",
      "context_used": "답변을 도출하는 데 사용된 출처와 페이지 (예: 1-1-1. 학칙.md, 3페이지)"
    }}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"} # SAIFEX에서 json_object를 지원할 경우
        )
        
        content = response.choices[0].message.content.strip()
        
        # 간혹 마크다운 찌꺼기가 붙어오는 경우 제거
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        parsed_data = json.loads(content)
        return parsed_data.get("qa_pairs", [])

    except Exception as e:
        print(f"\n❌ GPT API 호출 중 오류 발생: {e}")
        return []

if __name__ == "__main__":
    # 데이터 경로 설정 (상황에 맞게 수정)
    input_json = os.path.join("data", "rules_regulations", "chunks", "all_rules_chunks_meta.json")
    save_dir = os.path.join("evaluation", "datasets")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "rules_ragas_testset.json")

    # 1. 단일 JSON 파일에서 청크 데이터 로드
    if not os.path.exists(input_json):
        print(f"❌ '{input_json}' 파일을 찾을 수 없습니다.")
        exit()

    with open(input_json, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
        
    print(f"✅ 총 {len(all_chunks)}개의 학칙 청크를 로드했습니다.")

    # 2. 청크를 5개 단위의 '블록(Block)'으로 묶기 (문맥 유지를 위해)
    chunk_blocks = [all_chunks[i:i + 5] for i in range(0, len(all_chunks), 5)]
    
    # 3. 100개의 블록 무작위 샘플링 (100 * 3문제 = 약 300문제 생성)
    target_block_count = min(100, len(chunk_blocks))
    sampled_blocks = random.sample(chunk_blocks, target_block_count)
    
    final_dataset = []
    
    print("="*60)
    print(f"🚀 {target_block_count}개의 학칙 블록에서 3문제씩, 약 {target_block_count * 3}개의 고품질 평가셋을 생성합니다.")
    print("="*60)

    # 4. 샘플링된 블록들을 순회하며 문제 생성
    for idx, block in enumerate(sampled_blocks, 1):
        # 주로 사용된 문서명 추출 (대표 이름 표시용)
        main_source = block[0].get("source", "unknown")
        print(f"[{idx:03d}/{target_block_count:03d}] 출제 중: {main_source} 주변 내용 ... ", end="", flush=True)
        
        qa_pairs = generate_ragas_testset(block)
        
        if qa_pairs:
            for pair in qa_pairs:
                pair["main_source"] = main_source
                final_dataset.append(pair)
            
            print(f"✅ {len(qa_pairs)}문제 완료")
            
            # 실시간 덮어쓰기 저장 (중간에 끊겨도 데이터 보존)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_dataset, f, ensure_ascii=False, indent=2)
                
        else:
            print("❌ 실패 (빈 리스트 반환)")
            
        # API Rate Limit 방지를 위한 대기 (SAIFEX 서버 상황에 따라 조절)
        time.sleep(1.0) 

    print("="*60)
    print(f"🎉 대량 생성 완료! 총 {len(final_dataset)}개의 가상 QA 세트가 '{save_path}'에 저장되었습니다.")
    print("="*60)