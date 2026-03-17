import os
import json
import random
import time
from openai import OpenAI

# 1. 환경변수에서 OPENAI_API_KEY 가져오기
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 환경변수 'OPENAI_API_KEY'를 찾을 수 없습니다. 시스템 설정을 확인하세요.")

# OpenAI API 클라이언트 초기화
client = OpenAI(api_key=api_key)

def generate_ragas_testset(chunks_data):
    """
    여러 청크로 분할된 하나의 문서 데이터를 입력받아,
    Ragas 평가용 진화형 가상 QA 데이터셋 3개를 생성합니다.
    """
    
    # 딕셔너리 형태일 경우 'chunks' 키 내부의 리스트를 추출, 아닐 경우 그대로 사용
    if isinstance(chunks_data, dict):
        chunks_data = chunks_data.get("chunks", chunks_data)
        
    # 최대 텍스트 길이 제한 (혹시 모를 토큰 초과 방지)
    combined_text = "\n\n".join([f"[청크 {i+1}] {chunk.get('chunk_text', '')}" for i, chunk in enumerate(chunks_data)])
    
    system_prompt = (
        "너는 호서대학교 컴퓨터공학부에 재학 중인 20대 대학생 페르소나를 완벽하게 연기하며, "
        "RAG 시스템 평가를 위한 고품질 Q&A 데이터셋을 생성하는 전문가다."
    )
    
    user_prompt = f"""
다음 제공된 호서대학교 관련 문서를 읽고, 에브리타임(대학생 커뮤니티)이나 학과 단톡방, 조교에게 개인 카톡으로 물어볼 법한 극히 현실적인 질문과 그에 대한 모범 답안 쌍을 딱 3개만 생성해.

[제약사항]
1. 금지어: "이 공지사항에 따르면", "문서에서", "이 프로그램은", "이 행사는" 등 지시 대명사와 기계적인 표현 절대 금지.
2. ★핵심 명사(Entity) 절대 보존★: 질문이 아무리 짧고 구어체여도, 대체 '무엇'에 대해 묻는지 알 수 있도록 [프로그램명, 장학금명, 행사명, 부서명] 등 고유명사는 반드시 질문에 포함시켜라.
   - ❌ 나쁜 예: "이거 언제까지 신청해?", "그거 조건이 뭐야?" (주체 누락)
   - ⭕ 좋은 예: "2학기 캔두(CanDo) 마일리지 언제까지 신청해?", "연구실 안전교육 조건이 뭐야?"
3. 페르소나 및 말투: 고유명사는 포함하되, 문장 전체는 20대 대학생처럼 친근한 구어체, 줄임말, 툭 던지는 말투를 사용해라.
4. 독립성 보장: 질문 한 줄만 뚝 떼어놓고 봐도 어떤 공지에 대해 묻는지 다른 사람이 100% 알아들을 수 있어야 한다.
5. 상황 부여의 유연성: 본인이 '컴퓨터공학부 소속이며 2026년도에 재학 중'이라는 페르소나를 가지되, 모든 질문에 "나 26년도 컴공인데~"라고 기계적으로 앵무새처럼 반복하지 마라. 주로 '조건부' 질문 등 상황이 필요할 때만 자연스럽게 섞어라.
6. ★정답(Ground Truth) 팩트 조작 절대 금지★: 학생(질문자)이 2026년도나 컴공 상황을 가정하고 묻더라도, 조교(정답)는 반드시 제공된 원본 문서의 팩트(연도, 학과 제한, 일정 등)를 훼손하지 않고 있는 그대로 대답해야 한다.
   - ❌ 나쁜 예: (원본이 2024년 행사인데) "응, 2026년 신청 마감은 11월이야." (환각 발생)
   - ⭕ 좋은 예: "이 공지는 2024년 기준이라서 24년 11월에 마감됐어. 26년도 일정은 아직 안 나온 것 같아." 또는 "이건 간호학과 대상 행사라서 컴공은 신청 못 해."
7. 질문 유형 (반드시 아래 3가지 유형을 각각 1개씩 생성할 것):
   - 추론형 (Reasoning): 문서의 논리나 이유를 묻거나 결합해야 하는 질문.
   - 조건부 (Conditioning): 본인의 학과(컴공)나 특정 연도(26년) 등 특정 조건 하에서 어떻게 적용되는지 묻는 질문.
   - 다중 문맥 (Multi-context): 최소 2개 이상의 청크에 있는 정보가 결합되어야만 대답할 수 있는 질문.

[문서 내용]
{combined_text}

[출력 형식]
반드시 아래 JSON 포맷을 엄격하게 준수하여 출력해. (반드시 "qa_pairs"라는 단일 키를 가진 객체로 출력할 것)
{{
  "qa_pairs": [
    {{
      "question_type": "추론형",
      "question": "고유명사가 포함되고 지시대명사가 없는 현실적인 구어체 질문",
      "ground_truth": "문서에 근거한 명확하고 구체적인 정답",
      "context_used": "답변을 도출하는 데 사용된 청크 번호 (예: 청크 2, 청크 3)"
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
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        parsed_data = json.loads(content)
        
        return parsed_data.get("qa_pairs", [])

    except Exception as e:
        print(f"\n❌ GPT API 호출 중 오류 발생: {e}")
        return []

if __name__ == "__main__":
    chunk_dir = "data/processed/chunks"
    save_dir = "evaluation/datasets"
    os.makedirs(save_dir, exist_ok=True)
    
    # 300문제가 저장될 최종 파일명
    save_path = os.path.join(save_dir, "ragas_testset_300.json")

    # 1. 청크 디렉토리 검사 및 파일 목록 가져오기
    if not os.path.exists(chunk_dir):
        print(f"❌ '{chunk_dir}' 경로를 찾을 수 없습니다.")
        exit()

    all_files = [f for f in os.listdir(chunk_dir) if f.endswith(".json")]
    
    if not all_files:
        print(f"❌ '{chunk_dir}' 경로에 JSON 파일이 없습니다.")
        exit()

    # 2. 100개의 파일 무작위 샘플링
    target_file_count = min(100, len(all_files))
    sampled_files = random.sample(all_files, target_file_count)
    
    final_dataset = []
    
    print("="*60)
    print(f"🚀 총 {target_file_count}개의 공지사항 파일에서 3문제씩, 약 {target_file_count * 3}개의 고품질 평가셋을 생성합니다.")
    print("="*60)

    # 3. 샘플링된 파일들을 순회하며 문제 생성 (실시간 저장 적용)
    for idx, filename in enumerate(sampled_files, 1):
        file_path = os.path.join(chunk_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            print(f"[{idx:03d}/{target_file_count:03d}] 출제 중: {filename} ... ", end="", flush=True)
            
            qa_pairs = generate_ragas_testset(data)
            
            if qa_pairs:
                for pair in qa_pairs:
                    pair["source_file"] = filename  # 출처 기록
                    final_dataset.append(pair)
                
                print(f"✅ {len(qa_pairs)}문제 완료")
                
                # 🔥 [핵심] 성공할 때마다 누적된 데이터를 실시간으로 덮어쓰기 저장
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(final_dataset, f, ensure_ascii=False, indent=2)
                    
            else:
                print("❌ 실패 (빈 리스트 반환)")
                
        except Exception as e:
            print(f"❌ 파일 읽기 오류: {e}")
            
        # API Rate Limit 방지를 위한 0.5초 대기
        time.sleep(0.5)

    print("="*60)
    print(f"🎉 대량 생성 완료! 총 {len(final_dataset)}개의 가상 QA 세트가 실시간으로 '{save_path}'에 안전하게 저장되었습니다.")
    print("="*60)