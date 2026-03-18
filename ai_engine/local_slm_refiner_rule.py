import os
import json
import time
import ollama

# ==========================================
# ⚙️ 설정
# ==========================================
MODEL_NAME = 'exaone3.5:7.8b'
INPUT_JSON = os.path.join("data", "rules_regulations", "chunks", "all_rules_chunks.json")
OUTPUT_JSON = os.path.join("data", "rules_regulations", "chunks", "all_rules_chunks_space.json")

def refine_text_with_ollama(original_text):
    """지시 강도를 높이고 예시를 포함한 엑사원용 교정 함수"""
    if not original_text.strip():
        return original_text
        
    system_prompt = (
        "당신은 가독성이 낮은 한국어 문서를 정밀하게 교정하는 전문가입니다.\n"
        "반드시 다음 지침을 엄수하세요:\n"
        "1. 단어가 띄어쓰기 없이 붙어 있는 경우(예: '학교내에설치된') 문맥에 맞게 반드시 띄어쓰기를 적용하세요.\n"
        "2. 단어 중간에 잘못 파고든 줄바꿈(\\n)은 삭제하여 단어를 하나로 합치세요.\n"
        "3. 조항 번호(제1조), 기호(※, -, ①), 날짜 등은 절대 수정하거나 삭제하지 마세요.\n"
        "4. 인사말이나 수정했다는 설명 없이, 오직 '교정된 텍스트'만 결과로 출력하세요."
    )
    
    user_msg = (
        f"예시:\n"
        f"입력: 'CCTV는학교의안전과재산상의피해를방지하기위해'\n"
        f"출력: 'CCTV는 학교의 안전과 재산상의 피해를 방지하기 위해'\n\n"
        f"다음 텍스트를 위와 같이 교정하세요:\n\n{original_text}"
    )
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_msg}
            ],
            options={
                'temperature': 0.3,
                'top_p': 0.9,
                'num_predict': 2000
            }
        )
        refined_text = response['message']['content'].strip()
        if "```" in refined_text:
            refined_text = refined_text.replace("```text", "").replace("```", "").strip()
        return refined_text
    except Exception as e:
        print(f"\n❌ 호출 에러: {e}")
        return original_text

def process_chunks():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # 기존 결과 파일이 있으면 로드해서 이어서 진행 (중단 대비)
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
    else:
        output_data = []

    processed_count = len(output_data)
    total_chunks = len(chunks)

    print("=" * 60)
    print(f"🚀 EXAONE 실시간 텍스트 세탁 시작 (1개당 즉시 저장)")
    print(f"🤖 모델: {MODEL_NAME}")
    print(f"📊 진행 상황: {processed_count} / {total_chunks} 완료됨")
    print("=" * 60)

    start_time = time.time()
    
    # 아직 처리되지 않은 청크부터 시작
    for i in range(processed_count, total_chunks):
        chunk = chunks[i]
        doc_id = chunk.get('doc_id', 'Unknown')
        page = chunk.get('page_num', '?')
        
        print(f"\n🔄 [{i + 1}/{total_chunks}] {doc_id} (Page {page}) 교정 중...")
        
        original = chunk.get("text", "")
        refined = refine_text_with_ollama(original)
        
        # 결과 업데이트
        chunk["text"] = refined
        output_data.append(chunk)

        # ✨ 실시간 저장 (매 루프마다 파일 쓰기)
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 시각적 확인을 위한 요약 출력 (첫 30자만)
        display_original = original.replace('\n', ' ')[:35]
        display_refined = refined.replace('\n', ' ')[:35]
        print(f"  └ [전]: {display_original}...")
        print(f"  └ [후]: {display_refined}...")
        print(f"✅ 저장 완료 (누적: {len(output_data)}개)")

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 모든 교정 작업이 완료되었습니다!")
    print(f"📁 최종 파일: {OUTPUT_JSON}")
    print(f"⏰ 총 소요 시간: {elapsed_time:.1f}초")
    print("=" * 60)

if __name__ == "__main__":
    process_chunks()