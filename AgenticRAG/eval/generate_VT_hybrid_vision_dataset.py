import os
import sys
import json
import time
import io
import base64
import re
from tqdm import tqdm
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI

# 상위 디렉토리 경로 추가 (ai_engine 모듈 로드용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 💡 하이브리드 검색 함수 임포트
try:
    from ai_engine.rag_pipeline import retrieve_documents
except ImportError:
    print("⚠️ [경고] retrieve_documents 함수를 불러오지 못했습니다. 경로를 확인하세요.")
    def retrieve_documents(q, final_top_k=5): return []

# ==========================================
# ⚙️ 설정 및 API 클라이언트
# ==========================================
SAIFEX_API_KEY = os.getenv("SAIFEX_API_KEY")
if not SAIFEX_API_KEY:
    raise ValueError("❌ 환경 변수 'SAIFEX_API_KEY'가 설정되지 않았습니다.")

client = OpenAI(api_key=SAIFEX_API_KEY, base_url="https://ahoseo.saifex.ai/v1")
MODEL_NAME = "gpt-4o-mini" 

NOTICE_QA_PATH = "evaluation/datasets/ragas_testset_300.json"
RULES_QA_PATH = "evaluation/datasets/rules_ragas_testset.json"
OUTPUT_PATH = "evaluation/results/hybrid_vs_vision_gold_dataset.json"

TARGET_SAMPLES = 300 
MAX_DOCS_TO_VISION = 5  
MAX_PAGES_PER_FILE = 3  
SAVE_INTERVAL = 10      # 💡 10개마다 자동 저장

# ==========================================
# 🛠️ 유틸리티 함수
# ==========================================
def load_queries(file_path, limit):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("examples", data.get("data", []))
    return [item.get("question", item.get("query", "")) for item in items if item.get("question", item.get("query", ""))][:limit]

def get_notice_visual_files(notice_id):
    valid_files = []
    base_dir = f"data/raw/{notice_id}"
    
    img_dir = os.path.join(base_dir, "images")
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                valid_files.append(os.path.join(img_dir, f))
                
    att_dir = os.path.join(base_dir, "attachments")
    if os.path.exists(att_dir):
        for f in os.listdir(att_dir):
            ext = f.lower()
            if ext.endswith(('.pdf', '.jpg', '.jpeg', '.png')):
                valid_files.append(os.path.join(att_dir, f))
                
    return valid_files[:MAX_PAGES_PER_FILE]

def image_to_base64(img_or_path):
    """💡 이미지 크기 최적화 및 Base64 변환 (토큰 폭탄 방지)"""
    try:
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path)
        else:
            img = img_or_path
            
        # 가로/세로 최대 1024px로 리사이징
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # 알파 채널(투명도)이 있으면 RGB로 변환하여 에러 방지
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
            
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85) # 용량 최적화
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"\n⚠️ 이미지 변환 에러: {e}")
        return None

# ==========================================
# 🤖 답변 생성 엔진 (API 재시도 로직 포함)
# ==========================================
def generate_with_retry(messages, max_retries=3):
    """💡 API 에러 시 최대 3번 재시도하는 공통 함수"""
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=1000
            )
            return res.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⏳ API 지연/오류 발생. 3초 후 재시도합니다... ({attempt+1}/{max_retries}) | 사유: {e}")
                time.sleep(3)
            else:
                return f"API Error (Retries Failed): {e}"

def generate_text_answer(query, chunks):
    context_text = "\n".join([f"- {c.get('source', '')} (p.{c.get('page_num', 1)}): {c.get('text', '')}" for c in chunks])
    prompt = f"다음 [참고 문서]만을 바탕으로 질문에 답하세요.\n\n[참고 문서]\n{context_text}\n\n[질문] {query}"
    return generate_with_retry([{"role": "user", "content": prompt}])

def generate_vision_answer(query, chunks):
    rules_map = {}
    notice_ids = set()

    for c in chunks:
        src = c.get("source", "")
        page = int(c.get("page_num", 1))
        
        if src.endswith(".md"):
            pdf_name = os.path.basename(src).replace(".md", ".pdf")
            pdf_path = os.path.join("data/rules_regulations/raw_pdfs", pdf_name)
            
            if len(rules_map) + len(notice_ids) >= MAX_DOCS_TO_VISION and pdf_path not in rules_map:
                continue
                
            if pdf_path not in rules_map: rules_map[pdf_path] = set()
            if len(rules_map[pdf_path]) < MAX_PAGES_PER_FILE: rules_map[pdf_path].add(page)
            
        elif "[공지사항" in src:
            match = re.search(r'\]\s*(\d+)', src)
            if match:
                n_id = match.group(1)
                if len(rules_map) + len(notice_ids) >= MAX_DOCS_TO_VISION and n_id not in notice_ids:
                    continue
                notice_ids.add(n_id)

    image_contents = []
    
    # PDF 및 이미지 Base64 변환
    for pdf_path, pages in rules_map.items():
        if os.path.exists(pdf_path):
            for p_num in sorted(list(pages)):
                try:
                    pages_img = convert_from_path(pdf_path, dpi=100, first_page=p_num, last_page=p_num)
                    for img in pages_img:
                        b64_data = image_to_base64(img)
                        if b64_data: image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}", "detail": "high"}})
                except: pass

    for n_id in notice_ids:
        visual_files = get_notice_visual_files(n_id)
        for v_file in visual_files:
            if v_file.lower().endswith('.pdf'):
                try:
                    pages_img = convert_from_path(v_file, dpi=100, first_page=1, last_page=1)
                    if pages_img:
                        b64_data = image_to_base64(pages_img[0])
                        if b64_data: image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}", "detail": "high"}})
                except: pass
            else:
                b64_data = image_to_base64(v_file)
                if b64_data: image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}", "detail": "high"}})

    # 💡 HWP/DOCX만 있어서 이미지가 0장일 경우의 텍스트 Fallback 처리
    if not image_contents:
        context_text = "\n".join([f"- {c.get('source', '')}: {c.get('text', '')}" for c in chunks])
        fallback_prompt = (
            "당신은 호서대학교 학사 정보 판정관입니다. 현재 시각적 자료(PDF/이미지)가 제공되지 않았습니다.\n"
            f"대신 다음 텍스트 문서를 정밀하게 분석하여 질문에 대한 정답을 작성하세요.\n\n[문서]\n{context_text}\n\n[질문] {query}"
        )
        return generate_with_retry([{"role": "user", "content": fallback_prompt}])

    # 이미지가 있는 정상적인 Vision 처리
    prompt = (
        "당신은 호서대학교 학사 정보 및 규정 검토의 최종 판정관입니다.\n"
        "제공된 이미지(원본 문서, 표, 모집공고 포스터 등)를 정밀하게 분석하여 질문에 대한 '완벽한 정답(Ground Truth)'을 작성하세요.\n\n"
        f"[질문] {query}"
    )
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents}]
    return generate_with_retry(messages)

# ==========================================
# 🚀 메인 실행 루프 (이어하기 & 체크포인트 포함)
# ==========================================
def main():
    print("📥 1. 기존 데이터셋에서 질문 로드 중...")
    all_queries = load_queries(RULES_QA_PATH, TARGET_SAMPLES) + load_queries(NOTICE_QA_PATH, TARGET_SAMPLES)
    total_q_count = len(all_queries)
    print(f"✅ 총 {total_q_count}개의 쿼리 추출 완료!\n")

    results = []
    start_index = 0

    # 💡 이어하기(Resume) 로직: 기존 저장 파일이 있는지 확인
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                results = json.load(f)
            start_index = len(results)
            print(f"🔄 [이어하기] 기존에 저장된 데이터 {start_index}개를 발견했습니다.")
            print(f"👉 인덱스 {start_index}번 항목부터 처리를 재개합니다...\n")
        except json.JSONDecodeError:
            print("⚠️ 기존 파일이 손상되어 처음부터 다시 시작합니다.")
            results = []

    if start_index >= total_q_count:
        print("🎉 이미 600개의 데이터셋 생성이 완료되어 있습니다!")
        return

    # 🚨 아직 쫄리시면 all_queries[start_index:start_index+5] 로 미니 테스트 먼저 하세요!
    queries_to_process = all_queries[start_index:] 
    
    for i, q in enumerate(tqdm(queries_to_process, desc="Text vs Vision GT 생성 중", initial=start_index, total=total_q_count)):
        current_idx = start_index + i
        
        chunks = retrieve_documents(q, final_top_k=10) 
        if not chunks:
            continue
            
        text_ans = generate_text_answer(q, chunks)
        vision_ans = generate_vision_answer(q, chunks)
        
        results.append({
            "id": current_idx, # 추적용 ID
            "query": q,
            "hybrid_text_answer": text_ans,
            "multimodal_gold_answer": vision_ans,
            "retrieved_sources": list(set([c.get("source") for c in chunks])) 
        })
        
        time.sleep(1.0) # API Limit 방지
        
        # 💡 10개마다 실시간 체크포인트 저장
        if (current_idx + 1) % SAVE_INTERVAL == 0:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            # tqdm 위에 로그를 덮어쓰지 않게 출력
            tqdm.write(f"💾 [Checkpoint] {current_idx + 1}번째 데이터까지 안전하게 저장되었습니다.")

    # 최종 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 대장정 완료! 데이터셋이 최종 저장되었습니다: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()