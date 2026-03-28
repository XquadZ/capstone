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

# 💡 DecompressionBombWarning 강제 해제 (고해상도 이미지 폭탄 방어)
Image.MAX_IMAGE_PIXELS = None 

# 💡 상위 디렉토리(capstone 최상위) 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 💡 찐 공지사항 전용 RAG 파이프라인 클래스 임포트
try:
    from ai_engine.rag_pipeline_notice import HoseoRAGPipeline
except ImportError as e:
    print(f"⚠️ [경고] HoseoRAGPipeline 모듈을 불러오지 못했습니다. 경로 에러: {e}")
    sys.exit(1)

# ==========================================
# ⚙️ 1. 설정 및 듀얼 API 클라이언트 (Failover)
# ==========================================
SAIFEX_API_KEY = os.getenv("SAIFEX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

primary_client = OpenAI(api_key=SAIFEX_API_KEY, base_url="https://ahoseo.saifex.ai/v1") if SAIFEX_API_KEY else None
fallback_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if not primary_client and not fallback_client:
    raise ValueError("❌ 사용 가능한 API 키가 없습니다. 환경변수를 확인하세요.")

ACTIVE_CLIENT = primary_client if primary_client else fallback_client
MODEL_NAME = "gpt-4o-mini"

# 💡 [핵심 수정] 2000개 데이터셋 경로로 업데이트!
INPUT_PATH = "evaluation/datasets/notice_qa_2000_verified.json" 
OUTPUT_PATH = "evaluation/datasets/notice_qa_2000_compared.json" 
SAVE_INTERVAL = 10 

MAX_DOCS_TO_VISION = 3
MAX_PAGES_PER_FILE = 2

# RAG 엔진 초기화 (GPU 로드)
print("\n🚀 [시스템] 공지사항 전용 RAG 엔진(Milvus & BGE-m3)을 가동합니다...")
rag_engine = HoseoRAGPipeline()

# ==========================================
# 🛠️ 2. 유틸리티 함수 (이미지 변환 & 재시도 로직)
# ==========================================
def get_notice_visual_files(notice_id):
    valid_files = []
    base_dir = f"data/raw/{notice_id}"
    for folder in ["images", "attachments"]:
        target_dir = os.path.join(base_dir, folder)
        if os.path.exists(target_dir):
            for f in os.listdir(target_dir):
                if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
                    valid_files.append(os.path.join(target_dir, f))
    return valid_files[:MAX_PAGES_PER_FILE]

def image_to_base64(img_or_path):
    try:
        img = Image.open(img_or_path) if isinstance(img_or_path, str) else img_or_path
        img.thumbnail((1536, 1536), Image.Resampling.LANCZOS) 
        if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        return None

def generate_with_retry(messages, max_retries=4):
    global ACTIVE_CLIENT
    for attempt in range(max_retries):
        try:
            res = ACTIVE_CLIENT.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=1000
            )
            return res.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str or "insufficient" in error_str or "50" in error_str:
                if fallback_client and ACTIVE_CLIENT == primary_client:
                    print(f"\n🚨 [경고] API 이슈 감지! ({e})")
                    print("🔄 비상용 기본 OpenAI API로 즉시 전환하여 계속 진행합니다...")
                    ACTIVE_CLIENT = fallback_client
                    continue 
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return f"API Error: {e}"

# ==========================================
# 🤖 3. 답변 생성 분기 (TEXT vs VISION)
# ==========================================
def run_text_rag(query, chunks):
    if not chunks: return "검색된 관련 문서가 없습니다."
    context_text = "\n".join([f"- {c.get('source', '')} (p.{c.get('page_num', 1)}): {c.get('text', '')}" for c in chunks])
    prompt = f"다음 문서를 바탕으로 학생의 질문에 답변하세요.\n\n[문서]\n{context_text}\n\n[질문] {query}"
    return generate_with_retry([{"role": "user", "content": prompt}])

def run_vision_rag(query, chunks):
    if not chunks: return "검색된 관련 문서가 없습니다."
    rules_map = {}
    notice_ids = set()

    for c in chunks:
        src = c.get("source", "")
        page = int(c.get("page_num", 1))
        
        if src.endswith(".md"): 
            pdf_path = os.path.join("data/rules_regulations/raw_pdfs", os.path.basename(src).replace(".md", ".pdf"))
            if len(rules_map) < MAX_DOCS_TO_VISION:
                if pdf_path not in rules_map: rules_map[pdf_path] = set()
                if len(rules_map[pdf_path]) < MAX_PAGES_PER_FILE: rules_map[pdf_path].add(page)
        elif "[공지사항" in src: 
            match = re.search(r'\]\s*(\d+)', src)
            if match and len(notice_ids) < MAX_DOCS_TO_VISION:
                notice_ids.add(match.group(1))

    image_contents = []
    for pdf_path, pages in rules_map.items():
        if os.path.exists(pdf_path):
            for p_num in sorted(list(pages)):
                try:
                    imgs = convert_from_path(pdf_path, dpi=100, first_page=p_num, last_page=p_num)
                    for img in imgs:
                        b64 = image_to_base64(img)
                        if b64: image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                except: pass

    for n_id in notice_ids:
        for v_file in get_notice_visual_files(n_id):
            if v_file.lower().endswith('.pdf'):
                try:
                    imgs = convert_from_path(v_file, dpi=100, first_page=1, last_page=1)
                    if imgs:
                        b64 = image_to_base64(imgs[0])
                        if b64: image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                except: pass
            else:
                b64 = image_to_base64(v_file)
                if b64: image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    if not image_contents:
        context_text = "\n".join([f"- {c.get('source', '')}: {c.get('text', '')}" for c in chunks])
        prompt = f"시각 자료가 없습니다. 다음 텍스트 문서를 보고 답변하세요.\n\n[문서]\n{context_text}\n\n[질문] {query}"
        return generate_with_retry([{"role": "user", "content": prompt}])

    prompt = f"제공된 원본 문서 이미지를 꼼꼼히 분석하여 학생의 질문에 답변하세요.\n\n[질문] {query}"
    return generate_with_retry([{"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents}])

# ==========================================
# 🚀 4. 메인 실행 루프 (2000개 돌파)
# ==========================================
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 입력 파일이 없습니다: {INPUT_PATH}")
        return
        
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        
    print(f"📥 총 {len(qa_data)}개의 질문을 로드했습니다.")

    results = []
    start_index = 0

    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                results = json.load(f)
            start_index = len(results)
            print(f"🔄 [이어하기] 기존 {start_index}개의 결과 데이터를 발견했습니다.")
        except:
            print("⚠️ 기존 파일이 손상되어 처음부터 시작합니다.")

    if start_index >= len(qa_data):
        print("🎉 이미 모든 데이터 처리가 완료되었습니다!")
        return

    pbar = tqdm(total=len(qa_data), initial=start_index, desc="TEXT vs VISION RAG 답변 생성 중")

    for i in range(start_index, len(qa_data)):
        item = qa_data[i]
        q = item.get("question", "")
        
        # 💡 [핵심] 실제 파이프라인(Milvus DB)을 태워서 검색 결과를 가져옵니다.
        raw_hits = rag_engine.search_and_rerank(q, retrieve_k=30, final_k=7)
        chunks = []
        for hit in raw_hits:
            meta = hit['entity']
            pid = meta.get('parent_id', 'unknown')
            cat = meta.get('category', '일반')
            chunks.append({
                "source": f"[공지사항-{cat}] {pid}", 
                "page_num": 1,
                "text": meta.get('chunk_text', '')
            })
        
        # 실제 답변 2개 생성
        text_ans = run_text_rag(q, chunks)
        vision_ans = run_vision_rag(q, chunks)
        
        item["text_rag_answer"] = text_ans
        item["vision_rag_answer"] = vision_ans
        item["retrieved_sources"] = list(set([c.get("source") for c in chunks])) if chunks else []
        
        results.append(item)
        pbar.update(1)
        
        # 10개 단위로 실시간 저장 (좀비 모드)
        if len(results) % SAVE_INTERVAL == 0:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
        # API 과부하 방지
        time.sleep(0.5)

    pbar.close()

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 2000개 질문에 대한 TEXT vs VISION 비교 데이터 생성 완료!\n저장 위치: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()