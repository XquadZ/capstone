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

# 상위 디렉토리 경로 추가 (ai_engine 로드용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 💡 하이브리드 검색 엔진 임포트
try:
    from ai_engine.rag_pipeline import retrieve_documents
except ImportError:
    print("⚠️ [경고] retrieve_documents 함수를 불러오지 못했습니다. 경로를 확인하세요.")
    def retrieve_documents(q, final_top_k=5): return []

# ==========================================
# ⚙️ 1. 설정 및 듀얼 API 클라이언트 (핵심 변경점)
# ==========================================
SAIFEX_API_KEY = os.getenv("SAIFEX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 💡 비상용 OpenAI 키 로드

if not SAIFEX_API_KEY:
    raise ValueError("❌ 'SAIFEX_API_KEY'가 없습니다. 환경변수를 확인하세요.")

# 메인 클라이언트 (호서대 Saifex)
primary_client = OpenAI(
    api_key=SAIFEX_API_KEY,
    base_url="https://ahoseo.saifex.ai/v1"
)

# 비상용 클라이언트 (찐 OpenAI)
fallback_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# 현재 활성화된 클라이언트 (초기값은 Saifex)
ACTIVE_CLIENT = primary_client

MODEL_NAME = "gpt-4o-mini"

# 파일 경로 설정
INPUT_PATH = "evaluation/datasets/final_qa_testset.json"
OUTPUT_PATH = "evaluation/datasets/router_comparison_results.json"
SAVE_INTERVAL = 10 

MAX_DOCS_TO_VISION = 3
MAX_PAGES_PER_FILE = 2

# ==========================================
# 🛠️ 2. 유틸리티 함수 (이미지 변환 및 경로 추적)
# ==========================================
def get_notice_visual_files(notice_id):
    """공지사항 폴더에서 시각 자료(PDF, JPG 등) 수집"""
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
    """💡 이미지 리사이징 & Base64 변환 (토큰 폭탄 방지)"""
    try:
        img = Image.open(img_or_path) if isinstance(img_or_path, str) else img_or_path
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS) # 해상도 최적화
        if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
            
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        return None

def generate_with_retry(messages, max_retries=4):
    """💡 API 에러 시 자동 스위칭 및 재시도 로직"""
    global ACTIVE_CLIENT
    
    for attempt in range(max_retries):
        try:
            res = ACTIVE_CLIENT.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=1000
            )
            return res.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            
            # 💡 Saifex 쿼터 초과, 429 에러, 서버 다운 감지 시 Fallback 발동
            if "quota" in error_str or "429" in error_str or "insufficient" in error_str or "50" in error_str:
                if fallback_client and ACTIVE_CLIENT == primary_client:
                    print(f"\n🚨 [경고] Saifex API 쿼터 소진 감지! ({e})")
                    print("🔄 비상용 기본 OpenAI API로 즉시 전환하여 계속 진행합니다...")
                    ACTIVE_CLIENT = fallback_client
                    continue # 전환 후 카운트 차감 없이 즉시 재시도
            
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return f"API Error: {e}"

# ==========================================
# 🤖 3. 답변 생성 분기 (TEXT vs VISION)
# ==========================================
def run_text_rag(query, chunks):
    """분기 1: 순수 텍스트 기반 답변 생성"""
    if not chunks: return "검색된 관련 문서가 없습니다."
    
    context_text = "\n".join([f"- {c.get('source', '')} (p.{c.get('page_num', 1)}): {c.get('text', '')}" for c in chunks])
    prompt = f"다음 문서를 바탕으로 학생의 질문에 답변하세요.\n\n[문서]\n{context_text}\n\n[질문] {query}"
    return generate_with_retry([{"role": "user", "content": prompt}])

def run_vision_rag(query, chunks):
    """분기 2: 원본 PDF/이미지 기반 답변 생성"""
    if not chunks: return "검색된 관련 문서가 없습니다."

    rules_map = {}
    notice_ids = set()

    # 청크의 메타데이터를 기반으로 원본 파일 경로 매핑
    for c in chunks:
        src = c.get("source", "")
        page = int(c.get("page_num", 1))
        
        if src.endswith(".md"): # 학칙
            pdf_path = os.path.join("data/rules_regulations/raw_pdfs", os.path.basename(src).replace(".md", ".pdf"))
            if len(rules_map) < MAX_DOCS_TO_VISION:
                if pdf_path not in rules_map: rules_map[pdf_path] = set()
                if len(rules_map[pdf_path]) < MAX_PAGES_PER_FILE: rules_map[pdf_path].add(page)
                
        elif "[공지사항" in src: # 공지사항
            match = re.search(r'\]\s*(\d+)', src)
            if match and len(notice_ids) < MAX_DOCS_TO_VISION:
                notice_ids.add(match.group(1))

    image_contents = []
    
    # 학칙 PDF 이미지화
    for pdf_path, pages in rules_map.items():
        if os.path.exists(pdf_path):
            for p_num in sorted(list(pages)):
                try:
                    imgs = convert_from_path(pdf_path, dpi=100, first_page=p_num, last_page=p_num)
                    for img in imgs:
                        b64 = image_to_base64(img)
                        if b64: image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                except: pass

    # 공지사항 이미지화
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

    # 💡 HWP 깡통 에러 방어: 이미지가 1장도 없으면 텍스트 청크를 던져서 Fallback
    if not image_contents:
        context_text = "\n".join([f"- {c.get('source', '')}: {c.get('text', '')}" for c in chunks])
        prompt = f"시각 자료가 없습니다. 다음 텍스트 문서를 보고 답변하세요.\n\n[문서]\n{context_text}\n\n[질문] {query}"
        return generate_with_retry([{"role": "user", "content": prompt}])

    # 정상 Vision 처리
    prompt = f"제공된 원본 문서 이미지를 꼼꼼히 분석하여 학생의 질문에 답변하세요.\n\n[질문] {query}"
    return generate_with_retry([{"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents}])

# ==========================================
# 🚀 4. 메인 실행 루프 (이어하기 완벽 지원)
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

    # 💡 이어하기(Resume) 로직
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

    pbar = tqdm(total=len(qa_data), initial=start_index, desc="TEXT vs VISION 답변 생성 중")

    for i in range(start_index, len(qa_data)):
        item = qa_data[i]
        q = item.get("question", "")
        
        # 1. 공통 하이브리드 검색 (Milvus)
        chunks = retrieve_documents(q, final_top_k=7) 
        
        # 2. 두 갈래 길로 답변 생성
        text_ans = run_text_rag(q, chunks)
        vision_ans = run_vision_rag(q, chunks)
        
        # 3. 결과 합치기
        item["text_rag_answer"] = text_ans
        item["vision_rag_answer"] = vision_ans
        item["retrieved_sources"] = list(set([c.get("source") for c in chunks])) if chunks else []
        
        results.append(item)
        pbar.update(1)
        
        # 4. 실시간 저장 (10개 단위)
        if len(results) % SAVE_INTERVAL == 0:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
        time.sleep(0.5)

    pbar.close()

    # 최종 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 600개 질문에 대한 TEXT vs VISION 비교 데이터 생성 완료!\n저장 위치: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()