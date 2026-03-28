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

# 💡 고해상도 이미지 폭탄 방어
Image.MAX_IMAGE_PIXELS = None 

# 💡 상위 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 💡 찐 공지사항 전용 RAG 파이프라인 클래스 임포트
try:
    from ai_engine.rag_pipeline_notice import HoseoRAGPipeline
except ImportError as e:
    print(f"⚠️ [경고] HoseoRAGPipeline 모듈을 불러오지 못했습니다. 경로 에러: {e}")
    sys.exit(1)

# ==========================================
# ⚙️ 1. 설정 및 API 클라이언트
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ 환경변수에 'OPENAI_API_KEY'가 없습니다. 결제가 연동된 OpenAI API 키를 세팅해주세요!")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini" 

INPUT_PATH = "evaluation/datasets/notice_qa_2000_compared.json"
OUTPUT_PATH = "evaluation/datasets/notice_qa_2000_compared_fixed.json" 
SAVE_INTERVAL = 10 

# 🔥 형님의 요청대로 이미지 3장 유지!
MAX_DOCS_TO_VISION = 3
MAX_PAGES_PER_FILE = 2

# ==========================================
# 🛠️ 2. 유틸리티 함수 (이미지 다이어트 & 끈기 로직)
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
        # 💡 [비밀 무기] 해상도를 1024로 살짝 낮춰서 사람은 똑같이 보지만 토큰은 확 줄입니다.
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS) 
        if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        return None

def generate_with_retry(messages, max_retries=15):
    """🔥 429 에러를 만나면 토큰 양동이가 빌 때까지 60초 존버하는 끈기의 함수"""
    for attempt in range(max_retries):
        try:
            res = openai_client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=1000
            )
            return res.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            # TPM(분당 토큰) 제한에 걸렸을 때의 확실한 대처법 = 1분을 쉰다!
            if "429" in error_str or "rate limit" in error_str or "tokens per min" in error_str:
                wait_time = 60 # 1분 강제 휴식 (토큰 초기화 시간)
                print(f"\n🛑 [TPM 한도 초과] OpenAI 양동이가 꽉 찼습니다. {wait_time}초간 숨을 참고 재시도합니다... (시도 {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    print(f"\n⚠️ 기타 API 지연 감지 ({e}) -> 5초 대기 후 재시도...")
                    time.sleep(5)
                else:
                    return f"API Error: {e}"

# ==========================================
# 🤖 3. 답변 생성 분기
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
# 🚀 4. 메인 실행 루프 (수술실)
# ==========================================
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 입력 파일이 없습니다: {INPUT_PATH}")
        return
        
    load_path = OUTPUT_PATH if os.path.exists(OUTPUT_PATH) else INPUT_PATH
    with open(load_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"📥 데이터 {len(data)}개 로드 완료. (출처: {load_path})")

    error_indices = [
        i for i, item in enumerate(data) 
        if "API Error" in str(item.get("text_rag_answer", "")) or 
           "API Error" in str(item.get("vision_rag_answer", ""))
    ]

    print(f"🚨 치료가 필요한 'API Error' 환자 수: {len(error_indices)}명")

    if not error_indices:
        print("🎉 모든 데이터가 정상입니다! 에러 복구가 완료되었습니다.")
        if load_path == INPUT_PATH:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        return

    print("\n🚀 [시스템] 에러 복구를 위한 공지사항 전용 RAG 엔진을 가동합니다...")
    rag_engine = HoseoRAGPipeline()

    pbar = tqdm(total=len(error_indices), desc="OpenAI API로 에러 치료 중")

    fix_count = 0
    for idx in error_indices:
        item = data[idx]
        q = item.get("question", "")
        
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
        
        # OpenAI API로 다시 답변 받아오기
        text_ans = run_text_rag(q, chunks)
        vision_ans = run_vision_rag(q, chunks)
        
        item["text_rag_answer"] = text_ans
        item["vision_rag_answer"] = vision_ans
        item["retrieved_sources"] = list(set([c.get("source") for c in chunks])) if chunks else []
        
        data[idx] = item 
        fix_count += 1
        pbar.update(1)
        
        if fix_count % SAVE_INTERVAL == 0:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
        # 🔥 평소에도 1회 쏠 때마다 3초씩 쉬어서 429 에러 자체를 예방합니다.
        time.sleep(3.0) 

    pbar.close()

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 에러 데이터 {len(error_indices)}개 전원 치료 완료!")
    print(f"💾 완치된 데이터셋 저장 위치: {OUTPUT_PATH}")
    print("🔥 이제 Step 3(심판관 코드)로 넘어가시면 됩니다!")

if __name__ == "__main__":
    main()