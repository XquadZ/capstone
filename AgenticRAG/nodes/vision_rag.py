import os
import base64
import subprocess
import io
import time
from openai import OpenAI
from pdf2image import convert_from_path
from AgenticRAG.graph.state import AgentState

# 1. 하이브리드 검색 모듈 임포트 (Milvus 연동)
try:
    from ai_engine.rag_pipeline_rules import retrieve_documents 
except ImportError:
    print("⚠️ [Vision] ai_engine에서 검색 함수를 불러오지 못했습니다.")
    def retrieve_documents(q, *args, **kwargs): return [] 

# 2. SAIFEX API 설정
client = OpenAI(
    api_key=os.getenv("SAIFEX_API_KEY"),
    base_url="https://ahoseo.saifex.ai/v1"
)

# Poppler 경로 (필요 시 설정)
POPPLER_PATH = None 

def resolve_file_path(doc_id: str, source: str) -> str:
    """DB 메타데이터를 기반으로 실제 물리 파일 경로를 완벽하게 추적합니다."""
    if os.path.exists(source): return source
    
    filename = os.path.basename(source)
    if filename.endswith(".md"):
        filename = filename.replace(".md", ".pdf")

    search_paths = [
        os.path.join("data", "rules_regulations", "raw_pdfs", filename), 
        os.path.join("data", "raw", str(doc_id), "attachments", filename), 
    ]
    
    for p in search_paths:
        if os.path.exists(p): return p
    
    rule_dir = os.path.join("data", "rules_regulations", "raw_pdfs")
    if os.path.exists(rule_dir):
        prefix = filename.split('.')[0]
        for f in os.listdir(rule_dir):
            if f.startswith(prefix) and f.endswith('.pdf'):
                return os.path.join(rule_dir, f)
            
    return None

def vision_rag_node(state: AgentState) -> dict:
    question = state["question"]
    start_time = time.time()
    print(f"\n--- [NODE: Vision RAG] TV-RAG 정밀 타격 모드 가동 (Sniper + Context Padding) ---")

    # STEP 1: 하이브리드 검색 (검색 범위를 k=25로 대폭 확장하여 누락 방지)
    search_results = []
    try:
        search_results = retrieve_documents(question, 25) 
    except:
        search_results = retrieve_documents(question)

    if not search_results:
        return {"generation": "관련 문서를 찾지 못했습니다.", "context": []}

    # STEP 2: 파일별 타겟 페이지 및 컨텍스트 패딩(앞뒤 페이지) 설정
    target_map = {}
    for doc in search_results:
        metadata = getattr(doc, 'metadata', doc) if hasattr(doc, 'metadata') else doc
        source = metadata.get("source", "unknown")
        doc_id = metadata.get("doc_id", "unknown")
        page_num = int(metadata.get("page_num", 1))
        
        full_path = resolve_file_path(doc_id, source)
        if full_path:
            if full_path not in target_map:
                target_map[full_path] = {"name": source, "pages": set()}
            
            # 💡 핵심: 해당 페이지와 앞뒤 페이지를 함께 수집 (조항 단절 방지)
            target_map[full_path]["pages"].add(page_num)
            if page_num > 1: target_map[full_path]["pages"].add(page_num - 1)
            target_map[full_path]["pages"].add(page_num + 1)

    if not target_map:
        print("❌ [Vision] 원본 파일을 찾지 못했습니다.")
        return {"generation": "원본 문서를 확보하지 못했습니다.", "context": []}

    # STEP 3: 타겟 페이지 정밀 추출 및 이미지 변환 (기존 로직 유지)
    image_contents = [] 
    processed_pages_log = []
    MAX_VLM_IMAGES = 48 # API 한도 50장 준수

    for f_path, info in target_map.items():
        if len(image_contents) >= MAX_VLM_IMAGES: break
        
        f_name = info["name"]
        target_pages = sorted(list(info["pages"]))
        
        print(f"📄 [핀포인트 추출] {f_name} (Pages: {target_pages}) ...", end=" ", flush=True)
        
        try:
            for p_num in target_pages:
                if len(image_contents) >= MAX_VLM_IMAGES: break
                
                pages = convert_from_path(
                    f_path, dpi=130, first_page=p_num, last_page=p_num, poppler_path=POPPLER_PATH
                )
                
                for page in pages:
                    buf = io.BytesIO()
                    page.save(buf, format='JPEG')
                    b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_data}",
                            "detail": "high"
                        }
                    })
                    processed_pages_log.append(f"{f_name} (p.{p_num})")
            print(f"✅")
            
        except Exception as e:
            print(f"❌ 오류: {e}")

    if not image_contents:
        return {"generation": "이미지 변환 실패", "context": []}

    print(f"🚀 [VLM] 정밀 타격 이미지 {len(image_contents)}장을 SAIFEX 서버로 전송합니다.")

    # STEP 4: SAIFEX VLM 추론 및 수치 정밀 추출 프롬프트 (보강됨)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            "당신은 호서대학교 학사 규정 수치 분석 전문가입니다. 제공된 이미지는 질문과 직결된 원본 규정 페이지들입니다.\n\n"
                            "### 지시 사항 (필수):\n"
                            "1. **수치 엄밀성**: '조기졸업' 관련 조항(예: 제164조)에서 평점(예: 4.3 이상)과 학점 수치를 픽셀 단위로 확인하여 정확히 기재하세요.\n"
                            "2. **학점 테이블 추출**: 졸업요구학점(124, 126, 130 등)별로 필요한 취득학점 리스트(111, 113, 115...)를 빠짐없이 나열하세요.\n"
                            "3. **예외 및 자격 상실**: 표 하단이나 각주에 있는 '매학기 평점 4.0 유지'와 같은 자격 상실 조건을 반드시 포함하세요.\n"
                            "4. **복수전공 연관성**: 이미지 내에서 복수전공 시 이수 조건이 변경되는지 확인하여 답하세요.\n"
                            "5. 답변 끝에 반드시 '📚 [분석 근거 문서]' 섹션을 만들어 파일명과 페이지를 나열하세요.\n\n"
                            f"사용자 질문: {question}"
                        )
                    },
                    *image_contents
                ]
            }],
            max_tokens=2500,
            temperature=0.0
        )
        
        generation = response.choices[0].message.content
        
        # 💡 기존 로직: 시스템 레벨 로그 결합
        source_footer = "\n\n📍 **TV-RAG Traceability (Enhanced Sniper Mode):**\n"
        source_footer += "\n".join([f"- {log}" for log in processed_pages_log])
        generation += source_footer

        elapsed = time.time() - start_time
        print(f"✅ [Vision] 분석 완료! (소요시간: {elapsed:.1f}초)")

    except Exception as e:
        print(f"❌ [VLM API Error] {e}")
        generation = "문서 시각 분석 중 오류가 발생했습니다. (이미지 개수 초과 혹은 통신 장애)"

    return {
        "generation": generation, 
        "context": processed_pages_log
    }