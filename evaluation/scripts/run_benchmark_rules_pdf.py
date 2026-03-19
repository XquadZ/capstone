import os
import sys
import io
import json
import time
import base64
import fitz  # PyMuPDF
from openai import OpenAI
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from FlagEmbedding import BGEM3FlagModel, FlagReranker

# ==========================================
# 🛡️ 0. 터미널 인코딩 에러 방지
# ==========================================
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# ⚙️ 1. 설정 및 모델 로드
# ==========================================
SAIFEX_API_KEY = os.environ.get("SAIFEX_API_KEY")
if not SAIFEX_API_KEY:
    raise ValueError("❌ 환경변수 'SAIFEX_API_KEY'를 찾을 수 없습니다.")

client = OpenAI(
    api_key=SAIFEX_API_KEY,
    base_url="https://ahoseo.saifex.ai/v1",
    default_headers={"Content-Type": "application/json; charset=utf-8"}
)

print("🤖 [1/3] 임베딩 모델 로드 중 (BGE-M3)...")
embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print("🧠 [2/3] 리랭커 로드 중 (BGE-Reranker-v2-m3)...")
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

print("🔌 [3/3] Milvus 연결 중...")
connections.connect("default", host="localhost", port="19530")
collection = Collection("hoseo_rules_v1")
collection.load()
print("✅ 벤치마크 준비 완료!\n" + "="*50)

# PDF 원본 폴더 경로
PDF_DIR = os.path.join("data", "rules_regulations", "raw_pdfs")

# ==========================================
# 🖼️ 2. PDF 이미지 추출 함수
# ==========================================
def get_pdf_page_base64(source_name, page_num):
    # Milvus에 저장된 소스명이 .md 형태라면 .pdf로 치환
    pdf_filename = source_name.replace(".md", ".pdf")
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    
    if not os.path.exists(pdf_path):
        print(f" ⚠️ 경고: PDF 파일을 찾을 수 없습니다 ({pdf_filename})")
        return None

    try:
        doc = fitz.open(pdf_path)
        page_index = page_num - 1 
        if page_index < 0 or page_index >= len(doc):
            return None
            
        page = doc.load_page(page_index)
        mat = fitz.Matrix(2.0, 2.0) # 고화질
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("jpeg")
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
        doc.close()
        return base64_encoded
    except Exception as e:
        print(f" ⚠️ 이미지 변환 에러: {e}")
        return None

# ==========================================
# 🔍 3. 검색 및 Vision RAG 답변 생성
# ==========================================
def retrieve_and_generate_vision(query, top_k_milvus=15, final_k=3):
    # --- [A] 하이브리드 검색 및 리랭킹 ---
    query_embs = embedder.encode([query], return_dense=True, return_sparse=True)
    dense_vec = query_embs['dense_vecs'][0].tolist()
    sparse_vec = query_embs['lexical_weights'][0]
    
    req_dense = AnnSearchRequest([dense_vec], "dense_vector", {"metric_type": "IP", "params": {"ef": 64}}, limit=top_k_milvus)
    req_sparse = AnnSearchRequest([sparse_vec], "sparse_vector", {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, limit=top_k_milvus)
    
    res = collection.hybrid_search(
        [req_dense, req_sparse], rerank=RRFRanker(), limit=top_k_milvus,
        output_fields=["doc_id", "page_num", "source", "text"]
    )
    
    retrieved_chunks = [hit.entity.to_dict() for hit in res[0]]
    if not retrieved_chunks: 
        return "관련 문서를 찾을 수 없습니다.", []

    sentence_pairs = [[query, chunk['text']] for chunk in retrieved_chunks]
    rerank_scores = reranker.compute_score(sentence_pairs, normalize=True)
    
    for i in range(len(retrieved_chunks)):
        retrieved_chunks[i]['rerank_score'] = rerank_scores[i]
        
    top_chunks = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)[:final_k]
    
    # --- [B] Ragas 채점용 텍스트 Contexts 구성 ---
    # RAGAS는 이미지를 채점할 수 없으므로, 검색된 '텍스트 원본'을 채점용으로 따로 넘겨줍니다.
    contexts_for_ragas = [
        f"[출처: {chunk['source']} | 페이지: {chunk['page_num']}]\n{chunk['text']}" 
        for chunk in top_chunks
    ]

    # --- [C] Vision 이미지 구성 (중복 페이지 제거) ---
    unique_pages = set()
    base64_images = []
    
    for chunk in top_chunks:
        page_key = (chunk['source'], chunk['page_num'])
        if page_key not in unique_pages:
            img_b64 = get_pdf_page_base64(chunk['source'], chunk['page_num'])
            if img_b64:
                base64_images.append(img_b64)
                unique_pages.add(page_key)

    # --- [D] Vision LLM 호출 ---
    system_prompt = (
        "당신은 호서대학교 학칙 안내 AI입니다. "
        "제공된 여러 장의 문서 이미지를 꼼꼼히 분석하여 사용자의 질문에 답변하세요. "
        "표(Table)나 문맥이 잘린 부분도 시각적으로 정확히 파악해야 합니다."
    )
    
    # 프롬프트 메시지 조립 (질문 + 여러 장의 이미지)
    user_content = [{"type": "text", "text": query}]
    for b64_img in base64_images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "high"}
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content, contexts_for_ragas
    except Exception as e:
        print(f" ❌ Vision API 호출 오류: {e}")
        return "API 생성 오류", contexts_for_ragas


# ==========================================
# 🚀 4. 메인 벤치마크 루프
# ==========================================
if __name__ == "__main__":
    testset_path = os.path.join("evaluation", "datasets", "rules_ragas_testset.json")
    results_dir = os.path.join("evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 최종 결과물 저장 파일명 (Vision 결과)
    output_path = os.path.join(results_dir, "benchmark_rules_pdf.json")

    with open(testset_path, 'r', encoding='utf-8') as f:
        testset = json.load(f)

    print(f"🚀 Multimodal (Vision) RAG 벤치마크 시작 (총 {len(testset)}문제)")
    
    benchmark_results = []
    
    for idx, item in enumerate(testset, 1):
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        
        print(f"[{idx:03d}/{len(testset):03d}] Q: {question[:30]}... ", end="", flush=True)
        
        # 검색, 이미지 추출, 답변 생성까지 한 번에 처리
        answer, contexts = retrieve_and_generate_vision(question)
        
        # Ragas 포맷으로 저장
        result_item = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts, # Ragas 채점기가 읽을 수 있는 텍스트 형태의 Context
            "question_type": item.get("question_type", "unknown")
        }
        benchmark_results.append(result_item)
        
        # 실시간 덮어쓰기 저장 (안전장치)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
            
        print("✅ 완료")
        time.sleep(1) # API Rate Limit 방지

    print(f"🎉 멀티모달 RAG 벤치마크 완료! 결과: {output_path}")
    print("="*60)