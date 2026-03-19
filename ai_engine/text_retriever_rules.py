import os
import sys
import io
import time
from openai import OpenAI
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from FlagEmbedding import BGEM3FlagModel, FlagReranker

# ==========================================
# 🛡️ 0. 윈도우 터미널 인코딩 에러(ascii) 방지
# ==========================================
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==========================================
# ⚙️ 1. 설정 및 모델 로드
# ==========================================
COLLECTION_NAME = "hoseo_rules_v1"

# 환경 변수에서 SAIFEX API 키 불러오기
SAIFEX_API_KEY = os.getenv("SAIFEX_API_KEY")
if not SAIFEX_API_KEY:
    raise ValueError("❌ 환경 변수 'SAIFEX_API_KEY'가 설정되지 않았습니다.")

# API 엔드포인트 및 모델명 (SAIFE X API 규격에 맞게 수정 필요 시 변경)
SAIFEX_BASE_URL = "https://ahoseo.saifex.ai/v1"
LLM_MODEL_NAME = "gpt-4o-mini" 

# OpenAI 호환 클라이언트 생성 (인코딩 헤더 추가)
client = OpenAI(
    api_key=SAIFEX_API_KEY,
    base_url=SAIFEX_BASE_URL,
    default_headers={"Content-Type": "application/json; charset=utf-8"}
)

print("🤖 [1/3] 임베딩 모델 로드 중 (BGE-M3)...")
embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print("🧠 [2/3] 리랭커 모델 로드 중 (BGE-Reranker-v2-m3)...")
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

print("🔌 [3/3] Milvus DB 연결 중...")
connections.connect("default", host="localhost", port="19530")
collection = Collection(COLLECTION_NAME)
collection.load()
print("✅ 챗봇 준비 완료!\n" + "="*50)


# ==========================================
# 🔍 2. 하이브리드 검색 엔진
# ==========================================
def retrieve_documents(query, top_k_milvus=10, final_top_k=3):
    # 쿼리 임베딩 (Dense + Sparse)
    query_embs = embedder.encode([query], return_dense=True, return_sparse=True)
    dense_vec = query_embs['dense_vecs'][0].tolist()
    sparse_vec = query_embs['lexical_weights'][0]
    
    # 검색 파라미터 설정
    req_dense = AnnSearchRequest([dense_vec], "dense_vector", {"metric_type": "IP", "params": {"ef": 64}}, limit=top_k_milvus)
    req_sparse = AnnSearchRequest([sparse_vec], "sparse_vector", {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, limit=top_k_milvus)
    
    # 하이브리드 검색 (RRF 융합)
    res = collection.hybrid_search(
        [req_dense, req_sparse], rerank=RRFRanker(), limit=top_k_milvus,
        output_fields=["doc_id", "page_num", "source", "text"]
    )
    
    retrieved_chunks = [hit.entity.to_dict() for hit in res[0]]
    if not retrieved_chunks: 
        return []
        
    # Cross-Encoder 리랭킹
    sentence_pairs = [[query, chunk['text']] for chunk in retrieved_chunks]
    rerank_scores = reranker.compute_score(sentence_pairs, normalize=True)
    
    for i in range(len(retrieved_chunks)):
        retrieved_chunks[i]['rerank_score'] = rerank_scores[i]
        
    # 점수순 정렬 후 최종 Top K 반환
    retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)
    return retrieved_chunks[:final_top_k]


# ==========================================
# 🗣️ 3. LLM 답변 생성 엔진
# ==========================================
def generate_answer(query, retrieved_chunks):
    # 검색된 문서를 컨텍스트로 구성
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"[참고 문서 {i+1}]\n"
        context_text += f"- 출처: {chunk['source']} (페이지: {chunk['page_num']})\n"
        context_text += f"- 내용: {chunk['text']}\n\n"

    # 시스템 프롬프트 작성
    system_prompt = """당신은 호서대학교 학칙 및 규정을 안내하는 전문 AI 어시스턴트입니다.
아래 제공된 [참고 문서]만을 바탕으로 사용자의 질문에 정확하고 명확하게 답변하세요.

[답변 규칙]
1. 문서에 없는 내용은 절대 지어내지 마세요. 판단할 수 없는 경우 "제공된 규정에서는 해당 내용을 찾을 수 없습니다."라고 답변하세요.
2. 답변 시 반드시 근거가 된 [참고 문서]의 출처(파일명)와 조항 번호를 명시하세요.
3. 사용자가 읽기 쉽도록 줄바꿈과 글머리 기호를 적절히 사용하세요."""

    user_prompt = f"{context_text}\n\n[사용자 질문]\n{query}"

    print("\n⏳ LLM이 답변을 생성하고 있습니다...")
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        answer = response.choices[0].message.content
        elapsed = time.time() - start_time
        print(f"✨ 생성 완료 (소요 시간: {elapsed:.2f}초)\n")
        return answer
        
    except Exception as e:
        return f"❌ API 호출 중 에러가 발생했습니다: {e}"


# ==========================================
# 🚀 4. 메인 실행 루프
# ==========================================
if __name__ == "__main__":
    print("💡 [호서대 학칙 챗봇] 질문을 입력하세요. 종료하려면 'q'를 누르세요.")
    while True:
        try:
            user_query = input("\n🧑‍💻 질문: ")
            if user_query.strip().lower() == 'q':
                break
            if not user_query.strip():
                continue
                
            # 1. 문서 검색
            chunks = retrieve_documents(user_query)
            
            if not chunks:
                print("🤖 답변: 관련된 학칙을 찾을 수 없습니다.")
                continue
                
            # 2. 답변 생성 및 출력
            answer = generate_answer(user_query, chunks)
            print("🤖 답변:\n" + answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n프로그램을 강제 종료합니다.")
            break