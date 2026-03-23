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
# ✨ 두 개의 컨테이너 이름 지정 (두 번째 이름은 실제 DB에 맞게 수정하세요!)
COLLECTION_1_NAME = "hoseo_rules_v1" 
COLLECTION_2_NAME = "hoseo_notices" # <--- ⚠️ 여기를 두 번째 컬렉션 이름으로 변경하세요!

SAIFEX_API_KEY = os.getenv("SAIFEX_API_KEY")
if not SAIFEX_API_KEY:
    raise ValueError("❌ 환경 변수 'SAIFEX_API_KEY'가 설정되지 않았습니다.")

SAIFEX_BASE_URL = "https://ahoseo.saifex.ai/v1"
LLM_MODEL_NAME = "gpt-4o-mini" 

client = OpenAI(
    api_key=SAIFEX_API_KEY,
    base_url=SAIFEX_BASE_URL,
    default_headers={"Content-Type": "application/json; charset=utf-8"}
)

print("🤖 [1/3] 임베딩 모델 로드 중 (BGE-M3)...")
embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print("🧠 [2/3] 리랭커 모델 로드 중 (BGE-Reranker-v2-m3)...")
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

print("🔌 [3/3] Milvus DB 멀티 컨테이너 연결 중...")
connections.connect("default", host="localhost", port="19530")

# ✨ 두 컨테이너 모두 메모리에 로드
target_collections = []
for c_name in [COLLECTION_1_NAME, COLLECTION_2_NAME]:
    try:
        col = Collection(c_name)
        col.load()
        target_collections.append(col)
        print(f"   ✔️ {c_name} 로드 완료")
    except Exception as e:
        print(f"   ⚠️ {c_name} 로드 실패 (이름을 확인하세요): {e}")

print("✅ 챗봇 준비 완료!\n" + "="*50)


# ==========================================
# 🔍 2. 하이브리드 통합 검색 엔진 (Multi-Collection)
# ==========================================
def retrieve_documents(query, top_k_milvus=10, final_top_k=3):
    query_embs = embedder.encode([query], return_dense=True, return_sparse=True)
    dense_vec = query_embs['dense_vecs'][0].tolist()
    sparse_vec = query_embs['lexical_weights'][0]
    
    req_dense = AnnSearchRequest([dense_vec], "dense_vector", {"metric_type": "IP", "params": {"ef": 64}}, limit=top_k_milvus)
    req_sparse = AnnSearchRequest([sparse_vec], "sparse_vector", {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, limit=top_k_milvus)
    
    all_retrieved_chunks = []
    
    # ✨ 각 컨테이너를 순회하며 각각 하이브리드 검색 수행
    for col in target_collections:
        res = col.hybrid_search(
            [req_dense, req_sparse], rerank=RRFRanker(), limit=top_k_milvus,
            output_fields=["doc_id", "page_num", "source", "text"]
        )
        # 검색된 결과를 마스터 리스트에 병합
        all_retrieved_chunks.extend([hit.entity.to_dict() for hit in res[0]])
    
    if not all_retrieved_chunks: 
        return []
        
    # ✨ 컨테이너 A와 B에서 모인 모든 문서들을 대상으로 통합 리랭킹 진행
    sentence_pairs = [[query, chunk['text']] for chunk in all_retrieved_chunks]
    rerank_scores = reranker.compute_score(sentence_pairs, normalize=True)
    
    for i in range(len(all_retrieved_chunks)):
        all_retrieved_chunks[i]['rerank_score'] = rerank_scores[i]
        
    # 점수 기준으로 전체 정렬 후 최상위 K개 반환
    all_retrieved_chunks = sorted(all_retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)
    return all_retrieved_chunks[:final_top_k]


# ==========================================
# 🗣️ 3. LLM 답변 생성 엔진 (타자기 효과 적용)
# ==========================================
def generate_answer(query, retrieved_chunks):
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"[참고 문서 {i+1}]\n"
        context_text += f"- 출처: {chunk['source']} (페이지: {chunk['page_num']})\n"
        context_text += f"- 내용: {chunk['text']}\n\n"

    system_prompt = """당신은 호서대학교 학칙 및 규정을 안내하는 전문 AI 어시스턴트입니다.
아래 제공된 [참고 문서]만을 바탕으로 사용자의 질문에 정확하고 명확하게 답변하세요.

[답변 규칙]
1. 문서에 없는 내용은 절대 지어내지 마세요. 판단할 수 없는 경우 "제공된 규정에서는 해당 내용을 찾을 수 없습니다."라고 답변하세요.
2. 답변 시 반드시 근거가 된 [참고 문서]의 출처(파일명)와 조항 번호를 명시하세요.
3. 사용자가 읽기 쉽도록 줄바꿈과 글머리 기호를 적절히 사용하세요."""

    user_prompt = f"{context_text}\n\n[사용자 질문]\n{query}"

    print("\n🤖 답변: ", end="", flush=True)
    start_time = time.time()
    full_answer = ""
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=800,
            stream=True 
        )
        
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    text_chunk = delta.content
                    
                    # ⏱️ 한 글자씩 출력하며 인위적인 딜레이 생성
                    for char in text_chunk:
                        print(char, end="", flush=True)
                        time.sleep(0.015) 
                        
                    full_answer += text_chunk
                    
        elapsed = time.time() - start_time
        print(f"\n\n✨ 생성 완료 (소요 시간: {elapsed:.2f}초)\n")
        return full_answer
        
    except Exception as e:
        error_msg = f"❌ API 호출 중 에러가 발생했습니다: {e}"
        print(error_msg)
        return error_msg

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
                
            chunks = retrieve_documents(user_query)
            
            if not chunks:
                print("🤖 답변: 관련된 학칙을 찾을 수 없습니다.")
                continue
                
            generate_answer(user_query, chunks)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n프로그램을 강제 종료합니다.")
            break