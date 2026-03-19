import os
import sys
import io
import json
import time
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


# ==========================================
# 🔍 2. 하이브리드 검색 & Reverse Repacking
# ==========================================
def retrieve_and_repack(query, top_k_milvus=15, final_k=3):
    # 1. 하이브리드 검색 (Dense + Sparse)
    query_embs = embedder.encode([query], return_dense=True, return_sparse=True)
    dense_vec = query_embs['dense_vecs'][0].tolist()
    sparse_vec = query_embs['lexical_weights'][0]
    
    req_dense = AnnSearchRequest([dense_vec], "dense_vector", {"metric_type": "IP", "params": {"ef": 64}}, limit=top_k_milvus)
    req_sparse = AnnSearchRequest([sparse_vec], "sparse_vector", {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, limit=top_k_milvus)
    
    res = collection.hybrid_search(
        [req_dense, req_sparse], rerank=RRFRanker(), limit=top_k_milvus,
        output_fields=["pk", "doc_id", "page_num", "source", "text"]
    )
    
    retrieved_chunks = [hit.entity.to_dict() for hit in res[0]]
    if not retrieved_chunks: return []
        
    # 2. Cross-Encoder 리랭킹
    sentence_pairs = [[query, chunk['text']] for chunk in retrieved_chunks]
    rerank_scores = reranker.compute_score(sentence_pairs, normalize=True)
    
    for i in range(len(retrieved_chunks)):
        retrieved_chunks[i]['rerank_score'] = rerank_scores[i]
        
    # 점수 내림차순 정렬 (1위가 맨 앞) 후 Top-K 추출
    top_chunks = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)[:final_k]
    
    # 3. 🔥 Reverse Repacking (논문 핵심 기술)
    # LLM이 가장 중요한 정보를 잘 기억하도록, 1위 문서가 배열의 가장 '마지막'에 오도록 순서를 뒤집음
    top_chunks.reverse() 
    
    # 💡 [논문용 확장] Small2Big 적용 지점:
    # 현재는 chunk['text']를 그대로 쓰지만, 여기서 chunk['pk']의 앞뒤(+1, -1) 데이터를 
    # Milvus에서 추가 조회(query)하여 텍스트를 확장(Big)시킬 수 있습니다.
    
    return top_chunks


# ==========================================
# 🗣️ 3. 텍스트 RAG 답변 생성
# ==========================================
def generate_text_answer(query, repacked_chunks):
    contexts_for_ragas = []
    context_text = ""
    
    # 리패킹된 순서대로 프롬프트에 조립 (가장 중요한 문서가 프롬프트 하단에 위치함)
    for i, chunk in enumerate(repacked_chunks):
        doc_info = f"[출처: {chunk['source']} | 페이지: {chunk['page_num']}]\n{chunk['text']}"
        contexts_for_ragas.append(doc_info) # Ragas 평가를 위해 원문 저장
        context_text += f"문서 {i+1}:\n{doc_info}\n\n"

    system_prompt = (
        "당신은 호서대학교 학칙 안내 AI입니다. "
        "제공된 [참고 문서]만을 바탕으로 사용자의 질문에 정확하게 답변하세요. "
        "답변에는 반드시 출처 파일명과 관련 조항을 명시해야 합니다."
    )
    
    user_prompt = f"[참고 문서]\n{context_text}\n[사용자 질문]\n{query}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content, contexts_for_ragas
    except Exception as e:
        print(f"❌ 생성 오류: {e}")
        return "API 오류 발생", contexts_for_ragas


# ==========================================
# 🚀 4. 메인 벤치마크 루프
# ==========================================
if __name__ == "__main__":
    # 파일 경로 설정 (구조도 참고)
    testset_path = os.path.join("evaluation", "datasets", "rules_ragas_testset.json")
    results_dir = os.path.join("evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "benchmark_rules_text.json")

    with open(testset_path, 'r', encoding='utf-8') as f:
        testset = json.load(f)

    print(f"🚀 Text RAG 벤치마크 시작 (총 {len(testset)}문제)")
    
    benchmark_results = []
    
    for idx, item in enumerate(testset, 1):
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        
        print(f"[{idx:03d}/{len(testset):03d}] Q: {question[:30]}...")
        
        # 1. 검색 및 리패킹
        repacked_chunks = retrieve_and_repack(question)
        
        # 2. 답변 생성
        answer, contexts = generate_text_answer(question, repacked_chunks)
        
        # 3. Ragas 포맷으로 저장
        result_item = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts, # LLM이 실제로 본 텍스트 그대로 저장 (순서 포함)
            "question_type": item.get("question_type", "unknown")
        }
        benchmark_results.append(result_item)
        
        # 실시간 덮어쓰기 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
            
        print(f" ✅ 완료")
        time.sleep(1) # API Rate Limit 방지

    print(f"🎉 텍스트 RAG 벤치마크 완료! 결과: {output_path}")