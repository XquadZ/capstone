import sys
import os
from typing import Dict

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from AgenticRAG.graph.state import AgentState

from ai_engine.rag_pipeline_rules import embedder, reranker, generate_answer
from pymilvus import Collection, AnnSearchRequest, RRFRanker

# ==========================================
# 1. 두 개의 Milvus 컬렉션 로드
# ==========================================
print("🔌 [통합 검색] 학칙 및 공지사항 컬렉션 로드 중...")
collection_rules = Collection("hoseo_rules_v1")
collection_rules.load()

try:
    collection_notices = Collection("hoseo_notices")
    collection_notices.load()
    has_notices = True
except Exception as e:
    print(f"⚠️ 'hoseo_notices' 컬렉션을 찾을 수 없습니다. ({e})")
    has_notices = False

# ==========================================
# 2. 통합 하이브리드 검색 엔진
# ==========================================
def retrieve_unified_documents(query, top_k_milvus=10, final_top_k=3):
    query_embs = embedder.encode([query], return_dense=True, return_sparse=True)
    dense_vec = query_embs['dense_vecs'][0].tolist()
    sparse_vec = query_embs['lexical_weights'][0]
    
    req_dense = AnnSearchRequest([dense_vec], "dense_vector", {"metric_type": "IP", "params": {"ef": 64}}, limit=top_k_milvus)
    req_sparse = AnnSearchRequest([sparse_vec], "sparse_vector", {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, limit=top_k_milvus)
    
    combined_chunks = []
    
    # 2-1. 학칙 DB 검색 (필드: doc_id, page_num, source, text)
    try:
        res_rules = collection_rules.hybrid_search(
            [req_dense, req_sparse], rerank=RRFRanker(), limit=top_k_milvus,
            output_fields=["doc_id", "page_num", "source", "text"]
        )
        for hit in res_rules[0]:
            data = hit.entity.to_dict()
            data['db_type'] = 'rule'
            combined_chunks.append(data)
    except Exception as e:
        print(f"⚠️ 학칙 DB 검색 에러: {e}")
    
    # 2-2. 공지사항 DB 검색 (필드: chunk_text, category, year) - 진단 결과 반영!
    if has_notices:
        try:
            res_notices = collection_notices.hybrid_search(
                [req_dense, req_sparse], rerank=RRFRanker(), limit=top_k_milvus,
                output_fields=["chunk_text", "category", "year"] 
            )
            for hit in res_notices[0]:
                raw_data = hit.entity.to_dict()
                
                # 파이프라인 표준 포맷(text, source, page_num)으로 이름 강제 변환
                standard_data = {
                    'text': raw_data.get('chunk_text', ''),
                    'source': raw_data.get('category', '공지사항'),
                    'page_num': raw_data.get('year', '-'),
                    'doc_id': str(hit.id), # pk 값
                    'db_type': 'notice'
                }
                combined_chunks.append(standard_data)
        except Exception as e:
            print(f"⚠️ 공지사항 DB 검색 에러: {e}")

    if not combined_chunks:
        return []
        
    # 2-3. BGE-Reranker로 통합 리랭킹
    sentence_pairs = [[query, chunk['text']] for chunk in combined_chunks]
    rerank_scores = reranker.compute_score(sentence_pairs, normalize=True)
    
    for i in range(len(combined_chunks)):
        combined_chunks[i]['rerank_score'] = rerank_scores[i]
        
    combined_chunks = sorted(combined_chunks, key=lambda x: x['rerank_score'], reverse=True)
    return combined_chunks[:final_top_k]

# ==========================================
# 3. LangGraph Text RAG 노드
# ==========================================
def text_rag_node(state: AgentState) -> Dict:
    question = state["question"]
    chunks = retrieve_unified_documents(question)
    
    if not chunks:
        return {"generation": "제공된 규정 및 공지에서는 해당 내용을 찾을 수 없습니다.", "context": "검색 결과 없음"}
        
    context_text = ""
    for i, chunk in enumerate(chunks):
        source = chunk.get('source', '알수없음')
        page = chunk.get('page_num', '?')
        db_type = "학칙" if chunk.get('db_type') == 'rule' else "공지"
        context_text += f"[문서 {i+1} | {db_type}] {source} (p.{page})\n"
        
    final_answer = generate_answer(question, chunks)
    return {"generation": final_answer, "context": context_text}