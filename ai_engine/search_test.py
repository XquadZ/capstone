import numpy as np
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from FlagEmbedding import BGEM3FlagModel

class HybridSearcher:
    def __init__(self, collection_name="hoseo_notices"):
        print("🤖 BGE-M3 검색 엔진 로드 중... (잠시만 기다려주세요)")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(collection_name)
        self.collection.load() # DB를 메모리에 올려서 검색 대기
        print("✅ 하이브리드 검색 준비 완료!\n")

    def search(self, query_text, top_k=3):
        # 1. 사용자 질문을 똑같은 BGE-M3 모델로 벡터화 (열쇠 깎기)
        query_embeddings = self.model.encode(
            [query_text], 
            return_dense=True, 
            return_sparse=True
        )
        
        # 2. Dense(의미) 검색 세팅
        dense_vec = query_embeddings['dense_vecs'][0].astype(np.float32)
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k
        )

        # 3. Sparse(키워드) 검색 세팅
        sparse_vec = query_embeddings['lexical_weights'][0]
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k
        )

        # 4. 하이브리드 검색 실행 (RRF 알고리즘으로 순위 융합)
        results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60), 
            limit=top_k,
            output_fields=["chunk_id", "year", "category", "entity", "chunk_text"]
        )

        return results[0]

if __name__ == "__main__":
    searcher = HybridSearcher()
    
    print("="*50)
    print("🎓 호서대 공지사항 하이브리드 검색 테스트 (종료: q)")
    print("="*50)
    
    while True:
        query = input("\n🔎 질문을 입력하세요: ")
        if query.lower() in ['q', 'exit', 'quit']:
            break
            
        if not query.strip(): continue
            
        hits = searcher.search(query, top_k=3)
        
        print(f"\n[검색 결과 Top 3]")
        for i, hit in enumerate(hits):
            meta = hit.entity
            text = meta.get('chunk_text', '').replace('\n', ' ')[:150] # 150자만 미리보기
            print(f"{i+1}. [점수: {hit.score:.4f}] {meta.get('year')}년 | {meta.get('category')} | {meta.get('entity')}")
            print(f"   내용: {text}...")