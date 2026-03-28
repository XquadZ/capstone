import os
from typing import Optional

import numpy as np
from openai import OpenAI
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from FlagEmbedding import BGEM3FlagModel, FlagReranker

class HoseoRAGPipeline:
    def __init__(self, collection_name="hoseo_notices"):
        # 환경변수에서 OPENAI_API_KEY 가져오기
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ 환경변수 'OPENAI_API_KEY'를 찾을 수 없습니다. 설정 확인이 필요합니다.")
        
        print("🤖 [1/3] 임베딩 & 리랭커 모델 4090 GPU에 로드 중...")
        self.embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        print("🔌 [2/3] Milvus 벡터 DB 연결 중...")
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(collection_name)
        self.collection.load()
        
        print("🧠 [3/3] GPT-4o-mini 엔진 연결 준비 완료!")
        self.llm_client = OpenAI(api_key=api_key)
        print("\n✅ RAG 파이프라인 구축 완료!\n")

    def search_and_rerank(self, query_text, retrieve_k=50, final_k=10):
        """1차 50개 넓은 검색 후 -> 2차 리랭커로 정밀 상위 10개 압축"""
        # 1. 쿼리 임베딩
        query_embeddings = self.embed_model.encode([query_text], return_dense=True, return_sparse=True)
        
        dense_req = AnnSearchRequest(
            data=[query_embeddings['dense_vecs'][0].astype(np.float32)],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=retrieve_k
        )
        sparse_req = AnnSearchRequest(
            data=[query_embeddings['lexical_weights'][0]],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=retrieve_k
        )

        # 2. Milvus 하이브리드 검색 (Top 50 추출)
        initial_results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60), 
            limit=retrieve_k,
            output_fields=["chunk_text", "year", "category", "entity", "parent_id"]
        )
        
        if not initial_results[0]:
            return []

        # 3. 리랭커 정밀 채점 (Cross-Encoder)
        candidates = initial_results[0]
        passages = [hit.entity.get('chunk_text') for hit in candidates]
        query_passage_pairs = [[query_text, p] for p in passages]
        
        rerank_scores = self.reranker.compute_score(query_passage_pairs)
        
        # 4. Hit 객체 에러 방지를 위한 새로운 딕셔너리 리스트 생성
        reranked_list = []
        for i in range(len(candidates)):
            reranked_list.append({
                "score": rerank_scores[i],
                "entity": candidates[i].entity
            })
            
        # 5. 리랭커 점수 기준으로 정렬 후 Top 10 반환
        sorted_hits = sorted(reranked_list, key=lambda x: x['score'], reverse=True)
        return sorted_hits[:final_k]

    def generate_answer(self, query_text):
        """정교한 프롬프팅과 Reverse Repacking이 적용된 GPT-4o-mini 답변 생성"""
        hits = self.search_and_rerank(query_text, retrieve_k=50, final_k=10)
        
        if not hits:
            return "관련된 공지사항을 찾을 수 없습니다."

        # 1. 문서 순위 기록 (딕셔너리 접근)
        ranked_contexts = []
        for rank, hit in enumerate(hits, start=1):
            meta = hit['entity']
            info = (
                f"[문서 중요도 순위: {rank}위]\n"
                f"- 작성연도: {meta.get('year')}년\n"
                f"- 담당부서: {meta.get('entity')}\n"
                f"- 분류: {meta.get('category')}\n"
                f"- 본문 내용:\n{meta.get('chunk_text')}"
            )
            ranked_contexts.append(info)

        # 2. 🔄 Reverse Repacking (가장 중요한 1위 문서가 맨 밑으로 오도록 배열 반전)
        ranked_contexts.reverse()
        context_block = "\n\n" + "="*40 + "\n\n".join(ranked_contexts) + "\n\n" + "="*40

        # 3. 📝 RAG 시스템 프롬프트
        system_prompt = """당신은 호서대학교 학생들의 질문에 정확히 답변하는 최고 수준의 AI 조교입니다.
아래 제공된 [검색된 공지사항 문서]들을 분석하여 사용자의 질문에 답변하세요.

[절대 준수 원칙]
1. 철저한 팩트 체크: 반드시 제공된 문서 내용에만 근거하여 답변하세요. 문서에 없는 내용은 절대 추론하거나 지어내지 마세요.
2. 정보 부재 처리: 제공된 문서에서 질문에 대한 답을 완전히 찾을 수 없다면, "제공된 공지사항 문서에서는 해당 정보를 찾을 수 없습니다."라고 명확히 안내하세요.
3. 출처 명시: 텍스트를 요약할 때, 가급적 '[0000년 OO부서 공지 기준]'과 같이 출처 맥락을 덧붙여 신뢰도를 높이세요.
4. 중요도 우선 반영: 문서는 중요도 역순(1위 문서가 가장 마지막에 위치)으로 제공됩니다. 충돌하는 정보가 있다면 최신 연도와 상위 중요도(1위 쪽에 가까운) 문서를 우선하여 답변하세요.
5. 가독성: 읽기 쉽게 굵은 글씨(**)와 글머리 기호(-, 1. 등)를 적절히 사용하세요."""

        user_prompt = f"""[검색된 공지사항 문서]
{context_block}

[사용자 질문]
{query_text}"""

        # 4. GPT-4o-mini 호출
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # ⬅️ 엔진을 mini로 변경했습니다.
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0 # 환각 방지를 위한 0.0 세팅
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ 답변 생성 중 오류가 발생했습니다: {e}"


_shared_notice_pipeline: Optional["HoseoRAGPipeline"] = None


def get_shared_notice_pipeline() -> "HoseoRAGPipeline":
    """Agentic 노드 등에서 임베딩·Milvus를 한 번만 로드하기 위한 공유 인스턴스."""
    global _shared_notice_pipeline
    if _shared_notice_pipeline is None:
        _shared_notice_pipeline = HoseoRAGPipeline()
    return _shared_notice_pipeline


if __name__ == "__main__":
    rag = HoseoRAGPipeline()
    
    print("="*60)
    print("🎓 호서대 공지사항 AI 챗봇 (GPT-4o-mini Version) - 종료: q")
    print("="*60)
    
    while True:
        query = input("\n👤 학생: ")
        if query.lower() in ['q', 'exit', 'quit']:
            break
        if not query.strip(): continue
            
        print("🤖 조교가 4090 GPU로 문서를 정밀 검색하고 GPT-4o-mini가 분석 중입니다...")
        answer = rag.generate_answer(query)
        
        print("\n" + "="*60)
        print(f"💡 AI 조교:\n{answer}")
        print("="*60)