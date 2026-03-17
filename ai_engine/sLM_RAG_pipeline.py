import os
import torch
import numpy as np
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class HoseoRAGPipelineSLM:
    def __init__(self, collection_name="hoseo_notices"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("🤖 [1/3] 임베딩 & 리랭커 모델 로드 중...")
        self.embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        print("🔌 [2/3] Milvus 벡터 DB 연결 중...")
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(collection_name)
        self.collection.load()
        
        print("🧠 [3/3] 로컬 sLM (Llama-3-Bllossom-8B) 로드 중...")
        # 한국어 튜닝이 매우 잘 되어 있고 접근이 쉬운 모델로 변경
        model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.slm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True  # 4090 VRAM 절약을 위한 4비트 양자화
        )
        
        # 텍스트 생성 파이프라인 설정
        self.slm_pipeline = pipeline(
            "text-generation",
            model=self.slm_model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.01, # 0에 가깝게 설정하여 일관성 유지
            top_p=0.9,
            repetition_penalty=1.1 # 같은 말 반복 방지
        )
        print("\n✅ 로컬 sLM RAG 파이프라인 구축 완료!\n")

    def search_and_rerank(self, query_text, retrieve_k=50, final_k=5):
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

        initial_results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60), 
            limit=retrieve_k,
            output_fields=["chunk_text", "year", "category", "entity"]
        )
        
        if not initial_results[0]: return []

        candidates = initial_results[0]
        passages = [hit.entity.get('chunk_text') for hit in candidates]
        query_passage_pairs = [[query_text, p] for p in passages]
        
        rerank_scores = self.reranker.compute_score(query_passage_pairs)
        
        reranked_list = []
        for i in range(len(candidates)):
            reranked_list.append({
                "score": rerank_scores[i],
                "entity": candidates[i].entity
            })
            
        return sorted(reranked_list, key=lambda x: x['score'], reverse=True)[:final_k]

    def generate_answer(self, query_text):
        hits = self.search_and_rerank(query_text, retrieve_k=50, final_k=5)
        if not hits: return "관련 정보를 찾을 수 없습니다."

        # Reverse Repacking
        hits_reversed = hits[::-1]
        context_texts = []
        for hit in hits_reversed:
            meta = hit['entity']
            context_texts.append(f"### [공지사항 출처: {meta.get('year')} {meta.get('entity')}]\n{meta.get('chunk_text')}")
            
        context_block = "\n\n".join(context_texts)

        # Bllossom(Llama-3) 포맷에 맞춘 프롬프트
        instruction = f"""제공된 공지사항들을 참고하여 사용자의 질문에 한국어로 친절하게 대답하세요. 
문서에 명시된 내용만 사용하고, 확실하지 않은 정보는 모른다고 답하세요."""

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

다음은 호서대학교 공지사항 내용입니다:
{context_block}

질문: {query_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        outputs = self.slm_pipeline(prompt)
        # 생성된 텍스트에서 프롬프트 이후의 답변만 추출
        full_text = outputs[0]['generated_text']
        answer = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return answer

if __name__ == "__main__":
    rag = HoseoRAGPipelineSLM()
    
    print("="*60)
    print("🎓 호서대 공지사항 AI 챗봇 (Local sLM Version) - 종료: q")
    print("="*60)
    
    while True:
        query = input("\n👤 학생: ")
        if query.lower() in ['q', 'exit', 'quit']: break
        if not query.strip(): continue
            
        print("🤖 4090 GPU가 로컬 모델로 답변을 생성 중입니다...")
        answer = rag.generate_answer(query)
        
        print("\n" + "="*60)
        print(f"💡 AI 조교(Local):\n{answer}")
        print("="*60)