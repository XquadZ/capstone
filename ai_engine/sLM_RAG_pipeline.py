import os
import torch
import gc
import numpy as np
from typing import List, Dict, Any
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def clear_vram():
    """GPU VRAM 캐시를 강제로 비워 메모리 파편화를 방지합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class LocalHoseoRAGPro:
    def __init__(self, collection_name: str = "hoseo_notices"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clear_vram()
        
        print("🤖 [1/3] 임베딩 & 리랭커 로드 중 (FP16 최적화)...")
        self.embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        print("🔌 [2/3] Milvus 벡터 DB 연결 중...")
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(collection_name)
        self.collection.load()
        
        print("🧠 [3/3] 로컬 sLM 로드 중 (NF4 Double Quantization)...")
        model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
        
        bnb_config = BitsAndBytesConfig(
         load_in_8bit=True,  # 8비트 활성화
          llm_int8_threshold=6.0, # 이상치(Outlier) 처리를 위한 정밀도 설정
          llm_int8_enable_fp32_cpu_offload=False
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 패딩 토큰 설정 (Llama-3 경고 방지)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.slm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
       quantization_config=bnb_config,
       device_map="auto",
       trust_remote_code=True
        )
        self.slm_model.eval() # 추론 모드 고정
        
        print("\n✅ Pro-Level 로컬 sLM 파이프라인 구축 완료!\n")

    def search_and_rerank(self, query_text: str, retrieve_k: int = 30, final_k: int = 3) -> List[Dict[str, Any]]:
        """하이브리드 검색 및 Cross-Encoder 리랭킹 최적화 모듈"""
        query_embeddings = self.embed_model.encode([query_text], return_dense=True, return_sparse=True)
        
        dense_req = AnnSearchRequest(
            data=[query_embeddings['dense_vecs'][0].astype(np.float32)],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 5}},
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
        
        if not initial_results[0]:
            return []

        candidates = initial_results[0]
        passages = [hit.entity.get('chunk_text') for hit in candidates]
        query_passage_pairs = [[query_text, p] for p in passages]
        
        rerank_scores = self.reranker.compute_score(query_passage_pairs)
        
        reranked_list = [
            {"score": rerank_scores[i], "entity": candidates[i].entity}
            for i in range(len(candidates))
        ]
            
        return sorted(reranked_list, key=lambda x: x['score'], reverse=True)[:final_k]

    def generate_answer(self, query_text: str) -> str:
        """Direct Model Generation 기법을 사용한 안정적인 답변 생성"""
        clear_vram()
        
        hits = self.search_and_rerank(query_text, retrieve_k=30, final_k=3)
        if not hits:
            return "관련된 공지사항 정보를 찾을 수 없습니다."

        # Reverse Repacking
        hits_reversed = hits[::-1]
        context_texts = [
            f"[문서 출처: {h['entity'].get('year', '')}년 {h['entity'].get('entity', '')}]\n{h['entity'].get('chunk_text', '')}" 
            for h in hits_reversed
        ]
        context_block = "\n\n---\n\n".join(context_texts)

        # 1. 시스템 프롬프트 및 메시지 구성
        system_prompt = """당신은 호서대학교 학생들을 돕는 전문적이고 친절한 AI 조교입니다.
[절대 규칙]
1. 반드시 아래 제공된 <공지사항 문서>만을 근거로 답변하세요.
2. 문서에 없는 내용은 절대 지어내지 말고 "해당 내용은 공지사항에서 확인할 수 없습니다"라고 답변하세요.
3. 정보가 여러 개일 경우 글머리 기호(-)를 사용하여 가독성 있게 정리하세요."""

        user_content = f"""<공지사항 문서>
{context_block}

질문: {query_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 2. 토크나이저의 Chat Template 적용 (프롬프트를 모델이 이해하는 포맷으로 완벽 변환)
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)

        # 3. 모델 직접 생성 (Direct Generation)
        with torch.no_grad():
            outputs = self.slm_model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.01,         # 환각 방지
                do_sample=False,          # Greedy Decoding으로 일관성 유지
                repetition_penalty=1.15,  # 반복 방지
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 4. 입력 프롬프트 길이를 계산하여, 순수하게 새로 생성된(답변) 부분만 슬라이싱 후 디코딩
        input_length = input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        clear_vram()
        return answer

if __name__ == "__main__":
    rag = LocalHoseoRAGPro()
    
    print("="*60)
    print("🎓 호서대 공지사항 AI 챗봇 (Local Pro Version) - 종료: q")
    print("="*60)
    
    while True:
        query = input("\n👤 학생: ")
        if query.lower() in ['q', 'exit', 'quit']:
            break
        if not query.strip(): continue
            
        print("🤖 (Local 4090) 문서를 분석하고 답변을 추론 중입니다...")
        answer = rag.generate_answer(query)
        
        print("\n" + "="*60)
        print(f"💡 AI 조교(Local):\n{answer}")
        print("="*60)