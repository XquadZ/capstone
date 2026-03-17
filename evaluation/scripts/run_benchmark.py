import os
import json
import time
import sys

# ==========================================
# 🔌 [필수 연결] 본인의 RAG 파이프라인 임포트
# ==========================================
# 상위 디렉토리(CAPSTONE)를 시스템 경로에 추가하여 ai_engine 모듈을 불러올 수 있게 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 작성하신 HoseoRAGPipeline 클래스 임포트
from ai_engine.rag_pipeline import HoseoRAGPipeline

# RAG 엔진 초기화 (for문 밖에서 딱 한 번만 실행!)
print("🔄 RAG 엔진을 초기화합니다. 잠시만 기다려주세요...")
rag_engine = HoseoRAGPipeline()

def get_rag_response(query):
    """
    질문을 받아 RAG 파이프라인을 실행하고, '최종 답변'과 '검색된 청크 텍스트 리스트'를 반환합니다.
    """
    # 1. 문서 검색 (Retriever) - Ragas 평가를 위해 검색된 원문(Contexts) 수집
    # rag_pipeline.py의 search_and_rerank 메서드 활용
    hits = rag_engine.search_and_rerank(query, retrieve_k=50, final_k=10)
    
    # hits 리스트에서 실제 텍스트 내용만 뽑아서 문자열 리스트로 변환
    contexts = [hit['entity'].get('chunk_text', '') for hit in hits]
    
    # 2. 답변 생성 (Generator)
    # rag_pipeline.py의 generate_answer 메서드 활용
    answer = rag_engine.generate_answer(query)
    
    return answer, contexts

# ==========================================

if __name__ == "__main__":
    # 경로 설정 (프로젝트 루트 기준)
    testset_path = "evaluation/datasets/ragas_testset_300.json"
    results_dir = "evaluation/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 저장될 결과 파일명
    output_path = os.path.join(results_dir, "benchmark_gpt4o_mini.json")

    # 1. 시험지(Testset) 로드
    if not os.path.exists(testset_path):
        print(f"❌ 시험지 파일을 찾을 수 없습니다: {testset_path}")
        exit()
        
    with open(testset_path, 'r', encoding='utf-8') as f:
        testset = json.load(f)

    print("\n" + "="*60)
    print(f"🚀 GPT-4o-mini RAG 벤치마크 시작 (총 {len(testset)}문제)")
    print("="*60)

    benchmark_results = []
    
    # 2. 질문을 순회하며 RAG 모델에 질의 (실시간 저장)
    for idx, item in enumerate(testset, 1):
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        
        print(f"[{idx:03d}/{len(testset):03d}] 질의 중: {question[:30]}...")
        
        try:
            # 연동된 함수 호출!
            answer, contexts = get_rag_response(question)
            
            # Ragas 평가 포맷에 맞게 데이터 재구성
            result_item = {
                "question": question,
                "ground_truth": ground_truth, # Ragas 채점용 모범 답안
                "answer": answer,             # RAG가 생성한 답변
                "contexts": contexts,         # RAG가 검색해온 문서 리스트 (List[str] 형태 필수)
                "question_type": item.get("question_type", "unknown"),
                "source_file": item.get("source_file", "unknown")
            }
            
            benchmark_results.append(result_item)
            
            # 실시간 저장 (Incremental Save)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
                
            print(f" ✅ 완료 (검색된 문서: {len(contexts)}개)")
            
        except Exception as e:
            print(f" ❌ 오류 발생: {e}")
            
        # API Rate Limit 또는 서버 과부하 방지를 위한 대기
        time.sleep(1)

    print("="*60)
    print(f"🎉 벤치마크 완료! 결과가 '{output_path}'에 안전하게 저장되었습니다.")
    print("="*60)