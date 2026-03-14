import json
import os
import re
import gzip
from byaldi import RAGMultiModalModel

INDEX_PATH = "hoseo_vision_index"
EVAL_DATA_PATH = "evaluation/datasets/qa_active_v1.json"
# Byaldi가 숨겨둔 원본 파일명 번역기 경로
MAPPING_PATH = ".byaldi/hoseo_vision_index/doc_ids_to_file_names.json.gz" 

def extract_prefix(filename):
    """문자열에서 앞의 5자리 숫자만 정밀하게 잘라냅니다."""
    match = re.search(r'\d{5}', str(filename))
    if match:
        return match.group(0)
    return str(filename)

def evaluate():
    print("검색 엔진 로드 중...")
    RAG = RAGMultiModalModel.from_index(INDEX_PATH)
    
    # 1. 파일명 번역기 로드
    print("원본 파일명 매핑 데이터 로드 중...")
    with gzip.open(MAPPING_PATH, 'rt', encoding='utf-8') as f:
        id_to_filename = json.load(f)
        
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    metrics = {
        "hit_at_1": 0, "hit_at_3": 0, "mrr": 0, "total_queries": 0
    }

    print("평가를 시작합니다...")

    for item in eval_data:
        target_id = str(item['doc_id'])[:5] 
        
        for query in item['user_queries']:
            metrics["total_queries"] += 1
            
            results = RAG.search(query, k=5)
            
            # 2. 내부 번호(r.doc_id)를 원본 파일명으로 번역
            translated_filenames = [id_to_filename.get(str(r.doc_id), "Unknown") for r in results]
            
            # 3. 원본 파일명에서 앞 5자리 숫자만 추출
            retrieved_prefixes = [extract_prefix(fname) for fname in translated_filenames]
            
            # 🚨 [디버깅] 첫 번째 질문일 때만 터미널에 구조 출력
            if metrics["total_queries"] == 1:
                print("\n" + "-"*50)
                print(f"📝 1번 질문: {query}")
                print(f"🎯 찾는 정답: {target_id}")
                print(f"🔍 Byaldi 내부 번호: {[r.doc_id for r in results]}")
                print(f"🔄 번역된 파일명: {translated_filenames}")
                print(f"✂️ 5자리 자른 결과: {retrieved_prefixes}")
                print("-" * 50 + "\n")
            
            # 4. 채점
            if target_id in retrieved_prefixes:
                rank = retrieved_prefixes.index(target_id) + 1
                
                if rank == 1: metrics["hit_at_1"] += 1
                if rank <= 3: metrics["hit_at_3"] += 1
                metrics["mrr"] += (1.0 / rank)
            
            if metrics["total_queries"] % 100 == 0:
                print(f"진행률: {metrics['total_queries']}개 질문 채점 완료...")

    total = metrics["total_queries"]
    if total == 0: return
        
    print("\n" + "="*40)
    print("📊 RAG 최종 평가 결과 (매핑 완벽 적용판)")
    print("="*40)
    print(f"전체 질문 수: {total}")
    print(f"Hit Rate @ 1: {metrics['hit_at_1'] / total:.4f}  ({metrics['hit_at_1']}개)")
    print(f"Hit Rate @ 3: {metrics['hit_at_3'] / total:.4f}  ({metrics['hit_at_3']}개)")
    print(f"MRR (평균 순위 점수): {metrics['mrr'] / total:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()