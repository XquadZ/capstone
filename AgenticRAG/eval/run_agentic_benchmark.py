import sys
import os
import json
import time
from tqdm import tqdm

# 루트 경로를 sys.path에 추가하여 AgenticRAG 모듈을 불러올 수 있게 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 🚀 우리가 만든 Agentic RAG 파이프라인 (LangGraph app) 불러오기
from AgenticRAG.graph.main_agent import app 

# ==========================================
# ⚙️ 설정 (경로는 프로젝트 환경에 맞게 조정하세요)
# ==========================================
NOTICE_QA_PATH = "evaluation/datasets/ragas_testset_300.json" # 기존 공지 QA 데이터
RULES_QA_PATH = "evaluation/datasets/rules_ragas_testset.json" # 기존 학칙 QA 데이터
OUTPUT_PATH = "evaluation/results/agentic_benchmark_results.json"

NUM_SAMPLES_EACH = 150 # 각각 몇 개씩 뽑을지 설정 (시간이 부족하면 10~20개로 줄여서 먼저 테스트하세요!)

def load_and_sample_data(file_path, num_samples):
    """JSON 파일에서 QA 데이터를 읽어와 지정된 개수만큼 샘플링합니다."""
    if not os.path.exists(file_path):
        print(f"⚠️ 경고: {file_path} 파일이 없습니다. 빈 리스트를 반환합니다.")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터 구조가 리스트인지, dict 안에 있는지 확인 후 추출
    items = data if isinstance(data, list) else data.get("examples", data.get("data", []))
    return items[:num_samples]

def run_benchmark():
    print("📥 1. 기존 QA 데이터셋 로드 중...")
    notice_data = load_and_sample_data(NOTICE_QA_PATH, NUM_SAMPLES_EACH)
    rules_data = load_and_sample_data(RULES_QA_PATH, NUM_SAMPLES_EACH)
    
    combined_data = notice_data + rules_data
    total_q = len(combined_data)
    print(f"✅ 총 {total_q}개의 질문 세트가 준비되었습니다. (공지: {len(notice_data)}개, 학칙: {len(rules_data)}개)\n")
    
    if total_q == 0:
        print("❌ 테스트할 데이터가 없습니다. 파일 경로를 확인해주세요.")
        return

    results = []
    
    print("🚀 2. Agentic RAG 파이프라인 벤치마크 시작 (이 작업은 시간이 다소 걸립니다...)")
    
    # tqdm으로 진행률 표시
    for i, item in enumerate(tqdm(combined_data, desc="Agentic RAG 평가 중")):
        question = item.get("question", item.get("query", ""))
        ground_truth = item.get("ground_truth", item.get("answer", "")) # 기존 정답(Golden GT)
        
        if not question:
            continue
            
        initial_state = {
            "question": question,
            "retry_count": 0
        }
        
        try:
            # 💡 Agentic RAG 실행 (stream 대신 invoke를 써서 최종 상태만 바로 받음)
            final_state = app.invoke(initial_state)
            
            generated_answer = final_state.get("generation", "답변 생성 실패")
            retrieved_contexts = final_state.get("context", [])
            
        except Exception as e:
            print(f"\n❌ 에러 발생 (질문: {question}): {e}")
            generated_answer = f"Error: {str(e)}"
            retrieved_contexts = []
            
        # RAGAS 평가를 위한 포맷으로 저장
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": generated_answer,
            "contexts": retrieved_contexts # 리스트 형태의 출처
        })
        
        # API Rate Limit 방지용 딜레이
        time.sleep(1.5) 

    # 3. 결과 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 벤치마크 완료! 결과가 성공적으로 저장되었습니다: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_benchmark()