import os
import sys
import io
import json
import pandas as pd
from datasets import Dataset

# Langchain 모듈
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings 

# Ragas 모듈
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness
)
from ragas.run_config import RunConfig

# ==========================================
# 🛡️ 0. 터미널 인코딩 에러 방지
# ==========================================
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 📊 1. 10개 단위 실시간 저장 평가 로직
# ==========================================
def evaluate_with_checkpoint(input_file, output_file, evaluator_llm, evaluator_embeddings, chunk_size=10):
    if not os.path.exists(input_file):
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return

    # 원본 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    print(f"\n📂 [{os.path.basename(input_file)}] 데이터 로드 완료 (총 {len(full_data)}문항)")
    
    all_results_df = [] # 청크별 결과를 누적할 리스트

    # 전체 데이터를 chunk_size(10개) 단위로 쪼개서 순회
    for i in range(0, len(full_data), chunk_size):
        chunk_data = full_data[i:i + chunk_size]
        current_range = f"{i+1} ~ {min(i+chunk_size, len(full_data))}"
        print(f"\n🚀 [{current_range}] 구간 채점 시작...")
        
        # Ragas 포맷 검증 (contexts는 반드시 List[str] 형태여야 함)
        for item in chunk_data:
            if not isinstance(item.get('contexts'), list):
                item['contexts'] = [str(item.get('contexts', ''))]
        
        # 10개짜리 데이터셋 생성
        eval_dataset = Dataset.from_list(chunk_data)
        
        try:
            # 10개만 먼저 채점 (max_workers=2로 낮춰서 API Rate Limit 방어)
            result = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                raise_exceptions=False,
                run_config=RunConfig(max_workers=2, timeout=60) 
            )
            
            # 현재 10개의 결과를 DataFrame으로 변환 후 전체 리스트에 누적
            chunk_df = result.to_pandas()
            all_results_df.append(chunk_df)
            
            # 🔥 [핵심] 지금까지 누적된 모든 결과를 하나의 DataFrame으로 합쳐서 CSV로 강제 저장
            final_df = pd.concat(all_results_df, ignore_index=True)
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"✅ [{current_range}] 구간 완료! (현재까지 총 {len(final_df)}개 실시간 저장됨)")

        except Exception as e:
            print(f"⚠️ [{current_range}] 구간 채점 중 에러 발생 (건너뛰고 다음 진행): {e}")
            continue

    print(f"\n✨ {os.path.basename(input_file)} 전체 평가 완전 종료! 최종 결과: {output_file}")


# ==========================================
# 🚀 2. 메인 실행부
# ==========================================
def main():
    # 공식 OpenAI API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ 환경변수 'OPENAI_API_KEY'를 찾을 수 없습니다. (공식 키 필요)")

    results_dir = os.path.join("evaluation", "results")
    
    # 평가 대상 2가지 (텍스트 vs 멀티모달)
    targets = [
        {
            "in": os.path.join(results_dir, "benchmark_rules_text.json"),
            "out": os.path.join(results_dir, "ragas_report_text.csv")
        },
        {
            "in": os.path.join(results_dir, "benchmark_rules_pdf.json"),
            "out": os.path.join(results_dir, "ragas_report_pdf.csv")
        }
    ]

    print("🤖 채점관 LLM 준비 중 (OpenAI: gpt-4o-mini)...")
    evaluator_llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        temperature=0, 
        openai_api_key=api_key
    )
    
    print("🔥 로컬 GPU 임베딩 모델(BGE-M3) 탑재 중...")
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("\n" + "#"*60)
    print("🚀 [논문 비교 실험] 실시간 저장 방식 채점 시작 (10문항 단위)")
    print("#"*60)

    for target in targets:
        evaluate_with_checkpoint(
            input_file=target["in"],
            output_file=target["out"],
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
            chunk_size=10 # 10개 풀 때마다 저장
        )

if __name__ == "__main__":
    main()