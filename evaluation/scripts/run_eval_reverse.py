import os
import sys
import io
import json
import pandas as pd
from datasets import Dataset

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings 

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
# 📊 1. 10개 단위 실시간 저장 역방향 평가 로직
# ==========================================
def evaluate_reverse_checkpoint(dataset_list, output_file, evaluator_llm, evaluator_embeddings, chunk_size=10):
    """
    구성된 역방향 데이터셋(리스트)을 받아 10개 단위로 평가하고 저장합니다.
    """
    print(f"\n📂 [{os.path.basename(output_file)}] 평가 시작 (총 {len(dataset_list)}문항)")
    all_results_df = [] 

    for i in range(0, len(dataset_list), chunk_size):
        chunk_data = dataset_list[i:i + chunk_size]
        current_range = f"{i+1} ~ {min(i+chunk_size, len(dataset_list))}"
        print(f"🚀 [{current_range}] 구간 채점 시작...")
        
        # Ragas 포맷 강제 변환 (contexts는 리스트여야 함)
        for item in chunk_data:
            if not isinstance(item.get('contexts'), list):
                item['contexts'] = [str(item.get('contexts', ''))]
        
        eval_dataset = Dataset.from_list(chunk_data)
        
        try:
            # 평가 실행 (SAIFEX API Rate Limit 방지를 위해 일꾼 2명으로 제한)
            result = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                raise_exceptions=False,
                run_config=RunConfig(max_workers=2, timeout=60) 
            )
            
            chunk_df = result.to_pandas()
            all_results_df.append(chunk_df)
            
            # 실시간 덮어쓰기 저장
            final_df = pd.concat(all_results_df, ignore_index=True)
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✅ [{current_range}] 완료! 실시간 저장됨.")

        except Exception as e:
            print(f"⚠️ [{current_range}] 구간 채점 중 에러 발생: {e}")
            continue

    print(f"✨ 전체 평가 완료! 결과: {output_file}")

# ==========================================
# 🚀 2. 메인 실행부
# ==========================================
def main():
    # 1. 🔑 [수정됨] SAIFEX API 키 설정
    api_key = os.environ.get("SAIFEX_API_KEY")
    if not api_key:
        raise ValueError("❌ 'SAIFEX_API_KEY'가 없습니다. 환경변수를 확인해주세요.")

    results_dir = os.path.join("evaluation", "results")
    
    # 기존에 뽑아둔 벤치마크 결과 파일 로드
    pdf_file = os.path.join(results_dir, "benchmark_rules_pdf.json")
    text_file = os.path.join(results_dir, "benchmark_rules_text.json")
    
    if not os.path.exists(pdf_file) or not os.path.exists(text_file):
        print("❌ 벤치마크 원본 JSON 파일이 evaluation/results 폴더에 없습니다.")
        return

    with open(pdf_file, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)
    with open(text_file, 'r', encoding='utf-8') as f:
        text_data = json.load(f)

    print("\n" + "="*60)
    print("🔄 [논문 핵심 방어 로직] 역방향 데이터셋 재구성 중...")
    print("="*60)

    # 역방향(Reverse) 데이터셋 만들기
    dataset_text_vs_vision = []
    dataset_origGT_vs_vision = []

    for pdf_item, text_item in zip(pdf_data, text_data):
        # 🔥 핵심: 멀티모달(Vision)이 만든 디테일한 답변을 새로운 '절대 정답(Gold Standard)'으로 승격
        vision_gold_truth = pdf_item["answer"]
        
        # [실험 1] Text RAG가 만든 답변이 Vision 정답에 비해 얼마나 부족한가?
        dataset_text_vs_vision.append({
            "question": text_item["question"],
            "answer": text_item["answer"],          # Text RAG가 낸 답
            "contexts": text_item["contexts"],      # Text RAG가 본 문서
            "ground_truth": vision_gold_truth       # 기준점: Vision 답변
        })

        # [실험 2] 과거 텍스트로만 만들었던 기존 정답(GT)이 Vision 정답에 비해 얼마나 부족한가?
        dataset_origGT_vs_vision.append({
            "question": text_item["question"],
            "answer": text_item["ground_truth"],    # 과거 기존 정답
            "contexts": text_item["contexts"],      # (평가를 위해 동일 컨텍스트 제공)
            "ground_truth": vision_gold_truth       # 기준점: Vision 답변
        })

    # 2. 채점관 모델 세팅 (SAIFEX API로 다시 연결)
    print("🤖 채점관 LLM 준비 중 (SAIFEX API: gpt-4o-mini)...")
    evaluator_llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        temperature=0, 
        openai_api_key=api_key,
        openai_api_base="https://ahoseo.saifex.ai/v1" # SAIFEX 서버 주소 추가
    )
    
    print("🔥 로컬 GPU 임베딩 모델(BGE-M3) 탑재 중...")
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 3. 평가 시작
    print("\n🚀 [실험 1] Text RAG 답변을 Vision 정답 기준으로 채점합니다.")
    out_text = os.path.join(results_dir, "ragas_reverse_report_text.csv")
    evaluate_reverse_checkpoint(dataset_text_vs_vision, out_text, evaluator_llm, evaluator_embeddings)

    print("\n🚀 [실험 2] 과거 빈약했던 기존 GT를 Vision 정답 기준으로 채점합니다.")
    out_gt = os.path.join(results_dir, "ragas_reverse_report_orig_gt.csv")
    evaluate_reverse_checkpoint(dataset_origGT_vs_vision, out_gt, evaluator_llm, evaluator_embeddings)

if __name__ == "__main__":
    main()