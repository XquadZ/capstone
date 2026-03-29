import os
import sys
import json
import pandas as pd
from datasets import Dataset

# RAGAS 최신 모듈 임포트
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 환각 여부 (Context 기반 생성)
    answer_correctness,  # 정답 유사도 (Ground Truth와 비교)
    answer_relevancy,    # 답변 관련성 (질문 의도 파악)
    context_precision,   # 검색 정밀도 (정답 관련 문서가 상위에 랭크되었는가?)
    context_recall       # 검색 재현율 (정답 도출에 필요한 문서를 다 찾았는가?)
)

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# 1. API 키 및 LLM/Embedding 설정 (SAIFEX 강제 납치)
# =====================================================================
saifex_key = os.getenv("SAIFEX_API_KEY", os.getenv("OPENAI_API_KEY"))
if not saifex_key:
    raise ValueError("❌ 환경 변수에 API 키가 설정되지 않았습니다.")

print("🤖 [1/2] RAGAS 심판관 LLM 준비 중 (SAIFEX: gpt-4o-mini)...")
evaluator_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    api_key=saifex_key,
    base_url="https://ahoseo.saifex.ai/v1"
)

print("🧠 [2/2] RAGAS 임베딩 준비 중 (로컬 GPU: BGE-M3)...")
evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'}, 
    encode_kwargs={'normalize_embeddings': True}
)

# =====================================================================
# 2. 경로 및 설정
# =====================================================================
RESULTS_DIR = "evaluation/results"
EVAL_FILES = [
    "benchmark_Always_Text.json",
    "benchmark_Always_Vision.json",
    "benchmark_TV_RAG.json"
]

metrics = [
    faithfulness,
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall
]

def load_data_for_ragas(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in data:
        q = item.get("question", "")
        a = item.get("answer", item.get("generated_answer", "")) 
        c = item.get("contexts", item.get("retrieved_contexts", []))
        if isinstance(c, str):
            c = [c]
        gt = item.get("ground_truth", "")

        if not a: a = "No answer provided."
        if not c: c = ["No context retrieved."]
        if not gt: gt = "No ground truth available."

        questions.append(q)
        answers.append(a)
        contexts.append(c)
        ground_truths.append(gt)

    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    return Dataset.from_dict(data_dict)

def main():
    print("\n" + "="*60)
    print("🚀 [STEP 10] RAGAS 기반 파이프라인 성능 평가 시작")
    print("="*60)

    summary_results = []

    for file_name in EVAL_FILES:
        filepath = os.path.join(RESULTS_DIR, file_name)
        save_name = file_name.replace(".json", "_RAGAS.csv")
        save_path = os.path.join(RESULTS_DIR, save_name)

        if not os.path.exists(filepath):
            print(f"⚠️ 경고: {file_name} 파일을 찾을 수 없어 건너뜁니다.")
            continue

        # 🔥 [핵심 추가] 이어하기 로직: 이미 CSV 파일이 있으면 평가 스킵!
        if os.path.exists(save_path):
            print(f"\n⏩ [{file_name}] 이미 평가된 결과가 존재합니다. (평가 생략, 요약표에만 병합)")
            df_result = pd.read_csv(save_path)
        else:
            print(f"\n▶ [{file_name}] 평가 준비 중...")
            dataset = load_data_for_ragas(filepath)
            print(f"   - 총 {len(dataset)}개의 데이터를 평가합니다.")

            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm,               
                embeddings=evaluator_embeddings, 
                raise_exceptions=False 
            )

            df_result = result.to_pandas()
            df_result.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"✅ [{file_name}] 평가 완료 및 저장 성공!")

        # 🔥 [버그 수정] KeyError: 0 방지를 위해 pandas DataFrame에서 직접 평균값 추출
        summary = {"Pipeline": file_name.replace("benchmark_", "").replace(".json", "")}
        
        target_metrics = [
            "faithfulness", "answer_correctness", "answer_relevancy", 
            "context_precision", "context_recall"
        ]
        
        for metric in target_metrics:
            if metric in df_result.columns:
                summary[metric] = df_result[metric].mean()
            else:
                summary[metric] = 0.0 # 만약 해당 지표가 에러로 누락되었으면 0점 처리

        summary_results.append(summary)

    # 4. 전체 요약표 생성
    if summary_results:
        print("\n" + "="*60)
        print("🏆 [최종 결과 요약표]")
        print("="*60)
        df_summary = pd.DataFrame(summary_results)
        print(df_summary.to_string(index=False))
        
        summary_path = os.path.join(RESULTS_DIR, "RAGAS_Final_Summary.csv")
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n📊 최종 비교 요약표가 저장되었습니다: {summary_path}")

if __name__ == "__main__":
    main()