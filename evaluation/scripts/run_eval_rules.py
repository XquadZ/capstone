import os
import sys
import io
import json
import pandas as pd
from datasets import Dataset

# Langchain 통합 모듈 임포트
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings 

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness
)

# ==========================================
# 🛡️ 0. 터미널 인코딩 에러 방지
# ==========================================
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def evaluate_dataset(input_file, output_file, evaluator_llm, evaluator_embeddings):
    """
    단일 JSON 벤치마크 파일을 읽어 RAGAS 평가를 수행하고 CSV로 저장합니다.
    """
    if not os.path.exists(input_file):
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return

    print(f"\n📂 데이터 로드 중: {os.path.basename(input_file)}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ragas 데이터 포맷(contexts는 List[str] 필수) 검증
    for item in data:
        if not isinstance(item.get('contexts'), list):
            item['contexts'] = [str(item.get('contexts', ''))]

    eval_dataset = Dataset.from_list(data)
    print(f"✅ 총 {len(eval_dataset)}개의 Q&A 세트 로드 완료!")
    
    print("⏳ RAGAS 채점 진행 중... (약 10~20분 소요 가능)")
    
    # RAGAS 평가 실행
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,        # 환각 여부 (Vision이 텍스트보다 높을 것으로 예상)
            answer_relevancy,    # 질문 관련성 
            context_precision,   # 검색 정밀도 (둘 다 같은 엔진이므로 비슷해야 정상)
            answer_correctness   # 답변 정확도
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False 
    )

    print("\n" + "="*50)
    print(f"🎉 채점 완료! [{os.path.basename(input_file)}] 요약:")
    print(result)
    print("="*50)

    # DataFrame 변환 및 저장 (한글 깨짐 방지를 위해 utf-8-sig 사용)
    df = result.to_pandas()
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"💾 상세 성적표 저장 완료: {output_file}\n")


def main():
    # 1. SAIFEX API 설정 (채점관 LLM 용도)
    api_key = os.environ.get("SAIFEX_API_KEY")
    if not api_key:
        raise ValueError("❌ 환경변수 'SAIFEX_API_KEY'를 찾을 수 없습니다.")

    # 2. 파일 경로 설정
    results_dir = os.path.join("evaluation", "results")
    
    # 평가할 두 개의 파일 경로
    files_to_eval = [
        {
            "input": os.path.join(results_dir, "benchmark_rules_text.json"),
            "output": os.path.join(results_dir, "ragas_report_text.csv")
        },
        {
            "input": os.path.join(results_dir, "benchmark_rules_pdf.json"),
            "output": os.path.join(results_dir, "ragas_report_pdf.csv")
        }
    ]

    # 3. 채점관 모델 세팅
    print("🤖 채점관 LLM 준비 중 (SAIFEX API: gpt-4o-mini)...")
    evaluator_llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        temperature=0,
        openai_api_key=api_key,
        openai_api_base="https://ahoseo.saifex.ai/v1"
    )
    
    print("🔥 로컬 GPU 임베딩 모델(BGE-M3) 탑재 중...")
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 순차적으로 두 파일 모두 채점 실행
    print("\n" + "#"*60)
    print("🚀 [논문 비교 실험] 텍스트 RAG vs 멀티모달 RAG 자동 채점 시작")
    print("#"*60)
    
    for eval_config in files_to_eval:
        evaluate_dataset(
            input_file=eval_config["input"], 
            output_file=eval_config["output"], 
            evaluator_llm=evaluator_llm, 
            evaluator_embeddings=evaluator_embeddings
        )

if __name__ == "__main__":
    main()