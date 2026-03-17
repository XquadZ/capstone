import os
import json
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
# ⬇️ OpenAI 임베딩 대신 허깅페이스 로컬 임베딩을 가져옵니다.
from langchain_huggingface import HuggingFaceEmbeddings 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness
)

def main():
    # 1. 환경 변수 및 경로 설정
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ 환경변수 'OPENAI_API_KEY'를 찾을 수 없습니다.")

    results_dir = "evaluation/results"
    input_file = os.path.join(results_dir, "benchmark_gpt4o_mini.json")
    output_file = os.path.join(results_dir, "ragas_evaluation_report.csv")

    if not os.path.exists(input_file):
        print(f"❌ 벤치마크 결과 파일이 없습니다: {input_file}")
        return

    # 2. 데이터 로드 및 Ragas 포맷으로 변환
    print("📂 벤치마크 데이터를 불러오는 중...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ragas가 요구하는 데이터 타입에 맞게 검증 (contexts는 리스트여야 함)
    for item in data:
        if not isinstance(item.get('contexts'), list):
            item['contexts'] = [str(item.get('contexts', ''))]

    # HuggingFace Dataset 객체로 변환
    eval_dataset = Dataset.from_list(data)
    print(f"✅ 총 {len(eval_dataset)}개의 Q&A 세트 로드 완료!\n")

    # 3. 🧠 [핵심] 평가용 모델 세팅 (하이브리드 비용 방어)
    print("🤖 채점관 LLM 준비 중 (API: gpt-4o-mini)...")
    evaluator_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    print("🔥 RTX 4090 GPU에 로컬 임베딩 모델(BGE-M3) 탑재 중...")
    # ⬇️ 내 그래픽카드 VRAM을 직접 갈구는 로컬 임베딩 세팅!
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'}, # 여기서 4090으로 강제 배정합니다.
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. RAGAS 평가 시작
    print("⏳ RAGAS 자동 채점을 시작합니다. (작업 관리자에서 GPU가 일하는지 확인해보세요!)")
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,        # 환각 여부
            answer_relevancy,    # 질문 관련성 (여기서 로컬 GPU가 맹활약합니다!)
            context_precision,   # 검색 정밀도
            answer_correctness   # 답변 정확도
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False 
    )

    # 5. 결과 저장 및 출력
    print("\n" + "="*60)
    print("🎉 채점 완료! 최종 성적표 요약:")
    print("="*60)
    print(result)

    # 세부 문항별 채점 결과를 Pandas DataFrame으로 변환 후 CSV(엑셀) 저장
    df = result.to_pandas()
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print(f"💾 문항별 상세 성적표가 저장되었습니다: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()