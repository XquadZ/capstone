import os
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
    answer_correctness,
)

# RAGAS 입력 검증 (짧은 라벨만 있는 contexts / 빈 정답 제외)
MIN_QUESTION_LEN = 3
MIN_CONTEXT_CHARS = 50  # 본문 청크로 간주할 최소 길이
INVALID_GROUND_TRUTH = frozenset({"", "정답 없음", "정답 없음.", "none", "null"})


def validate_ragas_row(item: dict, min_context_chars: int = MIN_CONTEXT_CHARS) -> list:
    """위반 시 이유 코드 문자열 리스트 반환. 빈 리스트면 통과."""
    errs = []
    q = str(item.get("question", "")).strip()
    if len(q) < MIN_QUESTION_LEN:
        errs.append("question_too_short")

    ans = str(item.get("answer", "")).strip()
    if not ans or ans == "생성 실패":
        errs.append("empty_or_failed_answer")

    gt = str(item.get("ground_truth", "")).strip()
    if not gt or gt in INVALID_GROUND_TRUTH:
        errs.append("missing_ground_truth")

    ctx = item.get("contexts", [])
    if not isinstance(ctx, list):
        ctx = [str(ctx)]
    ctx = [str(c).strip() for c in ctx if str(c).strip()]
    if not ctx:
        errs.append("empty_contexts")
    else:
        long_chunks = sum(1 for c in ctx if len(c) >= min_context_chars)
        if long_chunks == 0:
            errs.append("contexts_not_substantive_maybe_labels_only")

    return errs


def filter_ragas_dataset(data: list) -> tuple[list, list]:
    """검증 실패 행은 제외. 반환: (valid_rows, skipped: list of (index, errors))."""
    valid = []
    skipped = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            skipped.append((i, ["not_a_dict"]))
            continue
        e = validate_ragas_row(item)
        if e:
            skipped.append((i, e))
            continue
        valid.append(item)
    return valid, skipped


def run_ragas_evaluation(
    input_file,
    output_file,
    evaluator_llm,
    evaluator_embeddings,
    fail_if_empty: bool = True,
):
    if not os.path.exists(input_file):
        print(f"❌ 벤치마크 파일이 없습니다: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ JSON 루트는 리스트여야 합니다.")
        return

    valid_rows, skipped = filter_ragas_dataset(data)

    if skipped:
        print(f"⚠️ RAGAS 검증 스킵: {len(skipped)} / {len(data)} 건")
        for idx, (i, errs) in enumerate(skipped[:8]):
            print(f"   - row {i}: {', '.join(errs)}")
        if len(skipped) > 8:
            print(f"   ... 외 {len(skipped) - 8}건")

    if not valid_rows:
        msg = "❌ 검증을 통과한 행이 없습니다. notice_qa_2000_verified 조인·청크 contexts를 확인하세요."
        if fail_if_empty:
            raise ValueError(msg)
        print(msg)
        return

    formatted_data = []
    for item in valid_rows:
        ctx = item.get("contexts", [])
        if not isinstance(ctx, list):
            ctx = [str(ctx)]
        else:
            ctx = [str(c) for c in ctx]

        gt_text = str(item.get("ground_truth", "정답 없음"))

        formatted_data.append(
            {
                "question": str(item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "contexts": ctx,
                "ground_truth": gt_text,
                "reference": gt_text,
            }
        )

    eval_dataset = Dataset.from_list(formatted_data)
    print(f"\n▶️ [{os.path.basename(input_file)}] RAGAS 채점: 사용 {len(eval_dataset)}건 (원본 {len(data)}건)")

    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                answer_correctness,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
        )

        print(f"🎉 성적 요약:\n{result}")

        df = result.to_pandas()
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"💾 저장: {output_file}")

    except Exception as e:
        print(f"❌ RAGAS 평가 중 에러: {e}")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ 환경변수 'OPENAI_API_KEY'가 없습니다.")

    results_dir = "evaluation/results"
    datasets_dir = "evaluation/datasets"

    # prepare_ragas_eval_from_benchmarks.py 로 생성한 파일이 있으면 우선 사용
    use_prepared = os.environ.get("RAGAS_INPUT", "auto").strip().lower()
    prepared_names = [
        ("eval_ragas_notice_Always_Text.json", "ragas_report_Always_Text.csv"),
        ("eval_ragas_notice_Always_Vision.json", "ragas_report_Always_Vision.csv"),
        ("eval_ragas_notice_TV_RAG.json", "ragas_report_TV_RAG.csv"),
    ]
    raw_names = [
        ("benchmark_Always_Text.json", "ragas_report_Always_Text.csv"),
        ("benchmark_Always_Vision.json", "ragas_report_Always_Vision.csv"),
        ("benchmark_TV_RAG.json", "ragas_report_TV_RAG.csv"),
    ]

    if use_prepared == "prepared":
        pipelines = [
            (os.path.join(datasets_dir, a), os.path.join(results_dir, b)) for a, b in prepared_names
        ]
    elif use_prepared == "raw":
        pipelines = [(os.path.join(results_dir, a), os.path.join(results_dir, b)) for a, b in raw_names]
    else:
        pipelines = []
        for (pa, ca), (ra, cr) in zip(prepared_names, raw_names):
            pp = os.path.join(datasets_dir, pa)
            pr = os.path.join(results_dir, ra)
            if os.path.isfile(pp):
                pipelines.append((pp, os.path.join(results_dir, ca)))
            elif os.path.isfile(pr):
                pipelines.append((pr, os.path.join(results_dir, cr)))
            else:
                print(f"⏭️ 건너뜀 (없음): {pa} / {os.path.basename(pr)}")

    print("\n🤖 RAGAS (gpt-4o-mini + BGE-M3) …")
    evaluator_llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    for in_file, out_file in pipelines:
        run_ragas_evaluation(
            in_file,
            out_file,
            evaluator_llm,
            evaluator_embeddings,
            fail_if_empty=False,
        )

    print("\n" + "=" * 60)
    print("🔥 RAGAS 평가 루프 종료. CSV는 evaluation/results/ 를 확인하세요.")
    print("   (입력 전환: RAGAS_INPUT=prepared|raw|auto , 기본 auto = datasets 쪽 파일 우선)")
    print("=" * 60)


if __name__ == "__main__":
    main()
