"""
기존 benchmark_*.json을 RAGAS 입력용으로 보정합니다.

- notice_qa_2000_verified.json 기준으로 question → ground_truth 재조인
- (선택) contexts가 짧은 라벨뿐이면 Milvus 검색으로 retrieved_chunk_texts 재충전

사용 (저장소 루트에서):
  python evaluation/scripts/prepare_ragas_eval_from_benchmarks.py \\
    --input evaluation/results/benchmark_Always_Text.json \\
    --output evaluation/datasets/eval_ragas_notice_Always_Text.json

  python evaluation/scripts/prepare_ragas_eval_from_benchmarks.py --all --refill-contexts
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_VERIFIED = os.path.join(ROOT, "evaluation", "datasets", "notice_qa_2000_verified.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_DIR = os.path.join(ROOT, "evaluation", "datasets")


def load_gt_map(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"⚠️ verified 파일 없음: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    m = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        q = (row.get("question") or "").strip()
        gt = row.get("ground_truth")
        if q and gt is not None and str(gt).strip():
            m[q] = str(gt).strip()
    return m


def contexts_look_like_labels_only(ctxs: list, min_chunk_chars: int = 120) -> bool:
    """모든 조각이 짧으면(라벨/로그만) True → refill 후보."""
    if not ctxs:
        return True
    return all(len(str(c).strip()) < min_chunk_chars for c in ctxs)


def refill_retrieved_chunks(question: str, final_k: int = 5) -> list:
    from AgenticRAG.nodes.vision_rag import _chunk_texts_from_hits
    from ai_engine.rag_pipeline_notice import get_shared_notice_pipeline

    pipe = get_shared_notice_pipeline()
    hits = pipe.search_and_rerank(question, retrieve_k=50, final_k=final_k)
    return _chunk_texts_from_hits(hits)


def process_file(
    input_path: str,
    output_path: str,
    gt_map: dict,
    refill: bool,
) -> tuple[int, int]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    n_refill = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        row = dict(item)
        if q in gt_map:
            row["ground_truth"] = gt_map[q]

        ctxs = row.get("contexts", [])
        if not isinstance(ctxs, list):
            ctxs = [str(ctxs)]
        ctxs = [str(c).strip() for c in ctxs if str(c).strip()]

        if refill and contexts_look_like_labels_only(ctxs):
            try:
                new_c = refill_retrieved_chunks(q)
                if new_c:
                    row["contexts"] = new_c
                    n_refill += 1
            except Exception as e:
                print(f"⚠️ refill 실패 (skip): {q[:40]}... | {e}")

        out.append(row)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return len(out), n_refill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, help="benchmark_*.json 경로")
    ap.add_argument("--output", type=str, help="출력 JSON 경로")
    ap.add_argument("--verified", type=str, default=DEFAULT_VERIFIED)
    ap.add_argument("--refill-contexts", action="store_true", help="짧은 contexts면 Milvus에서 재검색")
    ap.add_argument(
        "--all",
        action="store_true",
        help="benchmark_Always_Text/Vision/TV_RAG 세 파일 일괄 처리 → evaluation/datasets/eval_ragas_notice_*.json",
    )
    args = ap.parse_args()

    gt_map = load_gt_map(args.verified)
    print(f"✅ ground_truth 맵: {len(gt_map)}건 ({args.verified})")

    if args.all:
        pairs = [
            ("benchmark_Always_Text.json", "eval_ragas_notice_Always_Text.json"),
            ("benchmark_Always_Vision.json", "eval_ragas_notice_Always_Vision.json"),
            ("benchmark_TV_RAG.json", "eval_ragas_notice_TV_RAG.json"),
        ]
        for src, dst in pairs:
            inp = os.path.join(RESULTS_DIR, src)
            outp = os.path.join(OUT_DIR, dst)
            if not os.path.isfile(inp):
                print(f"⏭️ 건너뜀 (없음): {inp}")
                continue
            n, nr = process_file(inp, outp, gt_map, args.refill_contexts)
            print(f"💾 {src} → {dst} | rows={n}, refill={nr}")
        return

    if not args.input or not args.output:
        ap.error("--input 와 --output 이 필요합니다 (--all 사용 시 제외)")

    n, nr = process_file(args.input, args.output, gt_map, args.refill_contexts)
    print(f"💾 완료 rows={n}, refill 적용={nr} → {args.output}")


if __name__ == "__main__":
    main()
