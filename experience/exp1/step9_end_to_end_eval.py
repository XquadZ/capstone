import os
import sys
import json
import random
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm

from AgenticRAG.graph import build_graph
from AgenticRAG.nodes.router import classify_intent_v4

FINAL_INTENT_DATASET_PATH = "evaluation/datasets/final_intent_balanced_dataset.json"
RESULTS_DIR = "evaluation/results"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def norm(s: str) -> str:
    return " ".join(str(s).strip().split())

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_dataset_fixed_seed(data: List[dict], seed: int = 42):
    items = data[:]
    rnd = random.Random(seed)
    rnd.shuffle(items)

    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

def safe_list_str(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    x = str(x).strip()
    return [x] if x else []

def contexts_for_eval(res: dict) -> List[str]:
    retrieved = safe_list_str(res.get("retrieved_chunk_texts", []))
    if retrieved:
        return retrieved

    fallback = safe_list_str(res.get("context", ""))
    return fallback

def parse_gold_route(item: dict) -> str:
    candidates = [
        item.get("route"),
        item.get("routing_label"),
        item.get("label"),
        item.get("output")
    ]
    for c in candidates:
        if not c:
            continue
        text = str(c).strip().upper()
        if "VISION" in text:
            return "VISION"
        if "TEXT" in text:
            return "TEXT"
    return ""

def parse_ground_truth(item: dict) -> str:
    candidates = [
        item.get("ground_truth"),
        item.get("answer"),
        item.get("reference_answer")
    ]
    for c in candidates:
        if c is None:
            continue
        text = str(c).strip()
        if text and text != "정답 없음":
            return text
    return ""

def predict_route(question: str) -> str:
    pred = classify_intent_v4(question)
    pred = str(pred).strip().upper()
    if pred not in {"TEXT", "VISION"}:
        if "VISION" in pred:
            return "VISION"
        return "TEXT"
    return pred

def run_pipeline(question: str, force_route: str) -> dict:
    graph = build_graph()
    state = {
        "question": question,
        "force_route": force_route
    }

    start = time.time()
    result = graph.invoke(state)
    latency = time.time() - start

    return {
        "answer": str(result.get("answer", "")).strip(),
        "contexts": contexts_for_eval(result),
        "raw_context": result.get("context", ""),
        "retrieved_sources": result.get("retrieved_sources", []),
        "latency_sec": round(latency, 4)
    }

def build_ragas_row(
    question: str,
    ground_truth: str,
    answer: str,
    contexts: List[str],
    item: dict,
    pipeline: str,
    route_gold: str,
    route_pred: str
) -> dict:
    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": contexts,
        "pipeline": pipeline,
        "route_gold": route_gold,
        "route_pred": route_pred,
        "question_type": item.get("question_type", ""),
        "source_file": item.get("source_file", ""),
        "context_used": item.get("context_used", "")
    }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = load_json(FINAL_INTENT_DATASET_PATH)
    _, _, test_items = split_dataset_fixed_seed(data, seed=SEED)

    always_text_rows = []
    always_vision_rows = []
    tv_rag_rows = []
    analysis_rows = []

    route_total = 0
    route_correct = 0
    text_latencies = []
    vision_latencies = []
    tv_latencies = []

    skipped_no_question = 0
    skipped_no_gt = 0

    for item in tqdm(test_items, desc="step9 eval"):
        question = norm(item.get("question", ""))
        if not question:
            skipped_no_question += 1
            continue

        ground_truth = parse_ground_truth(item)
        if not ground_truth:
            skipped_no_gt += 1
            continue

        route_gold = parse_gold_route(item)
        route_pred = predict_route(question)

        if route_gold:
            route_total += 1
            if route_pred == route_gold:
                route_correct += 1

        text_res = run_pipeline(question, force_route="TEXT")
        vision_res = run_pipeline(question, force_route="VISION")

        chosen_res = text_res if route_pred == "TEXT" else vision_res

        text_latencies.append(text_res["latency_sec"])
        vision_latencies.append(vision_res["latency_sec"])
        tv_latencies.append(chosen_res["latency_sec"])

        always_text_rows.append(
            build_ragas_row(
                question=question,
                ground_truth=ground_truth,
                answer=text_res["answer"],
                contexts=text_res["contexts"],
                item=item,
                pipeline="Always_Text",
                route_gold=route_gold,
                route_pred=route_pred
            )
        )

        always_vision_rows.append(
            build_ragas_row(
                question=question,
                ground_truth=ground_truth,
                answer=vision_res["answer"],
                contexts=vision_res["contexts"],
                item=item,
                pipeline="Always_Vision",
                route_gold=route_gold,
                route_pred=route_pred
            )
        )

        tv_rag_rows.append(
            build_ragas_row(
                question=question,
                ground_truth=ground_truth,
                answer=chosen_res["answer"],
                contexts=chosen_res["contexts"],
                item=item,
                pipeline="TV_RAG",
                route_gold=route_gold,
                route_pred=route_pred
            )
        )

        analysis_rows.append({
            "question": question,
            "ground_truth": ground_truth,
            "question_type": item.get("question_type", ""),
            "route_gold": route_gold,
            "route_pred": route_pred,
            "text_rag_answer": text_res["answer"],
            "vision_rag_answer": vision_res["answer"],
            "tv_rag_answer": chosen_res["answer"],
            "text_contexts": text_res["contexts"],
            "vision_contexts": vision_res["contexts"],
            "tv_contexts": chosen_res["contexts"],
            "text_sources": text_res["retrieved_sources"],
            "vision_sources": vision_res["retrieved_sources"],
            "text_latency_sec": text_res["latency_sec"],
            "vision_latency_sec": vision_res["latency_sec"],
            "tv_latency_sec": chosen_res["latency_sec"],
            "source_file": item.get("source_file", ""),
            "context_used": item.get("context_used", "")
        })

    save_json(os.path.join(RESULTS_DIR, "benchmark_Always_Text.json"), always_text_rows)
    save_json(os.path.join(RESULTS_DIR, "benchmark_Always_Vision.json"), always_vision_rows)
    save_json(os.path.join(RESULTS_DIR, "benchmark_TV_RAG.json"), tv_rag_rows)
    save_json(os.path.join(RESULTS_DIR, "benchmark_analysis_full.json"), analysis_rows)

    route_acc = (route_correct / route_total) if route_total > 0 else 0.0
    avg_text_latency = sum(text_latencies) / len(text_latencies) if text_latencies else 0.0
    avg_vision_latency = sum(vision_latencies) / len(vision_latencies) if vision_latencies else 0.0
    avg_tv_latency = sum(tv_latencies) / len(tv_latencies) if tv_latencies else 0.0

    print("\n" + "=" * 80)
    print("[STEP9 완료]")
    print("=" * 80)
    print(f"전체 데이터 수: {len(data)}")
    print(f"test split 수: {len(test_items)}")
    print(f"question 없음으로 스킵: {skipped_no_question}")
    print(f"ground_truth 없음으로 스킵: {skipped_no_gt}")
    print(f"Always_Text rows: {len(always_text_rows)}")
    print(f"Always_Vision rows: {len(always_vision_rows)}")
    print(f"TV_RAG rows: {len(tv_rag_rows)}")
    print(f"Routing Accuracy: {route_acc:.4f} ({route_correct}/{route_total})")
    print(f"Avg Always-Text Latency: {avg_text_latency:.2f}s")
    print(f"Avg Always-Vision Latency: {avg_vision_latency:.2f}s")
    print(f"Avg TV-RAG Latency: {avg_tv_latency:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
