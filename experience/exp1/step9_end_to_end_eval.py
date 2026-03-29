import os
import sys
import json
import random
import time
from typing import Any, List

# ==========================================
# 🛡️ 1. SAIFEX API 강제 주입 (OpenAI 과금/한도초과 에러 원천 차단)
# ==========================================
# 기존 노드들이 OpenAI 공식 서버로 가는 것을 막고, 무조건 SAIFEX로 가도록 환경변수를 덮어씁니다.
saifex_key = os.getenv("SAIFEX_API_KEY", os.getenv("OPENAI_API_KEY"))
if not saifex_key:
    raise ValueError("❌ 환경 변수에 'SAIFEX_API_KEY' 또는 'OPENAI_API_KEY'가 없습니다.")

os.environ["OPENAI_API_KEY"] = saifex_key
os.environ["OPENAI_BASE_URL"] = "https://ahoseo.saifex.ai/v1"

# ==========================================
# ⚙️ 2. 기존 코드 및 경로 설정 유지
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm

from AgenticRAG.nodes.router import slm_router_node
from AgenticRAG.nodes.text_rag import text_rag_node
from AgenticRAG.nodes.vision_rag import vision_rag_node


FINAL_INTENT_DATASET_PATH = "evaluation/datasets/final_intent_balanced_dataset.json"
RESULTS_DIR = "evaluation/results"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SAVE_EVERY = 5

FINAL_ALWAYS_TEXT = os.path.join(RESULTS_DIR, "benchmark_Always_Text.json")
FINAL_ALWAYS_VISION = os.path.join(RESULTS_DIR, "benchmark_Always_Vision.json")
FINAL_TV_RAG = os.path.join(RESULTS_DIR, "benchmark_TV_RAG.json")
FINAL_ANALYSIS = os.path.join(RESULTS_DIR, "benchmark_analysis_full.json")

PARTIAL_ALWAYS_TEXT = os.path.join(RESULTS_DIR, "benchmark_Always_Text.partial.json")
PARTIAL_ALWAYS_VISION = os.path.join(RESULTS_DIR, "benchmark_Always_Vision.partial.json")
PARTIAL_TV_RAG = os.path.join(RESULTS_DIR, "benchmark_TV_RAG.partial.json")
PARTIAL_ANALYSIS = os.path.join(RESULTS_DIR, "benchmark_analysis_full.partial.json")
PROGRESS_PATH = os.path.join(RESULTS_DIR, "step9_progress.json")


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
    return safe_list_str(res.get("context", []))


def parse_ground_truth(item: dict) -> str:
    candidates = [
        item.get("ground_truth"),
        item.get("answer"),
        item.get("reference_answer"),
    ]
    for c in candidates:
        if c is None:
            continue
        text = str(c).strip()
        if text and text != "정답 없음":
            return text
    return ""


def parse_gold_route(item: dict) -> str:
    candidates = [
        item.get("route"),
        item.get("routing_label"),
        item.get("label"),
        item.get("output"),
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


def run_router(question: str) -> str:
    state = {
        "question": question,
        "retry_count": 0
    }
    res = slm_router_node(state)
    decision = str(res.get("route_decision", "TEXT")).strip().upper()
    if decision not in {"TEXT", "VISION"}:
        decision = "TEXT"
    return decision


def run_text_rag(question: str) -> dict:
    state = {
        "question": question,
        "retry_count": 0
    }
    start = time.time()
    res = text_rag_node(state)
    latency = time.time() - start

    return {
        "answer": str(res.get("generation", "")).strip(),
        "contexts": contexts_for_eval(res),
        "debug_context": safe_list_str(res.get("context", [])),
        "latency_sec": round(latency, 4),
    }


def run_vision_rag(question: str) -> dict:
    state = {
        "question": question,
        "retry_count": 0
    }
    start = time.time()
    res = vision_rag_node(state)
    latency = time.time() - start

    return {
        "answer": str(res.get("generation", "")).strip(),
        "contexts": contexts_for_eval(res),
        "debug_context": safe_list_str(res.get("context", [])),
        "latency_sec": round(latency, 4),
    }


def build_ragas_row(
    question: str,
    ground_truth: str,
    answer: str,
    contexts: List[str],
    item: dict,
    pipeline: str,
    route_gold: str,
    route_pred: str,
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
        "context_used": item.get("context_used", ""),
    }


def save_partial(
    processed_count: int,
    total_count: int,
    always_text_rows: list,
    always_vision_rows: list,
    tv_rag_rows: list,
    analysis_rows: list,
    route_correct: int,
    route_total: int,
    text_latencies: list,
    vision_latencies: list,
    tv_latencies: list,
):
    save_json(PARTIAL_ALWAYS_TEXT, always_text_rows)
    save_json(PARTIAL_ALWAYS_VISION, always_vision_rows)
    save_json(PARTIAL_TV_RAG, tv_rag_rows)
    save_json(PARTIAL_ANALYSIS, analysis_rows)

    route_acc = (route_correct / route_total) if route_total > 0 else 0.0
    avg_text_latency = sum(text_latencies) / len(text_latencies) if text_latencies else 0.0
    avg_vision_latency = sum(vision_latencies) / len(vision_latencies) if vision_latencies else 0.0
    avg_tv_latency = sum(tv_latencies) / len(tv_latencies) if tv_latencies else 0.0

    progress = {
        "processed_count": processed_count,
        "total_count": total_count,
        "route_correct": route_correct,
        "route_total": route_total,
        "route_accuracy_so_far": round(route_acc, 4),
        "avg_text_latency_so_far": round(avg_text_latency, 4),
        "avg_vision_latency_so_far": round(avg_vision_latency, 4),
        "avg_tv_latency_so_far": round(avg_tv_latency, 4),
        "partial_files": {
            "always_text": PARTIAL_ALWAYS_TEXT,
            "always_vision": PARTIAL_ALWAYS_VISION,
            "tv_rag": PARTIAL_TV_RAG,
            "analysis": PARTIAL_ANALYSIS,
        },
    }
    save_json(PROGRESS_PATH, progress)
    print(f"\n💾 중간 저장 완료: {processed_count}/{total_count}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = load_json(FINAL_INTENT_DATASET_PATH)
    _, _, test_items = split_dataset_fixed_seed(dataset, seed=SEED)

    # ==========================================
    # 🔄 3. 이어하기(Resume) 데이터 로드 로직
    # ==========================================
    always_text_rows = []
    always_vision_rows = []
    tv_rag_rows = []
    analysis_rows = []

    route_total = 0
    route_correct = 0

    text_latencies = []
    vision_latencies = []
    tv_latencies = []
    processed_count = 0
    
    completed_questions = set()

    # 이전에 저장된 부분 진행 데이터가 있다면 불러옵니다.
    if os.path.exists(PROGRESS_PATH) and os.path.exists(PARTIAL_ANALYSIS):
        print("🔄 기존에 저장된 진행 데이터를 발견했습니다. 이어하기를 준비합니다...")
        try:
            progress = load_json(PROGRESS_PATH)
            route_correct = progress.get("route_correct", 0)
            route_total = progress.get("route_total", 0)
            
            always_text_rows = load_json(PARTIAL_ALWAYS_TEXT)
            always_vision_rows = load_json(PARTIAL_ALWAYS_VISION)
            tv_rag_rows = load_json(PARTIAL_TV_RAG)
            analysis_rows = load_json(PARTIAL_ANALYSIS)
            
            processed_count = len(analysis_rows)
            completed_questions = set([row["question"] for row in analysis_rows])
            
            # 평균 계산을 위해 지연 시간 복구
            text_latencies = [row["text_latency_sec"] for row in analysis_rows]
            vision_latencies = [row["vision_latency_sec"] for row in analysis_rows]
            tv_latencies = [row["tv_latency_sec"] for row in analysis_rows]
            
            print(f"✅ 총 {processed_count}개의 데이터를 성공적으로 불러왔습니다. 나머지 작업을 시작합니다!\n")
        except Exception as e:
            print(f"⚠️ 부분 데이터 로드 실패. 처음부터 다시 시작합니다. (에러: {e})")

    skipped_no_question = 0
    skipped_no_gt = 0

    for idx, item in enumerate(tqdm(test_items, desc="STEP9 running"), start=1):
        question = norm(item.get("question", ""))
        
        # 🔥 이미 처리된 질문이면 가볍게 건너뜁니다!
        if question in completed_questions:
            continue
            
        if not question:
            skipped_no_question += 1
            continue

        ground_truth = parse_ground_truth(item)
        if not ground_truth:
            skipped_no_gt += 1
            continue

        route_gold = parse_gold_route(item)

        try:
            route_pred = run_router(question)
        except Exception as e:
            print(f"❌ [Router Error] {question} | {e}")
            route_pred = "TEXT"

        if route_gold:
            route_total += 1
            if route_pred == route_gold:
                route_correct += 1

        try:
            text_res = run_text_rag(question)
        except Exception as e:
            print(f"❌ [Text RAG Error] {question} | {e}")
            text_res = {
                "answer": "생성 실패",
                "contexts": [],
                "debug_context": [],
                "latency_sec": 0.0,
            }

        try:
            vision_res = run_vision_rag(question)
        except Exception as e:
            print(f"❌ [Vision RAG Error] {question} | {e}")
            vision_res = {
                "answer": "생성 실패",
                "contexts": [],
                "debug_context": [],
                "latency_sec": 0.0,
            }

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
                route_pred=route_pred,
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
                route_pred=route_pred,
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
                route_pred=route_pred,
            )
        )

        analysis_rows.append(
            {
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
                "text_debug_context": text_res["debug_context"],
                "vision_debug_context": vision_res["debug_context"],
                "text_latency_sec": text_res["latency_sec"],
                "vision_latency_sec": vision_res["latency_sec"],
                "tv_latency_sec": chosen_res["latency_sec"],
                "source_file": item.get("source_file", ""),
                "context_used": item.get("context_used", ""),
            }
        )

        processed_count += 1

        if processed_count % SAVE_EVERY == 0:
            save_partial(
                processed_count=processed_count,
                total_count=len(test_items),
                always_text_rows=always_text_rows,
                always_vision_rows=always_vision_rows,
                tv_rag_rows=tv_rag_rows,
                analysis_rows=analysis_rows,
                route_correct=route_correct,
                route_total=route_total,
                text_latencies=text_latencies,
                vision_latencies=vision_latencies,
                tv_latencies=tv_latencies,
            )

    save_json(FINAL_ALWAYS_TEXT, always_text_rows)
    save_json(FINAL_ALWAYS_VISION, always_vision_rows)
    save_json(FINAL_TV_RAG, tv_rag_rows)
    save_json(FINAL_ANALYSIS, analysis_rows)

    route_acc = (route_correct / route_total) if route_total > 0 else 0.0
    avg_text_latency = sum(text_latencies) / len(text_latencies) if text_latencies else 0.0
    avg_vision_latency = sum(vision_latencies) / len(vision_latencies) if vision_latencies else 0.0
    avg_tv_latency = sum(tv_latencies) / len(tv_latencies) if tv_latencies else 0.0

    final_progress = {
        "status": "completed",
        "processed_count": processed_count,
        "total_count": len(test_items),
        "skipped_no_question": skipped_no_question,
        "skipped_no_gt": skipped_no_gt,
        "route_correct": route_correct,
        "route_total": route_total,
        "route_accuracy": round(route_acc, 4),
        "avg_text_latency": round(avg_text_latency, 4),
        "avg_vision_latency": round(avg_vision_latency, 4),
        "avg_tv_latency": round(avg_tv_latency, 4),
        "final_files": {
            "always_text": FINAL_ALWAYS_TEXT,
            "always_vision": FINAL_ALWAYS_VISION,
            "tv_rag": FINAL_TV_RAG,
            "analysis": FINAL_ANALYSIS,
        },
    }
    save_json(PROGRESS_PATH, final_progress)

    print("\n" + "=" * 80)
    print("🎉 [STEP9 평가 완료]")
    print("=" * 80)
    print(f"전체 데이터 수: {len(dataset)}")
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