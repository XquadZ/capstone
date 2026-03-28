import os
import json

TEST_DATA_PATH = "evaluation/datasets/sft_splits/test.jsonl"
RICH_DATA_PATH = "evaluation/datasets/final_intent_balanced_dataset.json"

def norm(s):
    if s is None:
        return ""
    return " ".join(str(s).strip().split())

def main():
    with open(RICH_DATA_PATH, "r", encoding="utf-8") as f:
        rich_data = json.load(f)

    q_to_gt_raw = {
        item.get("question", "").strip(): item.get("answer", "정답 없음")
        for item in rich_data
    }
    q_to_gt_norm = {
        norm(item.get("question", "")): item.get("answer", "정답 없음")
        for item in rich_data
    }

    total = 0
    raw_hit = 0
    norm_hit = 0
    misses = []

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item["input"].replace("질문: ", "").strip()
            total += 1

            if q in q_to_gt_raw:
                raw_hit += 1
            if norm(q) in q_to_gt_norm:
                norm_hit += 1
            if norm(q) not in q_to_gt_norm:
                misses.append(q)

    print("=" * 80)
    print("[GT 매핑 점검]")
    print("=" * 80)
    print(f"총 test 질문 수: {total}")
    print(f"raw exact match 수: {raw_hit}")
    print(f"normalized match 수: {norm_hit}")
    print(f"normalized miss 수: {len(misses)}")

    print("\n[miss 샘플 20개]")
    for q in misses[:20]:
        print("-", q)

    print("\n[rich_data 샘플 5개]")
    for item in rich_data[:5]:
        print("Q:", repr(item.get("question", "")))
        print("A:", repr(item.get("answer", ""))[:120])
        print("-" * 40)

if __name__ == "__main__":
    main()
