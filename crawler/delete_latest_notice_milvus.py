"""
data/processed/chunks 수정 시각이 가장 최근인 공지 1건을
Milvus hoseo_notices 에서 parent_id 기준으로 삭제합니다.

  python crawler/delete_latest_notice_milvus.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pymilvus import Collection, connections  # noqa: E402


def main():
    chunks_dir = PROJECT_ROOT / "data" / "processed" / "chunks"
    files = sorted(
        chunks_dir.glob("*_chunks.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        print("[ERROR] data/processed/chunks 에 *_chunks.json 이 없습니다.")
        sys.exit(1)

    latest = files[0]
    parent_id = latest.stem.replace("_chunks", "")
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)

    print(f"[INFO] latest chunk file: {latest.name}")
    print(f"[INFO] parent_id: {parent_id}")
    print(f"[INFO] mtime: {mtime.isoformat()}")

    connections.connect("default", host="localhost", port="19530")
    col = Collection("hoseo_notices")
    col.load()

    expr = f'parent_id == "{parent_id}"'
    res = col.delete(expr)
    col.flush()

    print(f"[DELETE] expr: {expr}")
    print(f"[DELETE] result: {res}")
    print(f"[STATS] num_entities (ref): {col.num_entities}")


if __name__ == "__main__":
    main()
