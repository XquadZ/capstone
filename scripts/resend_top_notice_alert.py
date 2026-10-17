"""이미 크롤된 맨 위 공지(기본 99522)를 AI SSE + Spring으로 재전송.

알림 category/display 는 대분류로 강제.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from ai_engine.notice_alert_store import (
    normalize_alert_item,
    save_new_notice_alerts,
    to_spring_notice_item,
)


def build_item(notice_id: str = "99522") -> dict:
    processed = PROJECT_ROOT / "data" / "processed" / "text" / f"{notice_id}.json"
    raw_path = PROJECT_ROOT / "data" / "raw" / notice_id / "info.json"

    refined: dict = {}
    raw_info: dict = {}
    if processed.exists():
        refined = json.loads(processed.read_text(encoding="utf-8")) or {}
    if raw_path.exists():
        raw_info = json.loads(raw_path.read_text(encoding="utf-8")) or {}
    if not refined and not raw_info:
        raise FileNotFoundError(f"no data for notice_id={notice_id}")

    meta = refined.get("metadata") if isinstance(refined.get("metadata"), dict) else {}

    def _m(key: str) -> str:
        return str(meta.get(key) or refined.get(key) or "").strip()

    return normalize_alert_item(
        {
            "notice_id": str(notice_id),
            "title": str(refined.get("title") or raw_info.get("title") or "").strip(),
            "date": str(refined.get("date") or raw_info.get("date") or "").strip(),
            "url": str(refined.get("url") or raw_info.get("url") or "").strip(),
            "category": _m("category"),
            "major_category": _m("major_category"),
            "target": _m("target"),
            "entity": _m("entity"),
        }
    )


def main() -> int:
    notice_id = sys.argv[1] if len(sys.argv) > 1 else "99522"
    item = build_item(notice_id)
    print(f"SEND: {item['display']}")
    print(f"  category={item['category']} major={item['major_category']} sub={item.get('sub_category')}")
    save_new_notice_alerts([item])

    publish_url = os.getenv(
        "NOTICE_ALERT_PUBLISH_URL", "http://127.0.0.1:8000/notices/alerts/publish"
    )
    webhook_url = os.getenv(
        "NOTICE_EVENT_WEBHOOK_URL",
        "https://wrecker-motivator-overall.ngrok-free.dev/api/notices/new",
    )
    api_key = os.getenv("NOTICE_EVENT_API_KEY", "hoseo-lens-secret-key")

    pub_ok = False
    try:
        pub = requests.post(
            publish_url,
            headers={"Content-Type": "application/json"},
            json={"items": [item]},
            timeout=15,
        )
        print(f"AI publish: {pub.status_code} {pub.text[:300]}")
        pub_ok = pub.status_code == 200
    except Exception as e:
        print(f"AI publish: FAIL {type(e).__name__}: {e}")

    spring_item = to_spring_notice_item(item)
    print(
        f"Spring item: category={spring_item['category']!r} "
        f"major_category={spring_item['major_category']!r}"
    )
    spr_ok = False
    try:
        spr = requests.post(
            webhook_url,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
            },
            json={
                "source": "crawler",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "count": 1,
                "items": [spring_item],
            },
            timeout=30,
        )
        print(f"Spring: {spr.status_code} {spr.text[:300]}")
        spr_ok = spr.status_code in (200, 201)
    except Exception as e:
        print(f"Spring: FAIL {type(e).__name__}: {e}")

    return 0 if pub_ok and spr_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
