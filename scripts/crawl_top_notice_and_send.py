"""공지사항 목록 맨 위 1건 크롤 → 로컬 저장 → AI SSE + Spring 웹훅 전송."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from selenium.webdriver.common.by import By

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from ai_engine.notice_alert_store import (
    normalize_alert_item,
    save_new_notice_alerts,
    to_spring_notice_item,
)
from crawler.hoseo_spider import HoseoRealCrawler
from crawler.incremental_notice_update import _parse_list_row


def main() -> int:
    # 1=맨 위, 2=두 번째 ...
    rank = 1
    if len(sys.argv) > 1:
        try:
            rank = max(1, int(sys.argv[1]))
        except ValueError:
            rank = 1

    board = HoseoRealCrawler.DEFAULT_BOARD_ACTION
    cat = HoseoRealCrawler.DEFAULT_CATEGORY_CODE

    crawler = HoseoRealCrawler()
    target = None
    ok = False
    try:
        crawler.set_board(board_action=board, sch_categorycode=cat)
        crawler.driver.get(crawler.list_url_template.format(1))
        rows = crawler.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        parsed = []
        for row in rows:
            item = _parse_list_row(row, board, cat)
            if item:
                parsed.append(item)
        if len(parsed) < rank:
            print(f"ERROR: 목록 유효 공지 {len(parsed)}건 — {rank}번째 없음")
            return 1
        target = parsed[rank - 1]
        print(f"PICK#{rank}: id={target['id']} date={target['date']}")
        print(f"TITLE: {target['title']}")

        ok = crawler.crawl_details(
            target["id"],
            target["title"],
            target["date"],
            sch_categorycode=target.get("schCategorycode"),
            board_action=target.get("board_action"),
        )
        print(f"CRAWL_OK: {ok}")
    finally:
        try:
            crawler.driver.quit()
        except Exception:
            pass

    if not target or not ok:
        print("ERROR: 상세 크롤 실패")
        return 1

    nid = target["id"]
    raw_path = PROJECT_ROOT / "data" / "raw" / nid / "info.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    processed_path = PROJECT_ROOT / "data" / "processed" / "text" / f"{nid}.json"
    refined: dict = {}
    if processed_path.exists():
        try:
            refined = json.loads(processed_path.read_text(encoding="utf-8")) or {}
        except Exception:
            refined = {}
    meta = refined.get("metadata") if isinstance(refined.get("metadata"), dict) else {}

    item = normalize_alert_item(
        {
            "notice_id": str(nid),
            "title": str(refined.get("title") or info.get("title") or target["title"]).strip(),
            "date": str(refined.get("date") or info.get("date") or target["date"]),
            "url": str(refined.get("url") or info.get("url") or ""),
            "category": str(meta.get("category") or refined.get("category") or "").strip(),
            "major_category": str(
                meta.get("major_category") or refined.get("major_category") or ""
            ).strip(),
            "target": str(meta.get("target") or "").strip(),
            "entity": str(meta.get("entity") or "").strip(),
        }
    )
    print(f"SEND: {item['display']}")
    print(
        f"  category={item['category']} major={item['major_category']} "
        f"sub={item.get('sub_category')}"
    )

    saved = save_new_notice_alerts([item])
    print(f"SAVED: count={saved.get('count')}")

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
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    payload = {
        "source": "crawler",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": 1,
        "items": [spring_item],
    }
    spr_ok = False
    try:
        spr = requests.post(webhook_url, headers=headers, json=payload, timeout=30)
        print(
            f"Spring item: category={spring_item['category']!r} "
            f"major={spring_item['major_category']!r}"
        )
        print(f"Spring: {spr.status_code} {spr.text[:300]}")
        spr_ok = spr.status_code in (200, 201)
    except Exception as e:
        print(f"Spring: FAIL {type(e).__name__}: {e}")

    return 0 if pub_ok and spr_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
