"""증분 크롤 신규 공지 알림 저장소 (로컬 JSON + 웹 표시용).

알림 payload의 category / display 대괄호는 항상 대분류(MAJOR_CATEGORIES) 중 하나.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERTS_PATH = PROJECT_ROOT / "data" / "notice_alerts" / "latest.json"
MAX_ALERTS = 50

# GPTRefiner._infer_major_category 와 동일 폐쇄집합
MAJOR_CATEGORIES = (
    "학사/수업",
    "장학/등록금/근로",
    "취업/진로/채용",
    "창업/산학",
    "국제/교류/연수",
    "비교과/특강/행사",
    "모집/선발/신청",
    "생활/복지",
    "안전/보안/시설",
    "연구/사업/행정",
    "기타/일반",
)
DEFAULT_MAJOR = "기타/일반"

_LEADING_BRACKET_RE = re.compile(r"^\[([^\]]+)\]\s*")


def strip_leading_title_tags(title: str) -> Tuple[str, str]:
    """제목 앞 [태그] 제거. (clean_title, 첫 태그 또는 '')."""
    text = (title or "").strip()
    first_tag = ""
    while True:
        m = _LEADING_BRACKET_RE.match(text)
        if not m:
            break
        if not first_tag:
            first_tag = m.group(1).strip()
        text = text[m.end() :].strip()
    return text or (title or "").strip(), first_tag


def format_notice_display(category: str, title: str) -> str:
    cat = (category or "").strip() or DEFAULT_MAJOR
    tit, _ = strip_leading_title_tags(title)
    tit = tit or "(제목 없음)"
    return f"[{cat}] {tit}"


def resolve_alert_major_category(
    major_category: str = "",
    category: str = "",
    title: str = "",
) -> str:
    """알림용 대분류: 허용 목록에 없으면 세부 category·제목으로 추론."""
    major = (major_category or "").strip()
    if major in MAJOR_CATEGORIES:
        return major

    from ai_engine.local_slm_refiner import GPTRefiner

    hint = " ".join(
        x for x in [(category or "").strip(), (title or "").strip()] if x
    )
    inferred = GPTRefiner._infer_major_category(hint)
    return inferred if inferred in MAJOR_CATEGORIES else DEFAULT_MAJOR


def normalize_alert_item(it: Dict[str, Any]) -> Dict[str, Any]:
    """알림/웹훅 item: category·major_category·display 를 대분류로 통일."""
    if not isinstance(it, dict):
        return {
            "notice_id": "",
            "category": DEFAULT_MAJOR,
            "major_category": DEFAULT_MAJOR,
            "title": "",
            "display": format_notice_display(DEFAULT_MAJOR, ""),
            "url": "",
            "date": "",
            "sub_category": "",
            "entity": "",
        }

    raw_title = str(it.get("title") or "").strip()
    title, title_tag = strip_leading_title_tags(raw_title)
    # 이미 대분류가 category에 들어온 경우 vs 세부 category
    raw_cat = str(it.get("category") or "").strip()
    raw_major = str(it.get("major_category") or "").strip()
    raw_sub = str(it.get("sub_category") or "").strip()

    if raw_cat in MAJOR_CATEGORIES and not raw_sub:
        sub_category = ""
        hint_cat = raw_cat
    else:
        sub_category = raw_sub or (raw_cat if raw_cat not in MAJOR_CATEGORIES else "")
        hint_cat = sub_category or raw_cat or title_tag

    major = resolve_alert_major_category(
        major_category=raw_major,
        category=hint_cat,
        title=raw_title,
    )
    display = format_notice_display(major, title)

    return {
        "notice_id": str(it.get("notice_id") or ""),
        "category": major,
        "major_category": major,
        "sub_category": sub_category,
        "title": title,
        "display": display,
        "url": str(it.get("url") or "").strip(),
        "date": str(it.get("date") or "").strip(),
        "target": str(it.get("target") or "").strip(),
        "entity": str(it.get("entity") or "").strip(),
    }


def to_spring_notice_item(it: Dict[str, Any]) -> Dict[str, Any]:
    """Spring /api/notices/new 용 item.

    백엔드가 major_category + '/' + category 로 합치는 경우가 있어,
    category 에만 대분류를 넣고 major_category 는 비운다.
    (둘 다 넣으면 '비교과/특강/행사/비교과/특강/행사' 처럼 중복됨)
    """
    norm = normalize_alert_item(it)
    return {
        "notice_id": norm["notice_id"],
        "title": norm["title"],
        "date": norm["date"],
        "url": norm["url"],
        "category": norm["category"],
        "major_category": "",
        "target": norm.get("target") or "",
        "entity": norm.get("entity") or "",
        "display": norm["display"],
    }


def _ensure_parent() -> None:
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_alerts() -> Dict[str, Any]:
    if not ALERTS_PATH.exists():
        return {"updated_at": None, "count": 0, "items": []}
    try:
        with open(ALERTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            return {"updated_at": None, "count": 0, "items": []}
        items = data.get("items") if isinstance(data.get("items"), list) else []
        return {
            "updated_at": data.get("updated_at"),
            "count": int(data.get("count") or len(items)),
            "items": items,
        }
    except Exception:
        return {"updated_at": None, "count": 0, "items": []}


def save_new_notice_alerts(event_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """신규 크롤 공지를 앞에 붙여 저장. display = [대분류] 제목."""
    now = datetime.now().isoformat(timespec="seconds")
    prev = load_alerts()
    prev_items = prev.get("items") if isinstance(prev.get("items"), list) else []

    new_rows: List[Dict[str, Any]] = []
    for it in event_items or []:
        if not isinstance(it, dict):
            continue
        norm = normalize_alert_item(it)
        new_rows.append(
            {
                "notice_id": norm["notice_id"],
                "category": norm["category"],
                "major_category": norm["major_category"],
                "sub_category": norm.get("sub_category") or "",
                "title": norm["title"],
                "display": norm["display"],
                "url": norm["url"],
                "date": norm["date"],
                "entity": norm.get("entity") or "",
                "crawled_at": now,
            }
        )

    # 같은 notice_id는 새 것으로 교체
    seen = {r["notice_id"] for r in new_rows if r.get("notice_id")}
    merged = new_rows + [
        r for r in prev_items if str(r.get("notice_id") or "") not in seen
    ]
    merged = merged[:MAX_ALERTS]

    payload = {
        "updated_at": now,
        "count": len(new_rows),
        "items": merged,
    }
    _ensure_parent()
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
