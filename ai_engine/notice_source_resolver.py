"""공지 parent_id(schIdx) → 제목·원문 URL 조회 (processed/raw JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

_meta_cache: Dict[str, Dict[str, str]] = {}


def get_notice_meta(parent_id: str) -> Dict[str, str]:
    pid = str(parent_id or "").strip()
    if not pid or pid == "unknown":
        return {
            "notice_id": pid,
            "title": "공지",
            "url": "",
            "category": "",
        }
    if pid in _meta_cache:
        return _meta_cache[pid]

    title, url, category = "", "", ""
    processed_path = PROCESSED_TEXT_DIR / f"{pid}.json"
    if processed_path.is_file():
        try:
            data = json.loads(processed_path.read_text(encoding="utf-8"))
            title = str(data.get("title") or "").strip()
            url = str(data.get("url") or "").strip()
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            category = str(meta.get("category") or "").strip()
        except Exception:
            pass

    if not title or not url:
        info_path = RAW_DIR / pid / "info.json"
        if info_path.is_file():
            try:
                raw = json.loads(info_path.read_text(encoding="utf-8"))
                title = title or str(raw.get("title") or "").strip()
                url = url or str(raw.get("url") or "").strip()
            except Exception:
                pass

    out = {
        "notice_id": pid,
        "title": title or f"호서대 공지 ({pid})",
        "url": url,
        "category": category,
    }
    _meta_cache[pid] = out
    return out


def notice_source_item(parent_id: str) -> Dict[str, Any]:
    m = get_notice_meta(parent_id)
    return {
        "doc_id": m["notice_id"],
        "title": m["title"],
        "file_url": m["url"],
        "category": m.get("category") or "",
        "source_type": "notice",
    }


def notice_sources_from_parent_ids(parent_ids: List[str]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    items: List[Dict[str, Any]] = []
    for pid in parent_ids:
        pid = str(pid or "").strip()
        if not pid or pid == "unknown" or pid in seen:
            continue
        seen.add(pid)
        items.append(notice_source_item(pid))
    return items


def notice_context_label(parent_id: str) -> str:
    """LLM 컨텍스트용 — 제목 중심 (번호만 노출하지 않음)."""
    m = get_notice_meta(parent_id)
    cat = m.get("category") or "일반"
    return f"「{m['title']}」 ({cat})"


def notice_context_header(parent_id: str) -> str:
    """LLM 컨텍스트용 — 제목·분류·원문 URL."""
    m = get_notice_meta(parent_id)
    cat = m.get("category") or "일반"
    lines = [f"「{m['title']}」 ({cat})"]
    if m.get("url"):
        lines.append(f"원문 URL: {m['url']}")
    return "\n".join(lines)


RELATED_NOTICE_SECTION_TITLE = "관련된 공지"
MAX_RELATED_NOTICES_IN_FOOTER = 3
MAX_NOTICE_SOURCES_IN_API = 3


def limit_notice_sources(
    sources: List[Dict[str, Any]], max_items: int = MAX_NOTICE_SOURCES_IN_API
) -> List[Dict[str, Any]]:
    """API·하단 목록용 — 검색 순서 유지, 중복 URL 제거 후 상한."""
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in sources:
        if len(out) >= max_items:
            break
        url = str(item.get("file_url") or item.get("url") or "").strip()
        key = url or str(item.get("doc_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def format_related_notices_footer(
    sources: List[Dict[str, Any]],
    answer: str = "",
    max_items: int = MAX_RELATED_NOTICES_IN_FOOTER,
) -> str:
    """답변 하단 '관련된 공지' — 본문에 이미 나온 URL 제외, 최대 max_items건."""
    body = answer or ""
    lines: List[str] = []
    seen_urls: set[str] = set()
    for item in limit_notice_sources(sources, max_items=len(sources)):
        if len(lines) >= max_items:
            break
        url = str(item.get("file_url") or item.get("url") or "").strip()
        title = str(item.get("title") or "공지").strip()
        if not url or url in seen_urls or url in body:
            continue
        seen_urls.add(url)
        lines.append(f"- {title}\n  {url}")
    if not lines:
        return ""
    return RELATED_NOTICE_SECTION_TITLE + "\n" + "\n".join(lines)


def append_notice_links_to_answer(
    answer: str,
    sources: List[Dict[str, Any]],
    max_related: int = MAX_RELATED_NOTICES_IN_FOOTER,
) -> str:
    """답변 본문 하단에 관련 공지 링크를 붙입니다 (최대 max_related건)."""
    body = (answer or "").strip()
    if RELATED_NOTICE_SECTION_TITLE in body or "📎 공지 원문 링크" in body:
        return body
    footer = format_related_notices_footer(sources, answer=body, max_items=max_related)
    if not footer:
        return body
    return body + "\n\n" + footer
