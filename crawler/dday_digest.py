"""
게시일(post date)별 공지 요약을 data/dday_data/ 에 저장 (최대 7일 보관).
증분 크롤 파이프라인(DB·정제·Milvus·웹훅)과 분리된 추가 기능입니다.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env", override=True)

DATE_FILE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")
MAX_RETENTION_DAYS = 7
CONTENT_PREVIEW_CHARS = 2500
SAIFEX_BASE_URL = "https://ahoseo.saifex.ai/v1"


def _log(message: str) -> None:
    print(message, flush=True)


def _normalize_post_date(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{2}-\d{2}", text):
        return f"{datetime.now().year}-{text}"
    if re.fullmatch(r"\d{2}\.\s*\d{2}\.\s*\d{2}\.", text.replace(" ", "")):
        parts = re.findall(r"\d+", text)
        if len(parts) >= 3:
            y, m, d = parts[0], parts[1], parts[2]
            if len(y) == 2:
                y = "20" + y
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    return None


def _openai_client() -> OpenAI:
    """dday 요약은 SAIFEX 우선 (OpenAI quota 이슈 회피)."""
    saifex_key = os.getenv("SAIFEX_API_KEY")
    if saifex_key:
        return OpenAI(api_key=saifex_key, base_url=SAIFEX_BASE_URL)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("SAIFEX_API_KEY 또는 OPENAI_API_KEY 가 필요합니다.")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _load_notice_record(
    notice_id: str,
    processed_text_dir: Path,
    raw_dir: Path,
) -> Optional[Dict[str, Any]]:
    processed_path = processed_text_dir / f"{notice_id}.json"
    raw_info_path = raw_dir / notice_id / "info.json"

    refined: Dict[str, Any] = {}
    raw_info: Dict[str, Any] = {}

    if processed_path.exists():
        try:
            refined = json.loads(processed_path.read_text(encoding="utf-8")) or {}
        except Exception:
            refined = {}

    if raw_info_path.exists():
        try:
            raw_info = json.loads(raw_info_path.read_text(encoding="utf-8")) or {}
        except Exception:
            raw_info = {}

    post_date = _normalize_post_date(
        str(refined.get("date") or raw_info.get("date") or "")
    )
    if not post_date:
        return None

    meta = refined.get("metadata") if isinstance(refined.get("metadata"), dict) else {}

    def _m(key: str) -> str:
        return str(meta.get(key) or refined.get(key) or "").strip()

    content = str(refined.get("refined_content") or raw_info.get("content") or "").strip()
    if len(content) > CONTENT_PREVIEW_CHARS:
        content = content[:CONTENT_PREVIEW_CHARS] + "\n...(이하 생략)"

    return {
        "notice_id": str(notice_id),
        "title": str(refined.get("title") or raw_info.get("title") or "").strip(),
        "date": post_date,
        "url": str(refined.get("url") or raw_info.get("url") or "").strip(),
        "category": _m("category"),
        "major_category": _m("major_category"),
        "target": _m("target"),
        "entity": _m("entity"),
        "content_preview": content,
    }


def collect_notices_for_post_date(
    post_date: str,
    processed_text_dir: Path,
    raw_dir: Path,
) -> List[Dict[str, Any]]:
    notices: List[Dict[str, Any]] = []
    if not processed_text_dir.exists():
        return notices

    for path in sorted(processed_text_dir.glob("*.json")):
        notice_id = path.stem
        record = _load_notice_record(notice_id, processed_text_dir, raw_dir)
        if record and record.get("date") == post_date:
            notices.append(record)

    notices.sort(key=lambda x: x.get("notice_id", ""))
    return notices


def _dates_for_notice_ids(
    notice_ids: List[str],
    processed_text_dir: Path,
    raw_dir: Path,
) -> Set[str]:
    dates: Set[str] = set()
    for notice_id in notice_ids:
        record = _load_notice_record(str(notice_id), processed_text_dir, raw_dir)
        if record and record.get("date"):
            dates.add(record["date"])
    return dates


def _generate_digest_with_llm(post_date: str, notices: List[Dict[str, Any]]) -> Dict[str, Any]:
    using_saifex = bool(os.getenv("SAIFEX_API_KEY"))
    client = _openai_client()
    if using_saifex:
        _log("[dday_data] SAIFEX(gpt-4o-mini) 요약 생성 중...")
    model = os.getenv("DDAY_DIGEST_MODEL", "gpt-4o-mini")

    catalog = []
    for i, n in enumerate(notices, start=1):
        catalog.append(
            f"### 공지 {i}\n"
            f"- notice_id: {n.get('notice_id', '')}\n"
            f"- 제목: {n.get('title', '')}\n"
            f"- 게시일: {n.get('date', '')}\n"
            f"- 분류: {n.get('category', '')} / {n.get('major_category', '')}\n"
            f"- 대상: {n.get('target', '')}\n"
            f"- 주관: {n.get('entity', '')}\n"
            f"- URL: {n.get('url', '')}\n"
            f"- 본문 요약용 텍스트:\n{n.get('content_preview', '')}\n"
        )

    system_prompt = (
        "당신은 호서대학교 공지 브리핑 작성자입니다. "
        "주어진 같은 날짜의 공지 목록을 바탕으로 학생이 빠르게 파악할 수 있게 정리하세요. "
        "추측하지 말고 제공된 정보만 사용하세요. "
        "반드시 JSON 객체 하나만 출력하세요."
    )
    user_prompt = (
        f"게시일: {post_date}\n"
        f"공지 건수: {len(notices)}\n\n"
        + "\n".join(catalog)
        + "\n\n[출력 JSON 스키마]\n"
        "{\n"
        '  "notices": [\n'
        "    {\n"
        '      "notice_id": "...",\n'
        '      "title": "...",\n'
        '      "category": "...",\n'
        '      "target": "...",\n'
        '      "entity": "...",\n'
        '      "brief": "2~3문장 핵심 요약"\n'
        "    }\n"
        "  ],\n"
        '  "digest_text": "해당 날짜 전체 브리핑(카테고리별 bullet, 마감·대상 강조)"\n'
        "}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=int(os.getenv("DDAY_DIGEST_MAX_TOKENS", "2500")),
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM 응답이 JSON 객체가 아닙니다.")
    return parsed


def _fallback_digest(post_date: str, notices: List[Dict[str, Any]]) -> Dict[str, Any]:
    items = []
    lines = [f"[{post_date}] 신규 공지 {len(notices)}건"]
    for n in notices:
        brief = (
            f"{n.get('title', '(제목 없음)')} — "
            f"대상: {n.get('target') or '미상'}, "
            f"주관: {n.get('entity') or '미상'}"
        )
        items.append(
            {
                "notice_id": n.get("notice_id", ""),
                "title": n.get("title", ""),
                "category": n.get("category", ""),
                "target": n.get("target", ""),
                "entity": n.get("entity", ""),
                "brief": brief,
            }
        )
        lines.append(f"- {brief}")
    return {"notices": items, "digest_text": "\n".join(lines)}


def write_dday_file(dday_dir: Path, post_date: str, payload: Dict[str, Any]) -> Path:
    dday_dir.mkdir(parents=True, exist_ok=True)
    out_json = dday_dir / f"{post_date}.json"
    out_txt = dday_dir / f"{post_date}.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    digest_text = str(payload.get("digest_text") or "").strip()
    out_txt.write_text(digest_text + "\n", encoding="utf-8")
    return out_json


def prune_old_dday_files(dday_dir: Path, keep_days: int = MAX_RETENTION_DAYS) -> int:
    if not dday_dir.exists():
        return 0
    cutoff = (datetime.now().date() - timedelta(days=keep_days - 1))
    removed = 0
    for path in list(dday_dir.glob("*.json")) + list(dday_dir.glob("*.txt")):
        match = DATE_FILE_PATTERN.match(path.name)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def refresh_dday_for_post_date(
    post_date: str,
    processed_text_dir: Path,
    raw_dir: Path,
    dday_dir: Path,
) -> Optional[Path]:
    notices = collect_notices_for_post_date(post_date, processed_text_dir, raw_dir)
    if not notices:
        _log(f"📭 dday_data: {post_date} — 해당 게시일 공지 없음 (파일 생성 안 함)")
        return None

    try:
        llm_result = _generate_digest_with_llm(post_date, notices)
    except Exception as e:
        _log(f"⚠️ dday_data LLM 실패 ({post_date}), 규칙 기반 fallback: {e}")
        llm_result = _fallback_digest(post_date, notices)

    payload = {
        "date": post_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notice_count": len(notices),
        "notices": llm_result.get("notices") or [],
        "digest_text": str(llm_result.get("digest_text") or "").strip(),
        "source_notice_ids": [n.get("notice_id") for n in notices],
    }

    out_path = write_dday_file(dday_dir, post_date, payload)
    _log(f"📋 dday_data 저장: {out_path.name} ({len(notices)}건)")

    today = datetime.now().strftime("%Y-%m-%d")
    if post_date == today:
        send_dday_digest_webhook(post_date, dday_dir)

    return out_path


def _default_dday_webhook_url() -> str:
    explicit = os.getenv("NOTICE_DDAY_WEBHOOK_URL", "").strip()
    if explicit:
        return explicit
    base = os.getenv(
        "NOTICE_EVENT_WEBHOOK_URL",
        "http://101.79.20.120/api/notices/new",
    ).strip()
    if base.endswith("/new"):
        return base[: -len("/new")] + "/dday"
    return base.rstrip("/") + "/dday"


def send_dday_digest_webhook(post_date: str, dday_dir: Path) -> bool:
    """
    달력 '오늘'과 동일한 게시일 요약(JSON)만 Spring으로 전송.
  POST /api/notices/dday · X-API-Key (공지 웹훅과 동일)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if post_date != today:
        return False

    json_path = Path(dday_dir) / f"{post_date}.json"
    if not json_path.exists():
        _log(f"⚠️ dday 웹훅 건너뜀: {json_path.name} 없음")
        return False

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"⚠️ dday 웹훅 건너뜀: JSON 읽기 실패 ({e})")
        return False

    webhook_url = _default_dday_webhook_url()
    api_key = os.getenv("NOTICE_EVENT_API_KEY", "hoseo-lens-secret-key")

    try:
        import requests

        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }
        if "ngrok" in webhook_url.lower():
            headers["ngrok-skip-browser-warning"] = "true"

        body = {
            "source": "crawler",
            "date": payload.get("date", post_date),
            "generated_at": payload.get("generated_at"),
            "notice_count": payload.get("notice_count", 0),
            "notices": payload.get("notices") or [],
            "digest_text": payload.get("digest_text") or "",
            "source_notice_ids": payload.get("source_notice_ids") or [],
        }

        res = requests.post(webhook_url, headers=headers, json=body, timeout=30)
        if res.status_code in (200, 201):
            _log(f"🔔 dday_data 웹훅 전송 성공 ({post_date}, HTTP {res.status_code})")
            return True
        if res.status_code == 401:
            _log(f"⚠️ dday_data 웹훅 인증 실패(401)")
        else:
            _log(f"⚠️ dday_data 웹훅 실패: HTTP {res.status_code} - {res.text[:200]}")
    except Exception as e:
        _log(f"❌ dday_data 웹훅 연결 오류: {e}")
    return False


def update_dday_digests_for_crawl(
    crawled_ids: List[str],
    processed_text_dir: Path,
    raw_dir: Path,
    dday_dir: Path,
) -> None:
    """신규 공지가 처리된 경우에만, 해당 게시일 전체 요약을 LLM으로 갱신합니다."""
    if not crawled_ids:
        return

    dday_dir = Path(dday_dir)
    dates = _dates_for_notice_ids(crawled_ids, processed_text_dir, raw_dir)
    if not dates:
        _log("⚠️ dday_data: 게시일을 확인할 수 있는 공지가 없어 요약을 건너뜁니다.")
        return

    _log(f"📅 dday_data (신규 공지): 갱신 게시일 {', '.join(sorted(dates))}")
    for post_date in sorted(dates):
        refresh_dday_for_post_date(post_date, processed_text_dir, raw_dir, dday_dir)

    removed = prune_old_dday_files(dday_dir, MAX_RETENTION_DAYS)
    if removed:
        _log(f"🗑️ dday_data: {MAX_RETENTION_DAYS}일 초과 파일 {removed}개 삭제")


