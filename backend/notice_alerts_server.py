"""신규 공지 알림 보드 전용 가벼운 HTTP 서버 (GPU/RAG 미사용).

사용:
  python -m backend.notice_alerts_server
  브라우저: http://127.0.0.1:8010/
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERTS_PATH = PROJECT_ROOT / "data" / "notice_alerts" / "latest.json"
HOST = os.getenv("NOTICE_ALERTS_HOST", "0.0.0.0")
PORT = int(os.getenv("NOTICE_ALERTS_PORT", "8010"))


def _load() -> dict:
    if not ALERTS_PATH.exists():
        return {"updated_at": None, "count": 0, "items": []}
    try:
        with open(ALERTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {"updated_at": None, "count": 0, "items": []}
    except Exception:
        return {"updated_at": None, "count": 0, "items": []}


def _html(data: dict) -> str:
    updated = data.get("updated_at") or "-"
    items = data.get("items") if isinstance(data.get("items"), list) else []
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        display = str(it.get("display") or "").strip()
        if not display:
            cat = str(it.get("category") or "공지").strip()
            title = str(it.get("title") or "").strip()
            display = f"[{cat}] {title}"
        # 간단 XSS 방지
        display = (
            display.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        url = str(it.get("url") or "").strip()
        date = str(it.get("date") or "").strip()
        crawled = str(it.get("crawled_at") or "").strip()
        if url.startswith("http://") or url.startswith("https://"):
            link = f'<a href="{url}" target="_blank" rel="noopener">{display}</a>'
        else:
            link = display
        meta = " · ".join(x for x in [date, crawled] if x)
        rows.append(
            f'<li class="item"><div class="title">{link}</div>'
            f'<div class="meta">{meta}</div></li>'
        )
    body = "\n".join(rows) if rows else '<li class="empty">아직 신규 크롤 알림이 없습니다.</li>'
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="10" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>신규 공지 알림</title>
  <style>
    body {{
      margin: 0; font-family: "Noto Sans KR", system-ui, sans-serif;
      background: #0f1419; color: #e7ecf3; padding: 32px 20px;
    }}
    .wrap {{ max-width: 720px; margin: 0 auto; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 8px; }}
    .sub {{ color: #8b9bb4; margin-bottom: 24px; font-size: 0.9rem; }}
    ul {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }}
    .item {{
      background: #1a2332; border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px; padding: 14px 16px;
    }}
    .title a {{ color: #e7ecf3; text-decoration: none; }}
    .title a:hover {{ color: #3d8bfd; }}
    .meta {{ margin-top: 6px; color: #8b9bb4; font-size: 0.82rem; }}
    .empty {{ color: #8b9bb4; padding: 24px; text-align: center; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>신규 공지 알림</h1>
    <p class="sub">형식: [카테고리] 공지 제목 · 갱신 {updated} · 10초 새로고침</p>
    <ul>{body}</ul>
  </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        data = _load()
        if path in ("/", "/view", "/notices/alerts/view"):
            html = _html(data).encode("utf-8")
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path in ("/api", "/notices/alerts", "/json"):
            raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self._send(200, raw, "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Notice alerts board: http://127.0.0.1:{PORT}/")
    print(f"JSON: http://127.0.0.1:{PORT}/json")
    print(f"Watching: {ALERTS_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
