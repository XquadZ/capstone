"""신규 공지 토스트용 프로세스 내 SSE 버스."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Set

_lock = threading.Lock()
_queues: Set[asyncio.Queue] = set()
_loop: asyncio.AbstractEventLoop | None = None


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    with _lock:
        _queues.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    with _lock:
        _queues.discard(q)


def publish(items: List[Dict[str, Any]]) -> int:
    """동기 컨텍스트(크롤 HTTP 핸들러)에서 호출. 구독자 수 반환."""
    if not items:
        return 0
    payload = {"type": "notice_new", "count": len(items), "items": items}
    raw = json.dumps(payload, ensure_ascii=False)

    with _lock:
        targets = list(_queues)
        loop = _loop

    if not targets:
        return 0

    def _put_all() -> None:
        for q in targets:
            try:
                q.put_nowait(raw)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(raw)
                except Exception:
                    pass

    if loop and loop.is_running():
        loop.call_soon_threadsafe(_put_all)
    else:
        _put_all()
    return len(targets)
