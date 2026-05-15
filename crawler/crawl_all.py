"""
호서대 공지 전체 백필: 게시판의 모든 카테고리·페이지를 순회하며
Milvus에 없는 schIdx(= parent_id)만 incremental과 동일한 파이프라인으로 등록합니다.

목록/상세 크롬 세션 끊김 시 자동 재시작, 배치마다 BGE-M3·OCR·정제기 재사용으로 장시간 실행을 안정화합니다.
"""

from __future__ import annotations

import argparse
import codecs
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

import requests
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException
from selenium.webdriver.common.by import By

from ai_engine.chunker import ContextualChunker
from ai_engine.full_text_extractor import FullTextExtractor
from ai_engine.local_slm_refiner import GPTRefiner
from ai_engine.vector_db import MilvusIndexer
from crawler.hoseo_spider import HoseoRealCrawler
from crawler.incremental_notice_update import IncrementalNoticeUpdater, _log
from pymilvus import Collection


def _extract_notice_id_from_row(row) -> str:
    for sel in ("td.board-list-title a", "td[data-header='제목'] a"):
        try:
            link_el = row.find_element(By.CSS_SELECTOR, sel)
            href_val = link_el.get_attribute("href") or ""
            m = re.search(r"fn_viewData\('(\d+)'\)", href_val)
            if m:
                return m.group(1)
        except Exception:
            continue
    return ""


def _find_title_link(row):
    for sel in ("td.board-list-title a", "td[data-header='제목'] a"):
        try:
            return row.find_element(By.CSS_SELECTOR, sel)
        except Exception:
            continue
    return None


def _row_date_text(row) -> str:
    for sel in ("td[data-header='등록일자']", "td[data-header='등록일시']"):
        try:
            t = row.find_element(By.CSS_SELECTOR, sel).text.strip()
            if t:
                return t
        except Exception:
            pass
    try:
        cells = row.find_elements(By.CSS_SELECTOR, "td.txt-center.pc_view")
        if cells:
            return cells[-1].text.strip()
    except Exception:
        pass
    return ""


def parse_list_row(row, board_action: str, category: str) -> Optional[Dict[str, str]]:
    notice_id = _extract_notice_id_from_row(row)
    if not notice_id:
        return None
    link_el = _find_title_link(row)
    if not link_el:
        return None
    title = link_el.text.strip()
    date_text = _row_date_text(row)
    if len(date_text) <= 5 and date_text:
        date_text = f"{datetime.now().year}-{date_text}"
    return {
        "id": notice_id,
        "title": title,
        "date": date_text,
        "schCategorycode": category,
        "board_action": board_action,
    }


def _session_broken(exc: BaseException) -> bool:
    if isinstance(exc, InvalidSessionIdException):
        return True
    msg = str(exc).lower()
    return "invalid session" in msg or "disconnected" in msg or "not connected to devtools" in msg


class _ListBrowser:
    """목록 전용 크롬. 세션 끊기면 재생성."""

    def __init__(self, board_action: str):
        self.board_action = board_action
        self._c = HoseoRealCrawler(board_action=board_action)

    @property
    def crawler(self) -> HoseoRealCrawler:
        return self._c

    def recreate(self) -> None:
        _log("⚠️ 목록용 크롬 세션 끊김 → 재시작")
        try:
            self._c.driver.quit()
        except Exception:
            pass
        self._c = HoseoRealCrawler(board_action=self.board_action)

    def close(self) -> None:
        try:
            self._c.driver.quit()
        except Exception:
            pass


class _DetailBrowser:
    """상세 크롤 전용 크롬(배치 간 재사용)."""

    def __init__(self):
        self._c: Optional[HoseoRealCrawler] = None

    @property
    def crawler(self) -> HoseoRealCrawler:
        if self._c is None:
            self._c = HoseoRealCrawler()
        return self._c

    def recreate(self) -> None:
        _log("⚠️ 상세 크롬 세션 끊김 → 재시작")
        try:
            if self._c is not None:
                self._c.driver.quit()
        except Exception:
            pass
        self._c = HoseoRealCrawler()

    def close(self) -> None:
        if self._c is not None:
            try:
                self._c.driver.quit()
            except Exception:
                pass
            self._c = None


class _PipelineReuse:
    """OCR·정제·청크·BGE-M3 인덱서를 배치마다 재로드하지 않도록 유지."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._extractor: Optional[FullTextExtractor] = None
        self._refiner: Optional[GPTRefiner] = None
        self._chunker: Optional[ContextualChunker] = None
        self._indexer: Optional[MilvusIndexer] = None

    @property
    def extractor(self) -> FullTextExtractor:
        if self._extractor is None:
            self._extractor = FullTextExtractor()
        return self._extractor

    @property
    def refiner(self) -> GPTRefiner:
        if self._refiner is None:
            self._refiner = GPTRefiner()
        return self._refiner

    @property
    def chunker(self) -> ContextualChunker:
        if self._chunker is None:
            self._chunker = ContextualChunker()
        return self._chunker

    @property
    def indexer(self) -> MilvusIndexer:
        if self._indexer is None:
            self._indexer = MilvusIndexer(collection_name=self.collection_name)
            self._indexer.collection = Collection(self.collection_name)
            self._indexer.collection.load()
        return self._indexer


def _retry_list_op(list_browser: _ListBrowser, op, max_tries: int = 4):
    last_err: Optional[BaseException] = None
    for _ in range(max_tries):
        try:
            return op(list_browser.crawler)
        except (InvalidSessionIdException, WebDriverException) as e:
            last_err = e
            if _session_broken(e):
                list_browser.recreate()
                continue
            raise
    if last_err:
        raise last_err


def discover_category_codes_on(crawler: HoseoRealCrawler, board_action: str) -> List[str]:
    crawler.set_board(board_action=board_action, sch_categorycode=HoseoRealCrawler.DEFAULT_CATEGORY_CODE)
    crawler.driver.get(crawler.list_url_template.format(1))
    time.sleep(0.8)
    html = crawler.driver.page_source or ""
    codes: Set[str] = set(re.findall(r"fn_selectCategory\('(CTG_[^']+)'\)", html))
    codes.update(re.findall(r'fn_selectCategory\("(CTG_[^"]+)"\)', html))
    for m in re.finditer(r"schCategorycode=([A-Za-z0-9_]+)", html):
        codes.add(m.group(1))
    codes.add(HoseoRealCrawler.DEFAULT_CATEGORY_CODE)
    return sorted(codes)


def discover_category_codes(list_browser: _ListBrowser, board_action: str) -> List[str]:
    return _retry_list_op(list_browser, lambda c: discover_category_codes_on(c, board_action))


def scrape_list_page_safe(
    list_browser: _ListBrowser,
    board_action: str,
    category: str,
    page: int,
) -> List[Dict[str, str]]:
    try:
        return _retry_list_op(
            list_browser,
            lambda c: scrape_list_page(c, board_action, category, page),
        )
    except Exception as e:
        _log(f"⚠️ 목록 페이지 로드 실패 (cat={category} page={page}): {e}")
        return []


def scrape_list_page(
    crawler: HoseoRealCrawler,
    board_action: str,
    category: str,
    page: int,
) -> List[Dict[str, str]]:
    crawler.set_board(board_action=board_action, sch_categorycode=category)
    crawler.driver.get(crawler.list_url_template.format(page))
    try:
        crawler.wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "table tbody tr"))
    except Exception:
        return []
    time.sleep(0.4)
    rows = crawler.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    out: List[Dict[str, str]] = []
    for row in rows:
        try:
            item = parse_list_row(row, board_action, category)
            if item:
                out.append(item)
        except Exception:
            continue
    return out


def crawl_targets_with_category(
    updater: IncrementalNoticeUpdater,
    targets: List[Dict[str, str]],
    detail: Optional[_DetailBrowser] = None,
) -> List[str]:
    """상세 크롤. detail 을 넘기면 배치 간 브라우저 재사용(세션 끊기면 재생성)."""
    if not targets:
        return []

    own_detail = detail is None
    if own_detail:
        detail = _DetailBrowser()

    crawled_ids: List[str] = []
    try:
        _log(f"🧪 상세 크롤 대상: {len(targets)}개")
        for target in targets:
            notice_id = target["id"]
            local_dir = updater.raw_dir / notice_id

            if local_dir.exists():
                _log(f"📁 로컬 데이터 존재 (재사용): {notice_id}")
                crawled_ids.append(notice_id)
                continue

            ok = False
            for attempt in range(2):
                crawler = detail.crawler
                try:
                    ok = crawler.crawl_details(
                        notice_id,
                        target["title"],
                        target["date"],
                        sch_categorycode=target.get("schCategorycode"),
                        board_action=target.get("board_action"),
                    )
                    break
                except (InvalidSessionIdException, WebDriverException) as e:
                    if _session_broken(e) and attempt == 0:
                        detail.recreate()
                        continue
                    raise
            if ok:
                _log(f"✅ 상세 수집 완료: {notice_id}")
                crawled_ids.append(notice_id)
            time.sleep(0.5)
    finally:
        if own_detail:
            detail.close()
    return crawled_ids


def run_pipeline_for_batch(
    updater: IncrementalNoticeUpdater,
    targets: List[Dict[str, str]],
    *,
    send_webhook: bool,
    detail: _DetailBrowser,
    pipeline: _PipelineReuse,
) -> tuple[int, List[str]]:
    """통합 텍스트 → 정제 → 청크 → Milvus (모델·인덱서 재사용)."""
    crawled_ids = crawl_targets_with_category(updater, targets, detail=detail)
    updater.build_integrated_text_for_ids(crawled_ids, extractor=pipeline.extractor)
    updater.refine_integrated_text_for_ids(crawled_ids, refiner=pipeline.refiner)
    chunk_files = updater.chunk_refined_json_for_ids(crawled_ids, chunker=pipeline.chunker)
    inserted = updater.insert_chunk_files(chunk_files, indexer=pipeline.indexer)

    if send_webhook and crawled_ids:
        event_items = updater._build_notice_event_items(crawled_ids)
        if event_items:
            default_webhook = "https://wrecker-motivator-overall.ngrok-free.dev/api/notices/new"
            webhook_url = os.getenv("NOTICE_EVENT_WEBHOOK_URL", default_webhook)
            api_key = os.getenv("NOTICE_EVENT_API_KEY", "hoseo-lens-secret-key")
            try:
                headers = {
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                }
                if "ngrok" in (webhook_url or "").lower():
                    headers["ngrok-skip-browser-warning"] = "true"
                now_iso = datetime.now().isoformat(timespec="seconds")
                payload = {
                    "source": "crawler",
                    "generated_at": now_iso,
                    "count": len(event_items),
                    "items": event_items,
                }
                res = requests.post(webhook_url, headers=headers, json=payload, timeout=30)
                if res.status_code in (200, 201):
                    _log(f"🔔 백엔드 웹훅 전송 성공! (HTTP {res.status_code})")
                elif res.status_code == 401:
                    _log(f"⚠️ 백엔드 웹훅 인증 실패(401): {res.text}")
                else:
                    _log(f"⚠️ 백엔드 웹훅 전송 실패: {res.status_code} - {res.text}")
            except Exception as e:
                _log(f"❌ 백엔드 웹훅 연결 에러: {e}")

    return inserted, crawled_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="호서대 공지 전체 백필 → Milvus")
    parser.add_argument(
        "--board-action",
        default=HoseoRealCrawler.DEFAULT_BOARD_ACTION,
        help="BBSList.mbz 의 action 파라미터 (기본: 학교 공지 게시판)",
    )
    parser.add_argument(
        "--categories",
        default="",
        help="쉼표로 구분한 schCategorycode 목록. 비우면 목록 페이지에서 fn_selectCategory 로 자동 수집",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="한 번에 파이프라인 처리할 신규 공지 수")
    parser.add_argument("--max-pages", type=int, default=0, help="카테고리당 최대 페이지 (0=무제한)")
    parser.add_argument("--max-new", type=int, default=0, help="이번 실행에서 처리할 신규 상한 (0=무제한)")
    parser.add_argument("--dry-run", action="store_true", help="Milvus/크롤/정제 생략, 스캔만")
    parser.add_argument("--no-webhook", action="store_true", help="백엔드 웹훅 비활성화")
    parser.add_argument(
        "--collection",
        default="hoseo_notices",
        help="Milvus 컬렉션 이름 (incremental 과 동일 권장)",
    )
    args = parser.parse_args()

    board_action = args.board_action.strip()
    updater = IncrementalNoticeUpdater(collection_name=args.collection, scan_limit=10**9)

    _log(f"\n🕒 crawl_all 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    existing_ids: Set[str] = updater.load_existing_parent_ids()
    total_new_scanned = 0
    total_inserted = 0

    list_browser = _ListBrowser(board_action)
    detail: Optional[_DetailBrowser] = None
    pipeline: Optional[_PipelineReuse] = None
    try:
        if args.categories.strip():
            categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        else:
            categories = discover_category_codes(list_browser, board_action)
        _log(f"📂 카테고리 {len(categories)}개: {', '.join(categories[:12])}{' ...' if len(categories) > 12 else ''}")

        if not args.dry_run:
            detail = _DetailBrowser()
            pipeline = _PipelineReuse(args.collection)

        buffer: List[Dict[str, str]] = []

        def flush_buffer() -> None:
            nonlocal total_inserted, existing_ids, buffer
            if not buffer:
                return
            batch = buffer[:]
            buffer.clear()
            if args.dry_run:
                _log(f"🧪 [dry-run] 배치 {len(batch)}건 스킵 (첫 id={batch[0]['id']})")
                return
            if detail is None or pipeline is None:
                raise RuntimeError("내부 오류: detail/pipeline 이 초기화되지 않았습니다.")
            inserted, crawled = run_pipeline_for_batch(
                updater,
                batch,
                send_webhook=not args.no_webhook,
                detail=detail,
                pipeline=pipeline,
            )
            total_inserted += inserted
            existing_ids.update(crawled)
            _log(f"📥 배치 완료: 삽입 chunk 합계={inserted}, crawled_ids={len(crawled)}")

        stop_all = False
        for cat in categories:
            if stop_all or (args.max_new and total_new_scanned >= args.max_new):
                break
            page = 1
            while True:
                if stop_all or (args.max_new and total_new_scanned >= args.max_new):
                    break
                if args.max_pages and page > args.max_pages:
                    break

                rows = scrape_list_page_safe(list_browser, board_action, cat, page)
                if not rows:
                    _log(f"⏹ 카테고리 {cat} 페이지 {page}: 행 없음 → 다음 카테고리")
                    break

                for item in rows:
                    if args.max_new and total_new_scanned >= args.max_new:
                        stop_all = True
                        break
                    nid = item["id"]
                    if nid in existing_ids:
                        continue
                    if any(b["id"] == nid for b in buffer):
                        continue
                    buffer.append(item)
                    total_new_scanned += 1
                    if len(buffer) >= args.batch_size:
                        flush_buffer()

                page += 1
                time.sleep(0.45)

        flush_buffer()

    finally:
        list_browser.close()
        if detail is not None:
            detail.close()

    _log(
        f"🏁 crawl_all 종료: 스캔한 신규(미등록) 건수={total_new_scanned}, "
        f"삽입 chunk 합계={total_inserted}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
