import os
import sys
import codecs

# 🌟 윈도우 & Conda run 환경 한글 깨짐 초강력 방지 (맨 위에 있어야 함)
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import requests  # 웹훅 발송용
from pymilvus import Collection, connections
from selenium.webdriver.common.by import By

try:
    import schedule
except ImportError as exc:
    raise ImportError(
        "schedule 패키지가 필요합니다. `pip install schedule` 후 다시 실행하세요."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from crawler.hoseo_spider import HoseoRealCrawler
from ai_engine.full_text_extractor import FullTextExtractor
from ai_engine.local_slm_refiner import GPTRefiner
from ai_engine.chunker import ContextualChunker
from ai_engine.vector_db import MilvusIndexer


def _log(message: str):
    print(message, flush=True)


def _discover_category_codes(crawler: HoseoRealCrawler, board_action: str) -> List[str]:
    """목록 페이지 fn_selectCategory / schCategorycode 수집 (crawl_all과 동일 방식)."""
    crawler.set_board(
        board_action=board_action,
        sch_categorycode=HoseoRealCrawler.DEFAULT_CATEGORY_CODE,
    )
    crawler.driver.get(crawler.list_url_template.format(1))
    time.sleep(0.8)
    html = crawler.driver.page_source or ""
    codes: Set[str] = set(re.findall(r"fn_selectCategory\('(CTG_[^']+)'\)", html))
    codes.update(re.findall(r'fn_selectCategory\("(CTG_[^"]+)"\)', html))
    for m in re.finditer(r"schCategorycode=([A-Za-z0-9_]+)", html):
        codes.add(m.group(1))
    codes.add(HoseoRealCrawler.DEFAULT_CATEGORY_CODE)
    return sorted(codes)


def _extract_notice_id_from_row(row) -> str:
    for sel in ("td.board-list-title a", "td[data-header='제목'] a"):
        try:
            link_el = row.find_element(By.CSS_SELECTOR, sel)
            href_val = link_el.get_attribute("href") or ""
            match = re.search(r"fn_viewData\('(\d+)'\)", href_val)
            if match:
                return match.group(1)
        except Exception:
            continue
    return ""


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


def _parse_list_row(
    row,
    board_action: str,
    category: str,
) -> Optional[Dict[str, str]]:
    notice_id = _extract_notice_id_from_row(row)
    if not notice_id:
        return None
    link_el = None
    title = ""
    for sel in ("td.board-list-title a", "td[data-header='제목'] a"):
        try:
            link_el = row.find_element(By.CSS_SELECTOR, sel)
            title = link_el.text.strip()
            break
        except Exception:
            continue
    if not link_el or not title:
        return None
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


class IncrementalNoticeUpdater:
    def __init__(
        self,
        collection_name: str = "hoseo_notices",
        scan_limit: int = 20  # 카테고리 탭마다 최근 N개(기본 20) 스캔
    ):
        self.collection_name = collection_name
        self.scan_limit = scan_limit

        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.integrated_dir = PROJECT_ROOT / "data" / "processed" / "integrated_text"
        self.processed_text_dir = PROJECT_ROOT / "data" / "processed" / "text"
        self.chunks_dir = PROJECT_ROOT / "data" / "processed" / "chunks"

        self.integrated_dir.mkdir(parents=True, exist_ok=True)
        self.processed_text_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.dday_dir = PROJECT_ROOT / "data" / "dday_data"
        self._dday_last_calendar_date: Optional[str] = None

    def _try_dday_calendar_today(self) -> None:
        """달력 '오늘'이 바뀌었거나 아직 오늘 요약을 안 했으면, 게시일=오늘 공지만 LLM 요약."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._dday_last_calendar_date == today:
            return
        try:
            from crawler.dday_digest import refresh_dday_for_post_date, prune_old_dday_files

            _log(f"📅 dday_data: 달력 기준 오늘({today}) 게시 공지 요약")
            refresh_dday_for_post_date(
                today,
                self.processed_text_dir,
                self.raw_dir,
                self.dday_dir,
            )
            prune_old_dday_files(self.dday_dir)
            self._dday_last_calendar_date = today
        except Exception as e:
            _log(f"⚠️ dday_data 오늘 요약 실패: {e}")

    def _try_dday_for_new_notices(self, crawled_ids: List[str]) -> None:
        if not crawled_ids:
            return
        try:
            from crawler.dday_digest import (
                _dates_for_notice_ids,
                prune_old_dday_files,
                update_dday_digests_for_crawl,
            )

            update_dday_digests_for_crawl(
                crawled_ids=crawled_ids,
                processed_text_dir=self.processed_text_dir,
                raw_dir=self.raw_dir,
                dday_dir=self.dday_dir,
            )
            today = datetime.now().strftime("%Y-%m-%d")
            dates = _dates_for_notice_ids(
                crawled_ids, self.processed_text_dir, self.raw_dir
            )
            if today in dates:
                self._dday_last_calendar_date = today
            else:
                prune_old_dday_files(self.dday_dir)
        except Exception as e:
            _log(f"⚠️ dday_data 요약 갱신 실패: {e}")

    def _build_notice_event_items(self, notice_ids: List[str]) -> List[Dict[str, str]]:
        """정제 JSON/RAW info를 기반으로 웹훅 전송용 공지 이벤트 목록 구성."""
        items: List[Dict[str, str]] = []
        for notice_id in notice_ids:
            notice_id = str(notice_id)
            processed_path = self.processed_text_dir / f"{notice_id}.json"
            raw_info_path = self.raw_dir / notice_id / "info.json"

            refined: Dict[str, str] = {}
            raw_info: Dict[str, str] = {}

            if processed_path.exists():
                try:
                    with open(processed_path, "r", encoding="utf-8") as f:
                        refined = json.load(f) or {}
                except Exception:
                    refined = {}

            if raw_info_path.exists():
                try:
                    with open(raw_info_path, "r", encoding="utf-8") as f:
                        raw_info = json.load(f) or {}
                except Exception:
                    raw_info = {}

            # GPTRefiner 산출물: category·major_category·target·entity·year 는 metadata 안에 있음
            meta = refined.get("metadata") if isinstance(refined.get("metadata"), dict) else {}

            def _m(key: str) -> str:
                return str(meta.get(key) or refined.get(key) or "")

            item = {
                "notice_id": notice_id,
                "title": str(refined.get("title") or raw_info.get("title") or ""),
                "date": str(refined.get("date") or raw_info.get("date") or ""),
                "url": str(refined.get("url") or raw_info.get("url") or ""),
                "category": _m("category"),
                "major_category": _m("major_category"),
                "target": _m("target"),
                "entity": _m("entity"),
            }
            items.append(item)
        return items

    def load_existing_parent_ids(self) -> Set[str]:
        connections.connect("default", host="localhost", port="19530")
        col = Collection(self.collection_name)
        col.load()

        result: Set[str] = set()
        # Milvus: 단일 query 에서 offset+limit 는 16384 이하여야 함 → 대량 청크 시 offset 페이징 불가
        iterator = None
        try:
            iterator = col.query_iterator(
                expr="pk >= 0",
                output_fields=["parent_id"],
                batch_size=2048,
            )
            while True:
                rows = iterator.next()
                if not rows:
                    break
                for row in rows:
                    pid = str(row.get("parent_id", "")).strip()
                    if pid:
                        result.add(pid)
        finally:
            if iterator is not None:
                try:
                    iterator.close()
                except Exception:
                    pass

        _log(f"📦 Milvus 기존 parent_id 로드 완료: {len(result)}개")
        return result

    def collect_new_notices(self, existing_ids: Set[str]) -> List[Dict[str, str]]:
        """공지사항·학사공지·장학공지 등 모든 카테고리 탭에서 각각 최근 scan_limit개 스캔."""
        crawler = HoseoRealCrawler()
        scanned_targets: List[Dict[str, str]] = []
        seen_ids: Set[str] = set()
        board_action = HoseoRealCrawler.DEFAULT_BOARD_ACTION

        try:
            categories = _discover_category_codes(crawler, board_action)
            _log(
                f"📂 카테고리 {len(categories)}개 × 최근 {self.scan_limit}개씩 목록 스캔"
            )

            for cat in categories:
                cat_count = 0
                page = 1
                crawler.set_board(board_action=board_action, sch_categorycode=cat)

                while cat_count < self.scan_limit:
                    crawler.driver.get(crawler.list_url_template.format(page))
                    try:
                        crawler.wait.until(
                            lambda d: d.find_elements(
                                By.CSS_SELECTOR, "table tbody tr"
                            )
                        )
                    except Exception:
                        break

                    rows = crawler.driver.find_elements(
                        By.CSS_SELECTOR, "table tbody tr"
                    )
                    if not rows:
                        break

                    added_in_this_page = 0
                    for row in rows:
                        try:
                            item = _parse_list_row(row, board_action, cat)
                            if not item:
                                continue
                            notice_id = item["id"]
                            if notice_id in seen_ids:
                                continue
                            seen_ids.add(notice_id)
                            scanned_targets.append(item)
                            cat_count += 1
                            added_in_this_page += 1
                            if cat_count >= self.scan_limit:
                                break
                        except Exception:
                            continue

                    if added_in_this_page == 0:
                        break
                    page += 1
                    time.sleep(0.4)

                _log(f"   · {cat}: {cat_count}건 수집")
        finally:
            crawler.driver.quit()

        new_targets = [t for t in scanned_targets if t["id"] not in existing_ids]

        _log(f"🔎 전체 스캔 {len(scanned_targets)}개 (중복 제거 후 ID 대조)")
        _log(f"🆕 DB 미존재 신규 공지 발견: {len(new_targets)}개")
        return new_targets

    def crawl_targets(self, targets: List[Dict[str, str]]) -> List[str]:
        if not targets:
            return []

        crawler = HoseoRealCrawler()
        crawled_ids: List[str] = []
        try:
            _log(f"🧪 이번 실행 크롤 처리 대상: {len(targets)}개 (scan_limit={self.scan_limit})")
            for target in targets:
                notice_id = target["id"]
                local_dir = self.raw_dir / notice_id
                
                if local_dir.exists():
                    _log(f"📁 로컬 데이터 존재 (재사용): {notice_id}")
                    crawled_ids.append(notice_id)
                    continue

                ok = crawler.crawl_details(
                    notice_id,
                    target["title"],
                    target["date"],
                    sch_categorycode=target.get("schCategorycode"),
                    board_action=target.get("board_action"),
                )
                if ok:
                    _log(f"✅ 상세 수집 완료: {notice_id}")
                    crawled_ids.append(notice_id)
                time.sleep(0.5)
        finally:
            crawler.driver.quit()
        return crawled_ids

    def build_integrated_text_for_ids(self, notice_ids: List[str], extractor: Optional[Any] = None):
        ext = extractor or FullTextExtractor()
        for notice_id in notice_ids:
            folder = self.raw_dir / str(notice_id)
            if not folder.exists(): continue
            integrated_parts = []
            info_file = folder / "info.json"
            if info_file.exists():
                with open(info_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    integrated_parts.append(f"### TITLE: {data.get('title', '')}")
                    integrated_parts.append(f"### DATE: {data.get('date', '')}")
                    integrated_parts.append(f"### URL: {data.get('url', '')}")
                    integrated_parts.append(f"### CONTENT:\n{data.get('content', '')}")
            
            img_dir = folder / "images"
            if img_dir.exists():
                img_texts = [ext.extract_ocr(f) for f in img_dir.glob("*") if f.suffix.lower() in [".jpg",".png",".jpeg"]]
                if img_texts: integrated_parts.append("### IMAGE_OCR:\n" + "\n".join(img_texts))

            attach_dir = folder / "attachments"
            if attach_dir.exists():
                attach_texts = []
                for f in attach_dir.glob("*"):
                    if f.suffix.lower() == ".pdf": attach_texts.append(f"<{f.name}>\n" + ext.extract_pdf(f))
                    elif f.suffix.lower() == ".hwp": attach_texts.append(f"<{f.name}>\n" + ext.extract_hwp(f))
                if attach_texts: integrated_parts.append("### ATTACHMENT_TEXT:\n" + "\n".join(attach_texts))

            out_txt = self.integrated_dir / f"{notice_id}.txt"
            out_txt.write_text("\n\n".join(integrated_parts), encoding="utf-8")

    def refine_integrated_text_for_ids(self, notice_ids: List[str], refiner: Optional[Any] = None):
        ref = refiner or GPTRefiner()
        for notice_id in notice_ids:
            src = self.integrated_dir / f"{notice_id}.txt"
            if not src.exists(): continue
            out = self.processed_text_dir / f"{notice_id}.json"
            res = ref.refine(src.read_text(encoding="utf-8"))
            if res:
                with open(out, "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=2)

    def chunk_refined_json_for_ids(self, notice_ids: List[str], chunker: Optional[Any] = None) -> List[Path]:
        ch = chunker or ContextualChunker()
        files = []
        for notice_id in notice_ids:
            src = self.processed_text_dir / f"{notice_id}.json"
            if not src.exists(): continue
            chunks = ch.process_file(src)
            if chunks:
                out = self.chunks_dir / f"{notice_id}_chunks.json"
                with open(out, "w", encoding="utf-8") as f: json.dump(chunks, f, ensure_ascii=False, indent=2)
                files.append(out)
        return files

    def insert_chunk_files(self, chunk_files: List[Path], indexer: Optional[Any] = None) -> int:
        if not chunk_files:
            return 0
        own_indexer = indexer is None
        if own_indexer:
            indexer = MilvusIndexer(collection_name=self.collection_name)
            indexer.collection = Collection(self.collection_name)
            indexer.collection.load()
        total = 0
        for f_path in chunk_files:
            with open(f_path, "r", encoding="utf-8") as f: chunks = json.load(f)
            if not chunks: continue
            data = [
                [c["chunk_id"] for c in chunks], [c["parent_id"] for c in chunks],
                [str(c["metadata"].get("year", "")) for c in chunks],
                [str(c["metadata"].get("category", "")) for c in chunks],
                [str(c["metadata"].get("target", "")) for c in chunks],
                [str(c["metadata"].get("entity", "")) for c in chunks],
                [c["chunk_text"] for c in chunks],
                indexer.model.encode([c["chunk_text"] for c in chunks], return_dense=True)["dense_vecs"].astype(np.float32),
                indexer.model.encode([c["chunk_text"] for c in chunks], return_sparse=True)["lexical_weights"]
            ]
            indexer.collection.insert(data)
            total += len(chunks)
        indexer.collection.flush()
        return total

    def run_once(self):
        _log(f"\n🕒 실행 시각: {datetime.now().strftime('%H:%M:%S')} (pid={os.getpid()})")
        existing_ids = self.load_existing_parent_ids()
        targets = self.collect_new_notices(existing_ids)
        
        if not targets:
            _log("✨ 모든 데이터가 최신 상태입니다.")
            self._try_dday_calendar_today()
            return

        crawled_ids = self.crawl_targets(targets)
        self.build_integrated_text_for_ids(crawled_ids)
        self.refine_integrated_text_for_ids(crawled_ids)
        chunk_files = self.chunk_refined_json_for_ids(crawled_ids)
        inserted = self.insert_chunk_files(chunk_files)
        event_items = self._build_notice_event_items(crawled_ids)
        
        _log(f"🚀 {len(targets)}개의 새로운 공지사항이 Milvus에 업데이트되었습니다.")

        # ==========================================================
        # 🔥 2. 백엔드(Spring) 서버로 웹훅 알림 쏘기
        # POST /api/notices/new · snake_case 바디 · X-API-Key
        # 로컬 Spring: NOTICE_EVENT_WEBHOOK_URL=http://localhost:8080/api/notices/new
        # ngrok 주소는 바뀔 수 있음 → 환경변수로 덮어쓰기 권장
        # ==========================================================
        default_webhook = "http://101.79.20.120/api/notices/new"
        webhook_url = os.getenv("NOTICE_EVENT_WEBHOOK_URL", default_webhook)
        api_key = os.getenv("NOTICE_EVENT_API_KEY", "hoseo-lens-secret-key")
        _log(f"🔗 웹훅 URL: {webhook_url}")

        try:
            headers = {
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            }
            # ngrok 무료 도메인: 브라우저 경고 페이지 회피(스크립트 호출 시 권장)
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

            if res.status_code in [200, 201]:
                detail = ""
                try:
                    body = res.json()
                    if isinstance(body, dict):
                        proc = body.get("processed")
                        st = body.get("status")
                        if proc is not None:
                            detail = f" status={st} processed={proc}"
                except Exception:
                    pass
                _log(f"🔔 백엔드 웹훅 전송 성공! (HTTP {res.status_code}){detail}")
            elif res.status_code == 401:
                _log(f"⚠️ 백엔드 웹훅 인증 실패(401): {res.text}")
            else:
                _log(f"⚠️ 백엔드 웹훅 전송 실패: {res.status_code} - {res.text}")
        except Exception as e:
            _log(f"❌ 백엔드 웹훅 연결 에러: {e}")

        self._try_dday_for_new_notices(crawled_ids)
        self._try_dday_calendar_today()


def run_scheduler():
    updater = IncrementalNoticeUpdater()
    is_running = False

    def safe_run_once():
        nonlocal is_running
        if is_running:
            _log("⏭ 이전 run_once가 아직 실행 중이라 이번 스케줄은 건너뜁니다.")
            return
        is_running = True
        try:
            updater.run_once()
        finally:
            is_running = False

    # 운영 스케줄: 매일 10:00 ~ 17:30, 30분 간격
    run_times = [
        "10:00", "10:30",
        "11:00", "11:30",
        "12:00", "12:30",
        "13:00", "13:30",
        "14:00", "14:30",
        "15:00", "15:30",
        "16:00", "16:30",
        "17:00", "17:30",
    ]
    for t in run_times:
        schedule.every().day.at(t).do(safe_run_once)

    _log("⏰ 스케줄러 시작: 매일 10:00~17:30, 30분 간격으로 실행합니다.")
    
    safe_run_once()

    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        IncrementalNoticeUpdater().run_once()
    else:
        run_scheduler()