import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import requests
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

from crawler.hoseo_spider import HoseoRealCrawler
from ai_engine.full_text_extractor import FullTextExtractor
from ai_engine.local_slm_refiner import GPTRefiner
from ai_engine.chunker import ContextualChunker
from ai_engine.vector_db import MilvusIndexer


class IncrementalNoticeUpdater:
    def __init__(
        self,
        collection_name: str = "hoseo_notices",
        scan_limit: int = 20, # 최근 20개만 집중 스캔
    ):
        self.collection_name = collection_name
        self.scan_limit = scan_limit

        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.integrated_dir = PROJECT_ROOT / "data" / "processed" / "integrated_text"
        self.processed_text_dir = PROJECT_ROOT / "data" / "processed" / "text"
        self.chunks_dir = PROJECT_ROOT / "data" / "processed" / "chunks"
        self.outbox_dir = PROJECT_ROOT / "data" / "processed" / "event_outbox"

        self.integrated_dir.mkdir(parents=True, exist_ok=True)
        self.processed_text_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)

        # 백엔드 이벤트 수신 엔드포인트 (예: http://localhost:8080/events/notices/new)
        self.event_webhook_url = os.getenv("NOTICE_EVENT_WEBHOOK_URL", "").strip()
        self.event_api_key = os.getenv("NOTICE_EVENT_API_KEY", "").strip()

    def load_existing_parent_ids(self) -> Set[str]:
        connections.connect("default", host="localhost", port="19530")
        col = Collection(self.collection_name)
        col.load()

        result: Set[str] = set()
        offset = 0
        batch_size = 16384
        while True:
            rows = col.query(
                expr="pk >= 0",
                output_fields=["parent_id"],
                limit=batch_size,
                offset=offset,
            )
            if not rows:
                break
            for row in rows:
                pid = str(row.get("parent_id", "")).strip()
                if pid:
                    result.add(pid)
            if len(rows) < batch_size:
                break
            offset += batch_size

        print(f"📦 Milvus 기존 parent_id 로드 완료: {len(result)}개")
        return result

    @staticmethod
    def _extract_notice_id_from_row(row) -> str:
        try:
            link_el = row.find_element(By.CSS_SELECTOR, "td.board-list-title a")
            href_val = link_el.get_attribute("href") or ""
            match = re.search(r"fn_viewData\('(\d+)'\)", href_val)
            if match:
                return match.group(1)
        except Exception:
            return ""
        return ""

    def collect_new_notices(self, existing_ids: Set[str]) -> List[Dict[str, str]]:
        crawler = HoseoRealCrawler()
        scanned_targets: List[Dict[str, str]] = []

        try:
            page = 1
            # 최근 공지 scan_limit개만 훑음
            while len(scanned_targets) < self.scan_limit:
                crawler.driver.get(crawler.list_url_template.format(page))
                crawler.wait.until(
                    lambda d: d.find_elements(By.CSS_SELECTOR, "table tbody tr")
                )
                rows = crawler.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                if not rows:
                    break

                for row in rows:
                    try:
                        # [변경점] 숫자인지 검사하는 isdigit() 필터를 제거하여 '공지' 텍스트도 수집함
                        date_cells = row.find_elements(By.CSS_SELECTOR, "td.txt-center.pc_view")
                        date_text = date_cells[-1].text.strip() if date_cells else ""
                        if len(date_text) <= 5:
                            date_text = f"{datetime.now().year}-{date_text}"

                        link_el = row.find_element(By.CSS_SELECTOR, "td.board-list-title a")
                        notice_id = self._extract_notice_id_from_row(row)
                        if not notice_id:
                            continue

                        scanned_targets.append(
                            {
                                "id": notice_id,
                                "title": link_el.text.strip(),
                                "date": date_text,
                            }
                        )
                        if len(scanned_targets) >= self.scan_limit:
                            break
                    except Exception:
                        continue

                if len(scanned_targets) >= self.scan_limit:
                    break
                page += 1
                time.sleep(0.5)
        finally:
            crawler.driver.quit()

        dedup_scanned = {t["id"]: t for t in scanned_targets}
        scanned_unique = list(dedup_scanned.values())
        # DB에 없는 녀석만 필터링
        new_targets = [t for t in scanned_unique if t["id"] not in existing_ids]

        print(f"🔎 최신 공지 {self.scan_limit}개 스캔 완료 (ID 대조 중...)")
        print(f"🆕 DB 미존재 신규 공지 발견: {len(new_targets)}개")
        return new_targets

    def crawl_targets(self, targets: List[Dict[str, str]]) -> List[str]:
        if not targets:
            return []

        crawler = HoseoRealCrawler()
        crawled_ids: List[str] = []
        try:
            for target in targets:
                notice_id = target["id"]
                local_dir = self.raw_dir / notice_id
                
                # 로컬에 이미 파일이 있으면 상세 수집 건너뛰고 파이프라인으로 보냄
                if local_dir.exists():
                    print(f"📁 로컬 데이터 존재: {notice_id}")
                    crawled_ids.append(notice_id)
                    continue

                ok = crawler.crawl_details(notice_id, target["title"], target["date"])
                if ok:
                    print(f"✅ 상세 수집 완료: {notice_id}")
                    crawled_ids.append(notice_id)
                time.sleep(0.5)
        finally:
            crawler.driver.quit()
        return crawled_ids

    # [이하 기존 전처리 로직 유지 - 코드 간결화를 위해 설명 생략함]
    def build_integrated_text_for_ids(self, notice_ids: List[str]):
        extractor = FullTextExtractor()
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
                img_texts = [extractor.extract_ocr(f) for f in img_dir.glob("*") if f.suffix.lower() in [".jpg",".png",".jpeg"]]
                if img_texts: integrated_parts.append("### IMAGE_OCR:\n" + "\n".join(img_texts))

            attach_dir = folder / "attachments"
            if attach_dir.exists():
                attach_texts = []
                for f in attach_dir.glob("*"):
                    if f.suffix.lower() == ".pdf": attach_texts.append(f"<{f.name}>\n" + extractor.extract_pdf(f))
                    elif f.suffix.lower() == ".hwp": attach_texts.append(f"<{f.name}>\n" + extractor.extract_hwp(f))
                if attach_texts: integrated_parts.append("### ATTACHMENT_TEXT:\n" + "\n".join(attach_texts))

            out_txt = self.integrated_dir / f"{notice_id}.txt"
            out_txt.write_text("\n\n".join(integrated_parts), encoding="utf-8")

    def refine_integrated_text_for_ids(self, notice_ids: List[str]):
        refiner = GPTRefiner()
        for notice_id in notice_ids:
            src = self.integrated_dir / f"{notice_id}.txt"
            if not src.exists(): continue
            out = self.processed_text_dir / f"{notice_id}.json"
            res = refiner.refine(src.read_text(encoding="utf-8"))
            if res:
                with open(out, "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=2)

    def chunk_refined_json_for_ids(self, notice_ids: List[str]) -> List[Path]:
        chunker = ContextualChunker()
        files = []
        for notice_id in notice_ids:
            src = self.processed_text_dir / f"{notice_id}.json"
            if not src.exists(): continue
            chunks = chunker.process_file(src)
            if chunks:
                out = self.chunks_dir / f"{notice_id}_chunks.json"
                with open(out, "w", encoding="utf-8") as f: json.dump(chunks, f, ensure_ascii=False, indent=2)
                files.append(out)
        return files

    def insert_chunk_files(self, chunk_files: List[Path]) -> int:
        if not chunk_files: return 0
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

    def _build_notice_event_items(self, notice_ids: List[str]) -> List[Dict[str, str]]:
        """신규 공지 이벤트 payload 아이템 생성."""
        items: List[Dict[str, str]] = []
        for notice_id in notice_ids:
            info_path = self.raw_dir / str(notice_id) / "info.json"
            refined_path = self.processed_text_dir / f"{notice_id}.json"

            info = {}
            refined = {}
            try:
                if info_path.exists():
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
            except Exception:
                info = {}
            try:
                if refined_path.exists():
                    with open(refined_path, "r", encoding="utf-8") as f:
                        refined = json.load(f)
            except Exception:
                refined = {}

            meta = (refined.get("metadata") or {}) if isinstance(refined, dict) else {}
            items.append(
                {
                    "notice_id": str(notice_id),
                    "title": str(info.get("title", "")).strip(),
                    "date": str(info.get("date", "")).strip(),
                    "url": str(info.get("url", "")).strip(),
                    "category": str(meta.get("category", "")).strip(),
                    "major_category": str(meta.get("major_category", "")).strip(),
                    "target": str(meta.get("target", "")).strip(),
                    "entity": str(meta.get("entity", "")).strip(),
                }
            )
        return items

    def _post_notice_events(self, event_items: List[Dict[str, str]]) -> bool:
        """신규 공지 이벤트를 백엔드로 전송. 실패 시 outbox 저장."""
        if not event_items:
            return True
        if not self.event_webhook_url:
            print("ℹ️ NOTICE_EVENT_WEBHOOK_URL 미설정: 이벤트 전송은 건너뜁니다.")
            return True

        payload = {
            "source": "ai_incremental_updater",
            "generated_at": datetime.now().isoformat(),
            "count": len(event_items),
            "items": event_items,
        }
        headers = {"Content-Type": "application/json"}
        if self.event_api_key:
            headers["X-API-Key"] = self.event_api_key

        try:
            resp = requests.post(
                self.event_webhook_url,
                json=payload,
                headers=headers,
                timeout=10,
            )
            if 200 <= resp.status_code < 300:
                print(f"📨 이벤트 전송 성공: {len(event_items)}건 -> {self.event_webhook_url}")
                return True
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outbox_path = self.outbox_dir / f"notice_event_{ts}.json"
            with open(outbox_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"⚠️ 이벤트 전송 실패: {exc}")
            print(f"📦 outbox 저장: {outbox_path}")
            return False

    def run_once(self):
        print(f"\n🕒 실행 시각: {datetime.now().strftime('%H:%M:%S')}")
        existing_ids = self.load_existing_parent_ids()
        targets = self.collect_new_notices(existing_ids)
        if not targets:
            print("✨ 모든 데이터가 최신 상태입니다.")
            return

        crawled_ids = self.crawl_targets(targets)
        self.build_integrated_text_for_ids(crawled_ids)
        self.refine_integrated_text_for_ids(crawled_ids)
        chunk_files = self.chunk_refined_json_for_ids(crawled_ids)
        inserted = self.insert_chunk_files(chunk_files)
        event_items = self._build_notice_event_items(crawled_ids)
        self._post_notice_events(event_items)
        print(f"🚀 {len(targets)}개의 새로운 공지사항이 Milvus에 업데이트되었습니다.")


def run_scheduler():
    updater = IncrementalNoticeUpdater()
    run_times = [
        "10:00", "10:30",
        "11:00", "11:30",
        "12:00", "12:30",
        "13:00", "13:30",
        "14:00", "14:30",
        "15:00",
    ]
    for t in run_times:
        schedule.every().day.at(t).do(updater.run_once)

    print("⏰ 스케줄러 시작: 매일 10:00 ~ 15:00, 30분 간격 증분 업데이트")
    print("   즉시 1회 실행: python crawler/incremental_notice_update.py --once")

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