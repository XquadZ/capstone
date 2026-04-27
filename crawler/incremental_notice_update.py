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

from crawler.hoseo_spider import HoseoRealCrawler
from ai_engine.full_text_extractor import FullTextExtractor
from ai_engine.local_slm_refiner import GPTRefiner
from ai_engine.chunker import ContextualChunker
from ai_engine.vector_db import MilvusIndexer


class IncrementalNoticeUpdater:
    def __init__(
        self,
        collection_name: str = "hoseo_notices",
        scan_limit: int = 20, # 최근 20개만 집중 스캔 (고속 테스트용)
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
                        # [핵심 변경] isdigit() 필터 제거: '공지' 등 고정글도 모두 수집
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
                
                if local_dir.exists():
                    print(f"📁 로컬 데이터 존재 (재사용): {notice_id}")
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

    def run_once(self):
        print(f"\n🕒 실행 시각: {datetime.now().strftime('%H:%M:%S')}")
        existing_ids = self.load_existing_parent_ids()
        targets = self.collect_new_notices(existing_ids)
        
        if not targets:
            print("✨ 모든 데이터가 최신 상태입니다.")
            return

        # 1. 크롤링 및 DB 삽입
        crawled_ids = self.crawl_targets(targets)
        self.build_integrated_text_for_ids(crawled_ids)
        self.refine_integrated_text_for_ids(crawled_ids)
        chunk_files = self.chunk_refined_json_for_ids(crawled_ids)
        inserted = self.insert_chunk_files(chunk_files)
        
        print(f"🚀 {len(targets)}개의 새로운 공지사항이 Milvus에 업데이트되었습니다.")

        # ==========================================================
        # 🔥 2. 백엔드(Spring) 서버로 웹훅 알림 쏘기
        # ==========================================================
        # 재화님이 추후 ngrok 주소를 주면 아래 "http://localhost:8080..." 부분을 수정하세요!
        webhook_url = os.getenv("NOTICE_EVENT_WEBHOOK_URL", "http://localhost:8080/api/notices/new")
        api_key = os.getenv("NOTICE_EVENT_API_KEY", "hoseo-lens-secret-key")

        try:
            # 헤더 구성 (X-API-Key 방식)
            headers = {
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            }
            # 보낼 데이터
            payload = {
                "message": "새 공지사항 업데이트 완료",
                "new_notice_count": len(targets)
            }
            
            # 알림 발송!
            res = requests.post(webhook_url, headers=headers, json=payload, timeout=5)
            
            if res.status_code in [200, 201]:
                print(f"🔔 백엔드(Spring) 웹훅 전송 성공! (상태코드: {res.status_code})")
            else:
                print(f"⚠️ 백엔드 웹훅 전송 실패: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ 백엔드 웹훅 연결 에러 (서버/ngrok 주소 확인 필요): {e}")


def run_scheduler():
    updater = IncrementalNoticeUpdater()
    # 테스트를 위해 2분 간격으로 설정
    schedule.every(2).minutes.do(updater.run_once)

    print("⏰ [테스트 모드] 스케줄러가 2분 간격으로 작동합니다. (터미널을 유지하세요)")
    
    # 시작하자마자 한 번 실행해서 확인
    updater.run_once()

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