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
    """
    공지 증분 업데이트(기존 파이프라인 완전 재사용):
      1) Milvus hoseo_notices의 기존 parent_id 로드
      2) 목록 크롤링 중 "기존 ID 연속 5개" 만나면 조기 종료
      3) 신규 공지 상세 수집
      4) full_text_extractor 로 integrated_text 생성
      5) local_slm_refiner 로 processed/text JSON 생성
      6) chunker 로 chunks 생성
      7) vector_db 로 Milvus insert
    """

    def __init__(
        self,
        collection_name: str = "hoseo_notices",
        consecutive_existing_stop: int = 5,
        max_pages: int = 20,
    ):
        self.collection_name = collection_name
        self.consecutive_existing_stop = consecutive_existing_stop
        self.max_pages = max_pages

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
            # 최신 일반공지 200개를 먼저 전부 훑은 뒤 DB 대조
            while len(scanned_targets) < 200:
                crawler.driver.get(crawler.list_url_template.format(page))
                crawler.wait.until(
                    lambda d: d.find_elements(By.CSS_SELECTOR, "table tbody tr")
                )
                rows = crawler.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                if not rows:
                    break

                for row in rows:
                    try:
                        # 번호: td.pc_view (일반 공지는 숫자)
                        num_cells = row.find_elements(By.CSS_SELECTOR, "td.pc_view")
                        if not num_cells:
                            continue
                        num_text = num_cells[0].text.strip()
                        if not num_text.isdigit():  # 고정공지/비정상 행 제외
                            continue

                        # 날짜: td.txt-center.pc_view
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
                        if len(scanned_targets) >= 200:
                            break
                    except Exception:
                        continue

                if len(scanned_targets) >= 200:
                    break
                page += 1
                time.sleep(0.8)
        finally:
            crawler.driver.quit()

        # 수집한 200개(또는 그 이하)에서 중복 제거 후 기존 DB와 대조
        dedup_scanned = {t["id"]: t for t in scanned_targets}
        scanned_unique = list(dedup_scanned.values())
        new_targets = [t for t in scanned_unique if t["id"] not in existing_ids]

        print(
            f"🔎 최신 일반공지 스캔 완료: {len(scanned_targets)}행 / "
            f"고유 ID {len(scanned_unique)}개"
        )
        print(f"🆕 DB 미존재 신규 공지: {len(new_targets)}개")
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
                    # 로컬은 있고 DB만 비어 있을 수 있으므로 후속 파이프라인 대상에는 포함
                    print(f"📁 로컬 폴더 존재(상세수집 생략): {notice_id}")
                    crawled_ids.append(notice_id)
                    continue

                ok = crawler.crawl_details(notice_id, target["title"], target["date"])
                if ok:
                    print(f"✅ 수집 완료: {notice_id} | {target['title'][:36]}")
                    crawled_ids.append(notice_id)
                else:
                    print(f"❌ 수집 실패: {notice_id}")
                time.sleep(0.4)
        finally:
            crawler.driver.quit()
        return crawled_ids

    # ---------- 기존 전처리 파이프라인 재사용 (동일 로직) ----------
    def build_integrated_text_for_ids(self, notice_ids: List[str]) -> List[Path]:
        """
        full_text_extractor.process_all의 내부 포맷을 그대로 유지한 ID 단위 버전.
        출력: data/processed/integrated_text/<id>.txt
        """
        if not notice_ids:
            return []

        extractor = FullTextExtractor()
        created: List[Path] = []

        for notice_id in notice_ids:
            folder = self.raw_dir / str(notice_id)
            if not folder.exists():
                continue

            integrated_parts = []

            # 1) info.json
            info_file = folder / "info.json"
            if info_file.exists():
                with open(info_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    integrated_parts.append(f"### TITLE: {data.get('title', '')}")
                    integrated_parts.append(f"### DATE: {data.get('date', '')}")
                    integrated_parts.append(f"### URL: {data.get('url', '')}")
                    integrated_parts.append(f"### CONTENT:\n{data.get('content', '')}")

            # 2) images OCR
            img_dir = folder / "images"
            if img_dir.exists():
                img_texts = []
                for img_file in img_dir.glob("*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        img_texts.append(extractor.extract_ocr(img_file))
                if img_texts:
                    integrated_parts.append("### IMAGE_OCR:\n" + "\n".join(img_texts))

            # 3) attachments
            attach_dir = folder / "attachments"
            if attach_dir.exists():
                attach_texts = []
                for attach_file in attach_dir.glob("*"):
                    ext = attach_file.suffix.lower()
                    if ext == ".pdf":
                        attach_texts.append(
                            f"<{attach_file.name}>\n" + extractor.extract_pdf(attach_file)
                        )
                    elif ext == ".hwp":
                        attach_texts.append(
                            f"<{attach_file.name}>\n" + extractor.extract_hwp(attach_file)
                        )
                if attach_texts:
                    integrated_parts.append(
                        "### ATTACHMENT_TEXT:\n" + "\n".join(attach_texts)
                    )

            final_text = "\n\n".join(integrated_parts)
            out_txt = self.integrated_dir / f"{notice_id}.txt"
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(final_text)
            created.append(out_txt)
            print(f"🧾 통합 텍스트 생성: {out_txt.name}")

        return created

    def refine_integrated_text_for_ids(self, notice_ids: List[str]) -> List[Path]:
        """
        local_slm_refiner.process_directory의 핵심 로직(GPTRefiner.refine + JSON 저장)을
        ID 단위로 동일하게 수행.
        """
        if not notice_ids:
            return []

        refiner = GPTRefiner()
        created_jsons: List[Path] = []

        for notice_id in notice_ids:
            src_txt = self.integrated_dir / f"{notice_id}.txt"
            if not src_txt.exists():
                continue
            out_json = self.processed_text_dir / f"{notice_id}.json"

            try:
                raw_text = src_txt.read_text(encoding="utf-8")
                result = refiner.refine(raw_text)
                if not result:
                    print(f"❌ 정제 실패: {src_txt.name}")
                    continue
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                created_jsons.append(out_json)
                print(f"✨ 정제 JSON 생성: {out_json.name}")
            except Exception as exc:
                print(f"❌ 정제 중 예외: {notice_id} | {exc}")

        return created_jsons

    def chunk_refined_json_for_ids(self, notice_ids: List[str]) -> List[Path]:
        chunker = ContextualChunker()
        created_chunk_files: List[Path] = []

        for notice_id in notice_ids:
            src = self.processed_text_dir / f"{notice_id}.json"
            if not src.exists():
                continue

            try:
                chunks = chunker.process_file(src)
                if not chunks:
                    continue
                out_path = self.chunks_dir / f"{notice_id}_chunks.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                created_chunk_files.append(out_path)
                print(f"🧩 청킹 완료: {out_path.name} ({len(chunks)} chunks)")
            except Exception as exc:
                print(f"❌ 청킹 실패: {notice_id} | {exc}")

        return created_chunk_files

    def insert_chunk_files(self, chunk_files: List[Path]) -> int:
        if not chunk_files:
            return 0

        indexer = MilvusIndexer(collection_name=self.collection_name)
        # 증분 insert 이므로 create_collection()은 호출하지 않음
        indexer.collection = Collection(self.collection_name)
        indexer.collection.load()

        total_inserted = 0
        for file_path in chunk_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                if not chunks:
                    continue

                chunk_ids = [c["chunk_id"] for c in chunks]
                parent_ids = [c["parent_id"] for c in chunks]
                texts = [c["chunk_text"] for c in chunks]
                years = [str(c["metadata"].get("year", "")) for c in chunks]
                categories = [str(c["metadata"].get("category", "")) for c in chunks]
                targets = [str(c["metadata"].get("target", "")) for c in chunks]
                entities = [str(c["metadata"].get("entity", "")) for c in chunks]

                embeddings = indexer.model.encode(
                    texts, return_dense=True, return_sparse=True, batch_size=12
                )
                dense_vecs = embeddings["dense_vecs"].astype(np.float32)

                data = [
                    chunk_ids,
                    parent_ids,
                    years,
                    categories,
                    targets,
                    entities,
                    texts,
                    dense_vecs,
                    embeddings["lexical_weights"],
                ]
                indexer.collection.insert(data)
                total_inserted += len(chunks)
                print(f"📥 Milvus insert 완료: {file_path.name} ({len(chunks)} chunks)")
            except Exception as exc:
                print(f"❌ Milvus insert 실패: {file_path.name} | {exc}")

        indexer.collection.flush()
        return total_inserted

    def run_once(self):
        print("\n" + "=" * 70)
        print(f"🚀 증분 업데이트 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        existing_ids = self.load_existing_parent_ids()
        targets = self.collect_new_notices(existing_ids)
        crawled_ids = self.crawl_targets(targets)

        if not crawled_ids:
            print("ℹ️ 신규 수집 대상이 없습니다. 종료합니다.")
            return

        self.build_integrated_text_for_ids(crawled_ids)
        self.refine_integrated_text_for_ids(crawled_ids)
        chunk_files = self.chunk_refined_json_for_ids(crawled_ids)
        inserted = self.insert_chunk_files(chunk_files)

        print("-" * 70)
        print(f"✅ 수집 대상: {len(targets)} / 상세수집(또는 로컬재사용): {len(crawled_ids)}")
        print(f"✅ 신규 청킹 파일: {len(chunk_files)}")
        print(f"✅ Milvus 신규 삽입 청크 수: {inserted}")
        print("=" * 70)


def run_scheduler():
    updater = IncrementalNoticeUpdater()

    schedule.every().day.at("10:00").do(updater.run_once)
    schedule.every().day.at("13:00").do(updater.run_once)
    schedule.every().day.at("15:00").do(updater.run_once)

    print("⏰ 스케줄러 시작: 매일 10:00 / 13:00 / 15:00 증분 업데이트")
    print("   즉시 1회 실행: python crawler/incremental_notice_update.py --once")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        action="store_true",
        help="스케줄 대기 없이 즉시 1회 실행",
    )
    args = parser.parse_args()

    if args.once:
        IncrementalNoticeUpdater().run_once()
    else:
        run_scheduler()
