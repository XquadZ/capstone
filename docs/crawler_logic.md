# Crawler & Ingestion Logic (2026-05)

공지 수집·증분·Milvus 적재 흐름입니다.

## 1. 크롤러 스크립트 (`crawler/`)

| 파일 | 역할 |
|------|------|
| `hoseo_spider.py` | Selenium 목록·상세 크롤 → `data/raw/{notice_id}/` |
| `incremental_notice_update.py` | 최신 N건 스캔 → Milvus 없는 것만 파이프라인 + 웹훅 |
| `crawl_all.py` | 전 카테고리·전 페이지 백필 (미등록만) |
| `delete_latest_notice_milvus.py` | 최근 청크 파일 기준 Milvus 삭제 (테스트) |
| `delete_for_test.py` | parent_id 지정 삭제 |
| `rule_spider.py` | 학칙·규정 수집 보조 |

## 2. Raw 데이터 구조

```text
data/raw/{notice_id}/
├── info.json          # title, date, url, content, attachments, images
├── images/
└── attachments/
```

`hoseo_spider`는 `schCategorycode`·`board_action`별 상세 URL 지원.

## 3. 공지 파이프라인 (오프라인 / 증분 공통)

```text
1. full_text_extractor.py
   raw → data/processed/integrated_text/{id}.txt
   (본문 + OCR + PDF/HWP)

2. local_slm_refiner.py (GPTRefiner)
   → data/processed/text/{id}.json
   metadata: year, category, major_category, target, entity

3. chunker.py (ContextualChunker)
   → data/processed/chunks/{id}_chunks.json

4. vector_db.py / MilvusIndexer
   → hoseo_notices (BGE-M3 dense+sparse)
```

## 4. 증분 업데이트 (`incremental_notice_update.py`)

- Milvus `parent_id` 집합 로드 (`query_iterator`)
- 목록 최신 **20건** 스캔 (기본 `scan_limit=20`)
- 미등록만: 크롤 → 정제 → 청크 → insert
- (선택) Spring 웹훅 `POST /api/notices/new`

```powershell
python -m crawler.incremental_notice_update --once
python -m crawler.incremental_notice_update   # 스케줄러
```

## 5. 전체 백필 (`crawl_all.py`)

- 카테고리 탭 `CTG_*` 자동 수집 또는 `--categories` 지정
- 페이지 순회, Milvus에 없는 `schIdx`만 배치 처리
- 크롬 세션 끊김 시 재시작, BGE-M3·OCR 배치 간 재사용

```powershell
python -m crawler.crawl_all --no-webhook
python -m crawler.crawl_all --max-new 100 --batch-size 10
```

## 6. LLM 정제 category (웹 탭 ≠ DB category)

게시판 탭(`CTG_...`)과 별개로, **GPTRefiner**가 허용 목록 중 하나를 `metadata.category`에 부여합니다.

- 허용 목록: 기존 `data/processed/text/*.json`에서 수집, 없으면 기본값  
  `공지사항, 학사, 장학, 취업, …`
- 채팅 `domain`(notice/rules)과는 **다른 축**입니다.

## 7. 학칙 파이프라인

```text
md_parser_pdf → rule_data_chunker → (local_slm_refiner_rule) → vector_db_rules → hoseo_rules_v1
```

## 8. 운영 주의

- `data/` Git 미추적 — 백업 필요
- OCR·LLM·임베딩은 GPU·API 비용 큼
- `crawl_all`은 장시간 실행 (재실행 시 Milvus 스킵)

## 9. 관련 문서

- [infra_setup.md](infra_setup.md)
- [system_arch.md](system_arch.md)
