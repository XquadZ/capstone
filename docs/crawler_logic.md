# Crawler & Ingestion Logic (Current)

이 문서는 현재 저장소의 전처리 흐름을 기준으로 작성되었다.  
실제 크롤러 코드(`crawler/`)는 저장소에 포함되지 않았고, 수집 완료된 데이터가 `data/raw/`에 있다고 가정한다.

## 1. 입력 데이터 전제

`data/raw/{notice_id}/` 구조를 기본으로 사용한다.

```text
data/raw/
└── {notice_id}/
    ├── info.json
    ├── images/
    └── attachments/
```

`info.json`에는 제목/날짜/본문/저장된 첨부 경로가 포함된다고 가정한다.

## 2. 공지사항 텍스트 통합 파이프라인

### Step 1) 통합 추출
- 스크립트: `ai_engine/full_text_extractor.py`
- 처리:
  - `info.json` 본문 결합
  - 이미지 OCR (`easyocr`)
  - 첨부파일 텍스트 추출 (`pdf`, `hwp`)
- 출력: `data/processed/integrated_text/{notice_id}.txt`

### Step 2) 구조화 정제
- 스크립트: `ai_engine/local_slm_refiner.py`
- 처리:
  - 통합 txt를 LLM으로 정제
  - 메타데이터(`year`, `category`, `target`, `entity`) 추출
- 출력: `data/processed/text/{notice_id}.json`

### Step 3) 청킹
- 스크립트: `ai_engine/chunker.py`
- 처리:
  - 문단 단위 분할
  - 문서 글로벌 컨텍스트 태깅
- 출력: `data/processed/chunks/{notice_id}_chunks.json`

### Step 4) 벡터 적재
- 스크립트: `ai_engine/vector_db.py`
- 처리:
  - BGE-M3 dense+sparse 임베딩 생성
  - Milvus 컬렉션(`hoseo_notices`) 인덱싱

## 3. 학칙/규정 파이프라인

### Step 1) PDF -> Markdown(+page tag)
- `ai_engine/md_parser_pdf.py`

### Step 2) Markdown -> Chunk JSON
- `ai_engine/rule_data_chunker.py`

### Step 3) 텍스트 교정(옵션)
- `ai_engine/local_slm_refiner_rule.py`

### Step 4) Milvus 적재
- `ai_engine/vector_db_rules.py` -> `hoseo_rules_v1`

## 4. 운영 시 주의사항

- `data/`는 Git 미추적이므로 백업/동기화 정책이 필요하다.
- OCR/LLM 단계는 비용과 시간이 큰 구간이므로 배치 실행을 권장한다.
- 전처리 중단 대비를 위해 스크립트별 산출물을 단계적으로 저장한다.