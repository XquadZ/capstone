# PROJECT MAP — 호서대 공지(Notice) / 학칙(Rules) 멀티모달 RAG

**저장소 루트 `capstone/` 기준** (2026-05-15 코드 스캔)

---

## 데이터 소스별 분류

| 구분 | 크롤·전처리 | Milvus | RAG 진입점 |
|------|-------------|--------|------------|
| **공지** | `crawler/hoseo_spider.py` → `full_text_extractor` → `local_slm_refiner` → `chunker` | `vector_db.py` → **`hoseo_notices`** | `rag_pipeline_notice.py`, TV-RAG `text_rag`/`vision_rag` (`domain=notice`) |
| **학칙** | `md_parser_pdf` → `rule_data_chunker` | `vector_db_rules.py` → **`hoseo_rules_v1`** | `rag_pipeline_rules.py`, TV-RAG (`domain=rules`) |

**TV-RAG 운영 API:** `backend/api_service.py` — `POST /ask` + `slm_router_node` → TEXT | VISION

---

## RAG 파이프라인

| 역할 | 경로 | 비고 |
|------|------|------|
| **운영 API** | `backend/api_service.py` | FastAPI, SAIFEX, notice/rules |
| 공지 검색·생성 | `ai_engine/rag_pipeline_notice.py` | `HoseoRAGPipeline` |
| 학칙 검색·생성 | `ai_engine/rag_pipeline_rules.py` | |
| 공지+학칙 혼합 검색 | `ai_engine/rag_pipeline.py` | 실험용 (단일 검색창) |
| Agentic TEXT | `AgenticRAG/nodes/text_rag.py` | `domain` 분기 |
| Agentic VISION | `AgenticRAG/nodes/vision_rag.py` | PDF/이미지 VLM |
| 라우터 | `AgenticRAG/nodes/router.py` | Gemma-2B LoRA → `experience/exp1/gemma_router_lora_v4` |
| LangGraph 프로토 | `AgenticRAG/graph/main_agent.py` | Router→Text/Vision→Critic |
| 인덱싱 | `vector_db.py`, `vector_db_rules.py` | BGE-M3 |
| 검색 테스트 | `ai_engine/search_test.py` | 공지만 |

---

## 크롤러 (`crawler/`)

| 파일 | 역할 |
|------|------|
| `hoseo_spider.py` | 목록·상세, `schCategorycode` 지원 |
| `incremental_notice_update.py` | 증분(20건)·스케줄·웹훅·Milvus insert |
| `crawl_all.py` | 전체 백필, 크롬 복구, 파이프라인 재사용 |
| `delete_latest_notice_milvus.py` | 테스트용 삭제 |
| `rule_spider.py` | 학칙 보조 |

---

## 연동 (Spring / Flutter)

```text
Flutter ──SSE──▶ Spring /api/chat/ask
Spring  ──JSON──▶ AI POST /ask  (domain: notice | rules)
```

- Spring: `category` → `domain` 매핑 필요
- 문서: `docs/api_spec.md`

---

## 논문 실험 — `experience/exp1/`

| step | 역할 |
|------|------|
| step1~4 | 데이터 검증·비교·SFT 분할 |
| step5 | `gemma_router_lora_v4` 학습 |
| step6~10 | 평가·RAGAS·E2E |

---

## API 키 사용처 (grep 참고)

| 게이트웨이 | 키 | 예시 |
|-----------|-----|------|
| SAIFEX | `SAIFEX_API_KEY` | `api_service`, `rag_pipeline_rules`, `vision_rag` |
| OpenAI | `OPENAI_API_KEY` | `local_slm_refiner`, `text_rag`(일부) |
| api.ahoseo.com | `AHOSEO_API_KEY` | `step3.5_rewrite_vision_intent.py` |

---

## Legacy / Deprecated

| 경로 | 사유 |
|------|------|
| `ai_engine/colpali.py`, `loader.py`, `chain.py` | ColPali/Byaldi 실험 |
| `ai_engine/sLM_RAG_pipeline.py` | 로컬 sLM RAG |
| `ai_engine/PDIS.py` | 지연 비교 실험 |
| `AgenticRAG/eval/pareto_plot,py` | 파일명 오타 |

---

## 문서

| 파일 | 내용 |
|------|------|
| `README.md` | 개요·빠른 시작 |
| `docs/system_arch.md` | 아키텍처 |
| `docs/api_spec.md` | AI/Spring API |
| `docs/infra_setup.md` | 실행 |
| `docs/crawler_logic.md` | 크롤·증분 |
| `docs/prompt_rules.md` | 프롬프트·2026년 규칙 |
| `docs/frontend_srs.md` | 프론트 SRS |
| `docs/progress.md` | 진행 현황 |

---

## 로컬 산출물 (Git 제외 권장)

`data/`, `volumes/`, `hoseo_router_gemma_2b*`, `experience/exp1/gemma_router_lora_v4/`, `temp_*_checkpoints/`
