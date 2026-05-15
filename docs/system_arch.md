# System Architecture (2026-05)

현재 저장소(`capstone/`) 코드 기준 아키텍처 요약입니다.

## 1. 전체 연동 구조 (운영)

```text
[Flutter 앱]
    │  SSE   POST /api/chat/ask
    │        body: user_id, session_id, question, category
    ▼
[Spring 백엔드]  (hoseo-chatbot)
    │  JSON  POST {rag.server.url}/ask
    │        body: question, domain, use_tv_rag
    ▼
[AI 서버]  backend/api_service.py  (FastAPI, port 8000)
    │  TV-RAG: Gemma 라우터 → TEXT | VISION 노드
    ▼
[Milvus]  hoseo_notices | hoseo_rules_v1
```

| 구간 | 프로토콜 | 비고 |
|------|----------|------|
| 앱 ↔ Spring | **SSE** (`chunk`, `sources`, `[DONE]`) | ngrok 등으로 Spring 노출 |
| Spring ↔ AI | **JSON** (비스트리밍) | AI는 `answer` 한 번에 반환 |
| AI 내부 | 동기 RAG + LLM | Spring이 단어 단위로 SSE 재조립 |

## 2. 핵심 구성

| 레이어 | 기술 | 역할 |
|--------|------|------|
| Frontend | Flutter (연동) | SSE 채팅 UI |
| Backend | Spring Boot | 세션·DB·SSE 브릿지 |
| AI API | **FastAPI** `backend/api_service.py` | `/ask`, `/health` |
| Vector DB | Milvus 2.4 (`docker-compose.yml`) | BGE-M3 hybrid |
| Embedding | `BAAI/bge-m3` | dense + sparse |
| Reranker | `BAAI/bge-reranker-v2-m3` | 재정렬 |
| Router | Gemma-2B + LoRA (`AgenticRAG/nodes/router.py`) | TEXT / VISION |
| LLM | gpt-4o-mini (SAIFEX 게이트웨이) | 생성·Vision |

## 3. TV-RAG (`POST /ask`, `use_tv_rag: true`)

1. `domain` 정규화: `notice` | `rules` 만 허용
2. `slm_router_node` → `TEXT` 또는 `VISION`
3. **TEXT** (`AgenticRAG/nodes/text_rag.py`)
   - `domain=notice` → `hoseo_notices` (`rag_pipeline_notice`)
   - `domain=rules` → `hoseo_rules_v1` (`rag_pipeline_rules`)
4. **VISION** (`AgenticRAG/nodes/vision_rag.py`)
   - 동일 `domain` 분기, PDF/공지 이미지 + VLM
5. JSON 응답: `answer`, `route`, `contexts`, `sources`, `latency_sec`, `meta`

`use_tv_rag: false` 시 라우터 없이 단일 텍스트 RAG만 (`notice` / `rules` 각각).

## 4. 데이터 파이프라인

### 4.1 공지 (Notice)

```text
crawler/hoseo_spider.py
  → data/raw/{notice_id}/
  → full_text_extractor → integrated_text
  → local_slm_refiner (GPT) → processed/text
  → chunker → processed/chunks
  → vector_db.py → Milvus hoseo_notices
```

**증분 운영:** `crawler/incremental_notice_update.py` (스케줄 또는 `--once`)  
**전체 백필:** `crawler/crawl_all.py` (카테고리·페이지 순회, Milvus 미등록만)

### 4.2 학칙 (Rules)

```text
md_parser_pdf → rule_data_chunker → vector_db_rules → hoseo_rules_v1
```

## 5. Milvus 컬렉션

| 컬렉션 | 용도 | 적재 |
|--------|------|------|
| `hoseo_notices` | 공지 청크 | `vector_db.py` |
| `hoseo_rules_v1` | 학칙 청크 | `vector_db_rules.py` |

대량 `parent_id` 조회 시 `query_iterator` 사용 (`incremental_notice_update.load_existing_parent_ids`).

## 6. 디렉터리 스냅샷

```text
capstone/
├── backend/api_service.py      # FastAPI TV-RAG (Spring 연동)
├── ai_engine/                  # RAG·전처리·인덱싱
├── crawler/                    # hoseo_spider, incremental, crawl_all
├── AgenticRAG/                 # router, text_rag, vision_rag
├── evaluation/                 # 벤치마크·RAGAS
├── docs/                       # 본 문서군
├── docker-compose.yml
└── experience/exp1/gemma_router_lora_v4/  # 라우터 LoRA (기본)
```

## 7. 레거시·실험

- `ai_engine/rag_pipeline.py` — 공지+학칙 **단일 검색** (혼재 실험용, 운영 TV-RAG와 별도)
- `ai_engine/colpali.py`, `sLM_RAG_pipeline.py` — 대안 스택
- `AgenticRAG/graph/main_agent.py` — LangGraph 프로토타입

## 8. 관련 문서

- [api_spec.md](api_spec.md) — AI `/ask` + Spring SSE 계약
- [infra_setup.md](infra_setup.md) — 실행·환경변수
- [crawler_logic.md](crawler_logic.md) — 수집·증분
- [PROJECT_MAP.md](../PROJECT_MAP.md) — 파일 맵
