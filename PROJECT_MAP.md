# PROJECT MAP — 호서대 학칙(Regulation) / 공지사항(Notice) 멀티모달 RAG

본 문서는 **저장소 루트 `capstone/` 기준**(2026-03-28 시점 코드 스캔)으로 디렉터리·파일 역할을 정리한 맵입니다.

---

## 데이터 소스별 분류 (Notice vs Regulation)

| 구분 | 의미 | 크롤/전처리 | 청크·임베딩·Milvus | RAG·생성 진입점 |
|------|------|-------------|-------------------|----------------|
| **공지사항 (Notice)** | `hoseo_spider` 등으로 수집한 행정 공지 | `crawler/hoseo_spider.py` → `ai_engine/full_text_extractor.py` → `local_slm_refiner.py` → `chunker.py` | `ai_engine/vector_db.py` → 컬렉션 **`hoseo_notices`** (BGE-M3) | `ai_engine/rag_pipeline.py`(통합), **`ai_engine/rag_pipeline_notice.py`**(`HoseoRAGPipeline`, 공지 전용), `ai_engine/rag_pipeline.py`의 멀티컬렉션 검색, `ai_engine/sLM_RAG_pipeline.py`·`search_test.py`(테스트/대안) |
| **학칙·규정 (Regulation)** | PDF 학칙 | `crawler/rule_spider.py`(보조), `ai_engine/md_parser_pdf.py` | `ai_engine/rule_data_chunker.py` → (`local_slm_refiner_rule.py` 선택) → `vector_db_rules.py` → **`hoseo_rules_v1`** | **`ai_engine/rag_pipeline_rules.py`**, `evaluation/scripts/run_benchmark_rules_*.py` |

**주의 (Agentic 그래프):** `AgenticRAG/nodes/text_rag.py`·`vision_rag.py`는 `ai_engine.rag_pipeline_rules`의 **`retrieve_documents`만** import 합니다. 즉 **에이전트 TV-RAG의 텍스트·비전 검색 단계는 현재 Milvus `hoseo_rules_v1` 기준**이며, 공지(`hoseo_notices`)까지 합친 통합 검색은 **`ai_engine/rag_pipeline.py`**에서 구현됩니다(별도 진입 필요).

---

## RAG 파이프라인 식별

### 텍스트 검색 — Milvus + BGE-M3 (+ Reranker)

| 역할 | 경로 | 비고 |
|------|------|------|
| 학칙 단일 컬렉션 검색·리랭크·(스트리밍) 생성 | `ai_engine/rag_pipeline_rules.py` | `BGEM3FlagModel('BAAI/bge-m3')`, `FlagReranker`, Milvus `hoseo_rules_v1` |
| 공지 전용 클래스 | `ai_engine/rag_pipeline_notice.py` | 동일 BGE-M3 + Milvus `hoseo_notices`, LLM은 `OPENAI_API_KEY` |
| 학칙+공지 멀티 컬렉션 | `ai_engine/rag_pipeline.py` | 두 컬렉션 하이브리드 후 스키마 매핑 |
| 인덱스 구축 | `ai_engine/vector_db.py`(공지), `ai_engine/vector_db_rules.py`(학칙) | |
| 단독 검색 테스트 | `ai_engine/search_test.py` | 공지 컬렉션 |
| **Agentic Text 노드** | `AgenticRAG/nodes/text_rag.py` | 위 **`rag_pipeline_rules.retrieve_documents`** 호출 후 **공식 OpenAI** 클라이언트로 `gpt-4o-mini` 생성 |

### 비전 검색 — VLM + PDF 페이지 스니핑 (Image Snipping)

| 역할 | 경로 | 비고 |
|------|------|------|
| **Agentic Vision 노드** | `AgenticRAG/nodes/vision_rag.py` | `retrieve_documents`로 상위 히트 → `pdf2image.convert_from_path`로 **타깃·인접 페이지** JPEG → base64 → **VLM**(`gpt-4o-mini` 멀티모달) 호출 |
| 그래프 조립·실행 | `AgenticRAG/graph/main_agent.py` | Router → `text_rag` \| `vision_rag` → Critic |

**ColPali / Byaldi 비전 인덱스(별계열):** `ai_engine/colpali.py`, `ai_engine/loader.py`(모드별), `ai_engine/chain.py`(Byaldi 인덱스 내 doc_id 추적). TV-RAG 메인 경로와는 분리되어 있음 → 아래 **[Legacy]** 참고.

---

## 논문 실험 — `experience/exp1/` (step1 ~ step9)

| 파일 | 역할 |
|------|------|
| `step1_verify_dataset.py` | 공지 QA JSON(`notice_qa_2000_target.json`) 스키마·길이 검증 후 `notice_qa_2000_verified.json` 저장 |
| `step2_generate_comparison.py` | 공지 파이프라인 `HoseoRAGPipeline`으로 **Text vs Vision(이미지)** 답변 대량 생성·비교 JSON 산출; API는 **SAIFEX 우선 + OpenAI 폴백** |
| `step2_refill_errors.py` | 비교/생성 단계에서 실패한 인덱스만 **공식 OpenAI**로 재시도하여 데이터 보수 |
| `step3_agreement_filtering.py` | Ground Truth vs Text/Vision 답을 **SAIFEX** LLM 심판으로 라벨링(`TEXT` / `VISION` / `REJECT`)·골든셋 정제 |
| `step3.5_rewrite_intent.py` | 규칙 기반으로 질문에 **시각적 의도 문구** 삽입·밸런스 → `final_intent_balanced_dataset.json` |
| `step3.5_rewrite_vision_intent.py` | LLM으로 시각 질의 재작성; **`https://api.ahoseo.com/v1`** + `AHOSEO_API_KEY` (SAIFEX `saifex.ai`와 **엔드포인트 다름**) |
| `step4_prepare_sft_data.py` | 라우팅 라벨을 SFT 포맷으로 변환 후 **8:1:1** 분할 → `evaluation/datasets/sft_splits/*.jsonl` |
| `step5_train_gemma_router.py` | Gemma-2B + LoRA **SFT** 학습 → `experience/exp1/gemma_router_lora_v4/` |
| `step6_eval_router.py` | V4 LoRA 로드, `test.jsonl`에서 분류 성능·혼동행렬·리포트 |
| `step7_check_raw_data.py` | **[Legacy 경로]** `gemma_router_lora_stratified` 어댑터로 테스트 10건만 **생성 문구** 확인 (현행 V4 파이프라인과 체크포인트명 불일치 가능) |
| `step8_zero_shot_test.py` | 학습에 없던 표현의 질문 쌍으로 V4 라우터 **일반화** 스모크 테스트 |
| `step9_end_to_end_eval.py` | `text_rag_node` / `vision_rag_node` 직접 호출 + **공식 OpenAI**로 답변 생성, A/B/C 조건 벤치마크 JSON 저장 |

**같은 폴더 참고 (step10):** `step10_ragas_eval.py` — 벤치마크 JSON에 **RAGAS** 지표 적용; 평가 LLM은 `SAIFEX_API_KEY` 또는 `OPENAI_API_KEY` 선택 가능.

---

## 사용 중인 API 정보 (일괄 수정용)

### 공식 OpenAI (`OPENAI_API_KEY`, 기본 `base_url` 생략 = `api.openai.com`)

- `ai_engine/local_slm_refiner.py`, `ai_engine/rag_pipeline_notice.py`, `ai_engine/vision_processor.py`
- `evaluation/scripts/generate_qa.py`, `evaluation/scripts/run_eval.py`, `evaluation/scripts/run_eval_rules.py`
- `AgenticRAG/nodes/text_rag.py` (생성 단계)
- `experience/exp1/step2_refill_errors.py`, `step9_end_to_end_eval.py`
- `step2_generate_comparison.py` — **폴백** 클라이언트

### SAIFEX / Ahoseo 게이트웨이 (`SAIFEX_API_KEY`, `base_url=https://ahoseo.saifex.ai/v1`)

- `ai_engine/rag_pipeline.py`, `ai_engine/rag_pipeline_rules.py`
- `AgenticRAG/nodes/vision_rag.py`, `AgenticRAG/nodes/critic.py` (클라이언트는 `rag_pipeline_rules`에서 재사용)
- `evaluation/scripts/generate_qa_rules.py`, `run_benchmark_rules_text.py`, `run_benchmark_rules_pdf.py`, `run_eval_reverse.py`
- `experience/exp1/step2_generate_comparison.py`(primary), `step3_agreement_filtering.py`, `step10_ragas_eval.py`(키 우선순위에 SAIFEX 포함)
- `AgenticRAG/training/generate_600_VT_data.py`, `AgenticRAG/eval/generate_VT_hybrid_vision_dataset.py`

### 별도 Ahoseo 호스트 (`https://api.ahoseo.com/v1`, `AHOSEO_API_KEY`)

- `experience/exp1/step3.5_rewrite_vision_intent.py`

---

## [Deprecated] / [Legacy] 로 표시한 항목

| 표시 | 경로 | 사유 |
|------|------|------|
| **[Deprecated]** | `AgenticRAG/eval/pareto_plot,py` | 확장자가 `,py`로 비표준; 파레토 플롯용 자리만 있음 |
| **[Legacy]** | `ai_engine/chain.py` | Byaldi/ColPali 인덱스 **doc_id → 파일** 역추적 유틸; Agentic TV-RAG와 무관 |
| **[Legacy]** | `ai_engine/colpali.py`, `ai_engine/loader.py`(ColPali·요약 모드) | ColPali/Byaldi 기반 **대안 비전 인덱스** 스택 |
| **[Legacy]** | `ai_engine/sLM_RAG_pipeline.py` | 로컬 양자화 sLM 기반 공지 RAG (별도 추론 경로) |
| **[Legacy]** | `ai_engine/PDIS.py` | PDIS vs 기본 RAG 지연 비교 실험 스크립트 |
| **[Legacy]** | `experience/exp1/step7_check_raw_data.py` | `gemma_router_lora_stratified` 고정; step5/6의 **v4**와 불일치 시 참고용 |

**이전 문서에만 존재하던 파일:** `AgenticRAG/training/generate_dpo_datav2.py` 는 현재 저장소에 **없음** (삭제 또는 미추가).

---

## 그 외 핵심 디렉터리 요약

### `AgenticRAG/`

- `graph/state.py` — `AgentState`
- `graph/main_agent.py` — LangGraph 워크플로
- `nodes/router.py` — Gemma-2B + LoRA 라우터; **어댑터 기본 경로** `hoseo_router_gemma_2b_sft/` (`experience/exp1/step5` 산출물은 `gemma_router_lora_v4/` 등 별도 — 배포 시 경로 맞출 것)
- `training/*` — 라우터 SFT/DPO·데이터 준비 (`prepare_sft_data.py`, `train_router_sft.py`, `train_routerv2.py`, …)
- `eval/run_agentic_benchmark.py` — `main_agent.app`으로 공지+학칙 QA 샘플 벤치

### `evaluation/scripts/`

- 공지: `generate_qa.py`, `run_benchmark.py`, `run_eval.py`, `plot_results.py`
- 학칙 Text/Vision: `generate_qa_rules.py`, `run_benchmark_rules_text.py`, `run_benchmark_rules_pdf.py`, `run_eval_rules.py`, `plot_results_rules.py`, `run_eval_reverse.py`, `plot_reverse_results.py`

### `crawler/`

- `hoseo_spider.py` — 공지 수집 (`hoseo.ac.kr`)
- `rule_spider.py` — 규정/학칙 관련 수집 보조

### 인프라·문서

- `docker-compose.yml` — Milvus 등
- `docs/*.md` — 아키텍처, API, 크롤러, 프롬프트 등

### 로컬 산출물 (Git 제외 권장)

- `hoseo_router_gemma_2b_sft/`, `hoseo_router_gemma_2b/`, `hoseo_router_gemma_2b_v2/`, `experience/exp1/gemma_router_lora_v4/`, `temp_*_checkpoints/`, `data/`, `volumes/`

---

## 관찰 포인트

- **에이전트 라우터**(`router.py`)는 Gemma SFT 기반이며, **Text/Vision RAG의 벡터 검색**은 현재 **`rag_pipeline_rules`(학칙 컬렉션)** 에 묶여 있음. 공지까지 한 그래프에서 쓰려면 `retrieve_documents`를 `rag_pipeline.py` 쪽으로 통합하거나 이중 검색을 호출하도록 수정이 필요함.
- API가 **SAIFEX / 공식 OpenAI / api.ahoseo.com** 세 갈래로 나뉘어 있어, 키·`base_url` 일괄 변경 시 위 표를 기준으로 grep 하면 됨.

---

## 메타

| 파일 | 설명 |
|------|------|
| `PROJECT_MAP.md` | 본 문서 |
| `README.md` | 프로젝트 개요 |
