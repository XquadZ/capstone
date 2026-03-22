# PROJECT MAP - 호서대 학칙/공지사항 멀티모달 RAG

본 문서는 **현재 저장소 기준**으로 파일별 역할을 캡스톤 목적(학칙/공지 멀티모달 RAG + **Gemma-2B 동적 라우터**)에 맞춰 분류한 맵입니다.  
(경로는 리포지토리 루트 `capstone/` 기준)

---

## 1) Core Engine (`ai_engine/`)

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `full_text_extractor.py` | `raw` 공지 폴더 순회, 본문·이미지 OCR·PDF·HWP 텍스트 통합 추출 | 입력: `data/raw/*`, 출력: `data/processed/integrated_text/*.txt` |
| `local_slm_refiner.py` | 통합 텍스트를 GPT로 구조화(JSON·메타·정제 본문) | 입력: `integrated_text/*.txt`, 출력: `data/processed/text/*.json` |
| `chunker.py` | 정제 JSON을 의미 단위 청크로 분할·글로벌 컨텍스트 태그 부착 | 입력: `data/processed/text/*.json`, 출력: `data/processed/chunks/*_chunks.json` |
| `vector_db.py` | 공지 청크 BGE-M3(dense+sparse) 임베딩 후 Milvus 생성/적재 | 입력: `*_chunks.json`, 출력: Milvus `hoseo_notices` |
| `search_test.py` | Milvus 하이브리드 검색(dense+sparse+RRF) 단독 테스트 | 입력: 질의 문자열, 출력: Top-k hit |
| `rag_pipeline.py` | 공지 RAG(검색+리랭크+OpenAI 생성) | 입력: 질문, 출력: 답변 문자열 |
| `sLM_RAG_pipeline.py` | 로컬 sLM(양자화) 기반 공지 RAG | 입력: 질문, 출력: 로컬 생성 답변 |
| `vision_processor.py` | 공지 이미지/PDF를 비전 LLM으로 요약·멀티모달 확장 | 입력: `data/raw/{id}`, 출력: `data/processed/{id}/ai_extracted_info.json` |
| `loader.py` | 텍스트 요약 모드 / ColPali 임베딩 모드 CLI | 입력: `data/raw/*`, 출력: `processed/text` 또는 `processed/image/*.pt` |
| `colpali.py` | Byaldi/ColQwen 기반 비전 인덱스 생성 | 입력: `data/byaldi_input`, 출력: `.byaldi/hoseo_vision_index` |
| `find.py` | Byaldi 인덱스 내 doc_id → 파일·페이지 역추적 | 입력: `doc_ids_to_file_names.json.gz`, 출력: 콘솔 매핑 |
| `chain.py` | **현재 `find.py`와 동일한 Byaldi ID 추적 스크립트**(중복 복사 가능; 정리 시 하나로 통합 권장) | 동일 |
| `PDIS.py` | 기본 RAG 검색 vs PDIS 단계적 축소 검색 지연 비교 실험 | 입력: 질의셋+Milvus, 출력: CSV·HTML 그래프 |

### 학칙 전용

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `md_parser_pdf.py` | 학칙 PDF → 페이지 태그 포함 Markdown | 입력: `data/rules_regulations/raw_pdfs/*.pdf`, 출력: `markdown_parsed/*.md` |
| `rule_data_chunker.py` | 태그 기반 MD → 페이지 메타 청크 JSON | 입력: `markdown_parsed/*.md`, 출력: `chunks/all_rules_chunks.json` |
| `local_slm_refiner_rule.py` | 학칙 청크 띄어쓰기·줄바꿈 교정(Ollama) | 입력: `all_rules_chunks.json`, 출력: `all_rules_chunks_space.json` |
| `vector_db_rules.py` | 학칙 청크 Milvus `hoseo_rules_v1` 적재 | 입력: `all_rules_chunks_meta.json` 등, 출력: Milvus |
| `rag_pipeline_rules.py` | 학칙 검색·리랭크·스트리밍 생성(embedder/reranker/client 공개) | 입력: 질문, 출력: 답변·청크 |
| `test_force_ocr.py` | 지정 학칙 PDF 페이지 Tesseract OCR 점검 | 입력: PDF 경로, 출력: OCR 텍스트 |

---

## 2) Pipeline (Text vs Vision 실행 흐름)

### 공지
1. `full_text_extractor.py` → `local_slm_refiner.py` → `chunker.py` → `vector_db.py` → `rag_pipeline.py` / `sLM_RAG_pipeline.py`

### 학칙
1. `md_parser_pdf.py` → `rule_data_chunker.py` → (`local_slm_refiner_rule.py` 선택) → `vector_db_rules.py` → `rag_pipeline_rules.py`

### 벤치(Text vs Vision)
- Text: `evaluation/scripts/run_benchmark_rules_text.py`
- Vision(PDF 페이지 이미지): `evaluation/scripts/run_benchmark_rules_pdf.py`

### Agentic 통합 검색(Text 노드)
- `AgenticRAG/nodes/text_rag.py`가 `hoseo_rules_v1` + (선택) `hoseo_notices` 하이브리드 검색 후 `rag_pipeline_rules.generate_answer` 호출

---

## 3) Crawler (`crawler/`)

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `hoseo_spider.py` | 호서대 공지 등 수집 스크립트 | 입력: 타겟 URL/설정, 출력: `data/raw/{notice_id}/` 등 |
| `rule_spider.py` | 규정/학칙 관련 수집 보조 | 입력: 설정, 출력: raw 데이터 경로 |

---

## 4) Evaluation & Scripts (`evaluation/scripts/`)

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `generate_qa.py` | 공지 청크 기반 RAGAS용 Q&A 대량 생성 | 입력: `data/processed/chunks/*.json`, 출력: `datasets/ragas_testset_300.json` |
| `run_benchmark.py` | 공지 RAG 벤치마크(answer+contexts) | 입력: testset JSON, 출력: `results/benchmark_gpt4o_mini.json` |
| `run_eval.py` | RAGAS 4지표 채점 | 입력: benchmark JSON, 출력: `ragas_evaluation_report.csv` |
| `plot_results.py` | RAGAS 평균 막대그래프 | 입력: CSV, 출력: `evaluation_plot.png` |
| `generate_qa_rules.py` | 학칙 블록 기반 Q&A 생성 | 입력: `all_rules_chunks_meta.json`, 출력: `rules_ragas_testset.json` |
| `run_benchmark_rules_text.py` | 학칙 Text RAG 벤치(Reverse Repacking) | 입력: rules testset, 출력: `benchmark_rules_text.json` |
| `run_benchmark_rules_pdf.py` | 학칙 Vision RAG 벤치(PDF→이미지) | 입력: testset+PDF, 출력: `benchmark_rules_pdf.json` |
| `run_eval_rules.py` | Text vs Vision 10문항 단위 RAGAS | 입력: 위 benchmark JSON, 출력: `ragas_report_*.csv` |
| `plot_results_rules.py` | Text vs Multimodal 비교 플롯 | 입력: CSV, 출력: `evaluation_comparison_plot.png` |
| `run_eval_reverse.py` | Vision 답을 Gold로 역평가 | 입력: text/pdf benchmark, 출력: `ragas_reverse_report_*.csv` |
| `plot_reverse_results.py` | 역평가 플롯 | 입력: reverse CSV, 출력: `reverse_evaluation_plot.png` |

---

## 5) Agentic RAG (`AgenticRAG/`)

### Graph & 상태

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `graph/main_agent.py` | LangGraph: Router(룰베이스) → Text/Vision RAG → Critic → 재시도 엣지 | 입력: `question`, `retry_count`, 출력: `AgentState` 갱신 |
| `graph/state.py` | `AgentState` TypedDict 정의 | 필드: `question`, `route_decision`, `context`, `generation`, `critic_score`, `retry_count` |

### Nodes

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `nodes/text_rag.py` | 학칙+공지 Milvus 통합 검색·리랭크·`generate_answer` | 입력: state, 출력: `generation`, `context` |
| `nodes/vision_rag.py` | Vision RAG 노드(플레이스홀더; 실제 비전 파이프라인 TODO) | 입력: state, 출력: 더미 `generation`/`context` |
| `nodes/critic.py` | LLM으로 답변 품질 점수(0.5~1.0) 산출·재시도 카운트 | 입력: state, 출력: `critic_score`, `retry_count` |
| `nodes/router.py` | (현재 빈 파일) Gemma 라우터 연동용 확장 슬롯 | — |

### Training — Gemma-2B 동적 라우터 (TEXT/VISION)

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `training/generate_dpo_datav2.py` | DPO JSONL TEXT:VISION 5:5 밸런싱·페르소나 주입 | 입력: `dpo_dataset.jsonl`, 출력: `dpo_dataset_balanced_final.jsonl` (스크립트 내 경로) |
| `training/prepare_sft_data.py` | DPO `chosen` → SFT `messages`(user/model, 답은 TEXT/VISION) | 입력: balanced JSONL, 출력: `sft_dataset.jsonl` (**`train_router_sft.py`의 `dataset_path`와 경로 통일 권장**) |
| `training/train_router_sft.py` | `google/gemma-2-2b-it` + LoRA **SFT** → **`hoseo_router_gemma_2b_sft/`** 저장 | 입력: `AgenticRAG/training/sft_dataset.jsonl`, 출력: 루트 어댑터 폴더·`test_dataset_sft.jsonl` |
| `training/eval_router_sft.py` | SFT 어댑터 혼동 행렬·정확도 | 입력: `hoseo_router_gemma_2b_sft`, `test_dataset_sft.jsonl` |
| `training/check_raw_answers.py` | 소수 샘플 생성 문구 정성 점검 | 동일 모델·테스트셋 |
| `training/train_routerv2.py` | **DPO** 학습(별계열) → `hoseo_router_gemma_2b_v2` | 입력: balanced DPO JSONL |
| `training/debug.py` | 초기 DPO 어댑터 `hoseo_router_gemma_2b` VISION 샘플 디버그 | 입력: `test_dataset.jsonl` |
| `training/confusion_matrix.py` | DPO 어댑터 `hoseo_router_gemma_2b` 혼동 행렬 | 입력: `test_dataset.jsonl` |

### Training 데이터 산출물(저장소 내)

| 파일 | 설명 |
|---|---|
| `training/dpo_dataset.jsonl` | DPO 원본 |
| `training/dpo_dataset_balanced_final.jsonl` | 밸런싱·페르소나 적용본 |
| `training/sft_dataset.jsonl` | SFT용 messages JSONL |

### Eval

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `eval/pareto_plot,py` | 파일명 쉼표 오타(`.py` 권장); 파레토 플롯용 스크립트 자리 | 구현 상태 확인 필요 |

---

## 6) 루트·기타

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `check_raw_data.py` | `data/raw` 공지 폴더 통계(info.json·첨부·이미지 유무) | 입력: `data/raw`, 출력: 콘솔 리포트 |
| `Reference.txt` | 참고 문헌/링크 메모 | — |
| `PROJECT_MAP.md` | 본 문서 | — |
| `README.md` | 프로젝트 개요·Gemma 라우터·폴더 구조 | — |

---

## 7) Models & Checkpoints (로컬 산출물)

| 경로 | 주요 기능 | 비고 |
|---|---|---|
| `hoseo_router_gemma_2b_sft/` | **SFT 완료 LoRA** — 호서 RAG **동적 라우터**(TEXT/VISION) | 베이스: `google/gemma-2-2b-it` |
| `hoseo_router_gemma_2b/` | DPO(초기) 라우터 어댑터 | `debug.py` / `confusion_matrix.py` |
| `hoseo_router_gemma_2b_v2/` | DPO v2 라우터 어댑터 | `train_routerv2.py` |
| `temp_sft_checkpoints/`, `temp_router_checkpoints/`, `temp_router_checkpoints_v2/` | 학습 중간 체크포인트 | 용량 큼; Git 제외 권장 |

`.gitignore`에 모델/체크포인트 규칙을 팀 정책에 맞게 유지할 것.

---

## 8) Data & Config

| 파일/경로 | 주요 기능 |
|---|---|
| `docker-compose.yml` | Milvus(etcd·minio·standalone, `19530`) |
| `requirements.txt` | 최소 pip 의존성 |
| `.gitignore` | 데이터·결과·볼륨·일부 학습 산출물 제외 |
| `data/` | raw/processed/rules 등 (**통상 Git 제외**) |
| `volumes/` | Milvus 로컬 볼륨 (**제외**) |
| `docs/*.md` | `system_arch`, `api_spec`, `infra_setup`, `crawler_logic`, `prompt_rules`, `progress`, `frontend_srs` |
| `evaluation/datasets/*.json`, `evaluation/results/*` | 평가셋·결과(대부분 제외) |

---

## 9) 관찰 포인트

- **동적 라우터의 “제품” 산출물**은 **`hoseo_router_gemma_2b_sft/`**(SFT); `main_agent.py`의 `router_node`는 아직 룰베이스이며 Gemma 추론으로 교체 예정.
- `prepare_sft_data.py` 출력 경로와 `train_router_sft.py`의 `dataset_path`가 다르면 학습 전에 **경로 통일 또는 복사** 필요.
- `chain.py`와 `find.py` 내용이 동일하면 유지보수 시 **한 파일로 통합** 권장.
- `AgenticRAG/eval/pareto_plot,py`는 확장자명 오타 가능성 있음.
