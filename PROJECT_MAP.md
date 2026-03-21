# PROJECT MAP - 호서대 학칙/공지사항 멀티모달 RAG

본 문서는 현재 코드베이스를 기준으로, 파일별 역할을 캡스톤 목적(학칙/공지사항 멀티모달 RAG)에 맞춰 분류한 맵입니다.

## 1) Core Engine (`ai_engine/`)

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `ai_engine/full_text_extractor.py` | `raw` 공지 폴더를 순회하며 본문/이미지/OCR/PDF/HWP 텍스트를 통합 추출 | 입력: `data/raw/*`(info/images/attachments), 출력: `data/processed/integrated_text/*.txt` |
| `ai_engine/local_slm_refiner.py` | 통합 텍스트를 GPT로 구조화(JSON+메타데이터+정제 본문) | 입력: `integrated_text/*.txt`, 출력: `data/processed/text/*.json` |
| `ai_engine/chunker.py` | 정제 JSON을 의미 단위 청크로 분할하고 문서 컨텍스트 태그를 부착 | 입력: `data/processed/text/*.json`, 출력: `data/processed/chunks/*_chunks.json` |
| `ai_engine/vector_db.py` | 공지 청크를 BGE-M3(dense+sparse) 임베딩 후 Milvus 컬렉션 생성/적재 | 입력: `data/processed/chunks/*_chunks.json`, 출력: Milvus `hoseo_notices` |
| `ai_engine/search_test.py` | Milvus 하이브리드 검색(dense+sparse+RRF) 단독 테스트 | 입력: 사용자 질의 문자열, 출력: Top-k hit 메타/청크 텍스트 |
| `ai_engine/rag_pipeline.py` | 공지 텍스트 RAG의 메인 파이프라인(검색+리랭크+LLM 생성) | 입력: 질문 텍스트, 출력: 답변 문자열(근거 문맥 기반) |
| `ai_engine/sLM_RAG_pipeline.py` | 로컬 sLM(8bit) 기반 RAG 파이프라인(클라우드 LLM 대체) | 입력: 질문 텍스트, 출력: 로컬 생성 답변 |
| `ai_engine/vision_processor.py` | 공지 이미지/PDF를 비전 LLM으로 요약해 멀티모달 정보 확장 | 입력: `data/raw/{id}` 이미지/PDF, 출력: `data/processed/{id}/ai_extracted_info.json` |
| `ai_engine/loader.py` | 로더 유틸(텍스트 요약 모드, ColPali 임베딩 모드) 실행 스크립트 | 입력: `data/raw/*`, 출력: `data/processed/text/*.json` 또는 `data/processed/image/*.pt` |
| `ai_engine/colpali.py` | Byaldi/ColQwen 기반 비전 인덱스 생성 | 입력: `data/byaldi_input`, 출력: `.byaldi/hoseo_vision_index` |
| `ai_engine/find.py` | Byaldi 인덱스의 문서 ID를 실제 파일/페이지로 역추적 | 입력: `doc_ids_to_file_names.json.gz` + target id, 출력: 파일명/페이지 매핑 로그 |
| `ai_engine/PDIS.py` | 기존 RAG 검색과 PDIS 단계적 축소 검색의 지연/속도 향상 비교 실험 | 입력: 테스트 질의셋 + Milvus 벡터, 출력: `PDIS_Research_Final_Report.csv`, `PDIS_Analysis_Graph.html` |

### 학칙 전용 Core 보강

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `ai_engine/md_parser_pdf.py` | 학칙 PDF를 페이지 태그 포함 Markdown으로 변환 | 입력: `data/rules_regulations/raw_pdfs/*.pdf`, 출력: `markdown_parsed/*.md` |
| `ai_engine/rule_data_chunker.py` | 태그 기반 Markdown을 페이지 메타 포함 청크 JSON으로 변환 | 입력: `markdown_parsed/*.md`, 출력: `chunks/all_rules_chunks.json` |
| `ai_engine/local_slm_refiner_rule.py` | 학칙 청크 텍스트의 띄어쓰기/줄바꿈 교정(OLLAMA) | 입력: `all_rules_chunks.json`, 출력: `all_rules_chunks_space.json` |
| `ai_engine/vector_db_rules.py` | 학칙 청크를 Milvus 학칙 컬렉션(`hoseo_rules_v1`)에 임베딩 저장 | 입력: `all_rules_chunks_meta.json`, 출력: Milvus `hoseo_rules_v1` |
| `ai_engine/rag_pipeline_rules.py` | 학칙 질의용 텍스트 RAG(검색/리랭크/스트리밍 생성) 실행 | 입력: 질문 텍스트, 출력: 학칙 답변 문자열 |
| `ai_engine/test_force_ocr.py` | 특정 학칙 PDF 페이지에 강제 OCR 품질 점검 | 입력: 단일 PDF+페이지, 출력: OCR 텍스트 콘솔 결과 |

## 2) Pipeline (Text vs Vision RAG 실행 흐름)

### 공지 파이프라인 (일반)
1. `full_text_extractor.py` -> 원천 텍스트 통합  
2. `local_slm_refiner.py` -> JSON 정제  
3. `chunker.py` -> 검색용 청크화  
4. `vector_db.py` -> Milvus 적재  
5. `rag_pipeline.py` 또는 `sLM_RAG_pipeline.py` -> 질의응답

### 학칙/규정 파이프라인 (Text vs Vision 비교 실험용)
1. `md_parser_pdf.py` -> `rule_data_chunker.py` -> `vector_db_rules.py`  
2. Text 추론: `rag_pipeline_rules.py` 또는 `evaluation/scripts/run_benchmark_rules_text.py`  
3. Vision 추론: `evaluation/scripts/run_benchmark_rules_pdf.py` (PDF 페이지 이미지 포함 질의)

## 3) Evaluation & Scripts (`evaluation/scripts/`)

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `evaluation/scripts/generate_qa.py` | 공지 청크 기반 RAGAS용 Q&A(약 300) 자동 생성 | 입력: `data/processed/chunks/*.json`, 출력: `evaluation/datasets/ragas_testset_300.json` |
| `evaluation/scripts/run_benchmark.py` | 공지 RAG 파이프라인 벤치마크 실행(질문별 answer+contexts 저장) | 입력: `ragas_testset_300.json`, 출력: `results/benchmark_gpt4o_mini.json` |
| `evaluation/scripts/run_eval.py` | 벤치마크 결과를 RAGAS 4지표로 채점 | 입력: `benchmark_gpt4o_mini.json`, 출력: `ragas_evaluation_report.csv` |
| `evaluation/scripts/plot_results.py` | 단일 RAGAS 결과를 평균 막대그래프로 시각화 | 입력: `ragas_evaluation_report.csv`, 출력: `evaluation_plot.png` |
| `evaluation/scripts/generate_qa_rules.py` | 학칙 청크 블록 기반 Q&A 생성(추론/조건부/멀티문맥) | 입력: `all_rules_chunks_meta.json`, 출력: `datasets/rules_ragas_testset.json` |
| `evaluation/scripts/run_benchmark_rules_text.py` | 학칙 Text RAG 벤치마크(Reverse Repacking 포함) | 입력: `rules_ragas_testset.json`, 출력: `benchmark_rules_text.json` |
| `evaluation/scripts/run_benchmark_rules_pdf.py` | 학칙 Vision RAG 벤치마크(PDF 페이지 이미지 포함) | 입력: `rules_ragas_testset.json`+원본 PDF, 출력: `benchmark_rules_pdf.json` |
| `evaluation/scripts/run_eval_rules.py` | Text vs Vision 벤치 결과를 10개 단위 체크포인트로 평가 | 입력: `benchmark_rules_text/pdf.json`, 출력: `ragas_report_text.csv`, `ragas_report_pdf.csv` |
| `evaluation/scripts/plot_results_rules.py` | Text vs Multimodal 성능 비교 그래프 생성 | 입력: `ragas_report_text.csv`, `ragas_report_pdf.csv`, 출력: `evaluation_comparison_plot.png` |
| `evaluation/scripts/run_eval_reverse.py` | Vision 답변을 Gold로 두고 역평가(정보 유실 정량화) | 입력: `benchmark_rules_text/pdf.json`, 출력: `ragas_reverse_report_*.csv` |
| `evaluation/scripts/plot_reverse_results.py` | 역평가 결과 시각화 | 입력: `ragas_reverse_report_*.csv`, 출력: `reverse_evaluation_plot.png` |

## 4) Data & Config

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `docker-compose.yml` | Milvus 단독 스택(etcd/minio/standalone) 구동 설정 | 입력: Docker compose 실행, 출력: `19530` Milvus 서비스/`volumes/*` |
| `requirements.txt` | 핵심 파이썬 의존성 최소 정의(일부 모듈) | 입력: `pip install -r`, 출력: 실행환경 패키지 설치 |
| `evaluation/datasets/ragas_testset_300.json` | 공지 벤치마크용 Q&A 평가셋 | 입력: 생성 스크립트 산출물, 출력: 벤치마크 스크립트 입력 |
| `evaluation/datasets/rules_ragas_testset.json` | 학칙 벤치마크용 Q&A 평가셋 | 입력: 생성 스크립트 산출물, 출력: Text/Vision 벤치마크 입력 |
| `evaluation/results/benchmark_gpt4o_mini.json` | 공지 RAG 벤치마크 원시 결과 | 입력: `run_benchmark.py`, 출력: `run_eval.py` 입력 |
| `evaluation/results/benchmark_rules_text.json` | 학칙 Text RAG 벤치마크 원시 결과 | 입력: `run_benchmark_rules_text.py`, 출력: 평가/역평가 입력 |
| `evaluation/results/benchmark_rules_pdf.json` | 학칙 Vision RAG 벤치마크 원시 결과 | 입력: `run_benchmark_rules_pdf.py`, 출력: 평가/역평가 입력 |
| `evaluation/results/*.png` | 최종 비교/역비교 시각화 산출물 | 입력: plot 스크립트, 출력: 보고용 이미지 |
| `README.md` | 프로젝트 전체 목표/스택/문서 진입점 안내 | 입력: 개발/운영 참조, 출력: 문서 네비게이션 |

## 5) New Agentic (`AgenticRAG/`)

| 파일 | 주요 기능 | 입출력 데이터 |
|---|---|---|
| `AgenticRAG/graph/main_agent.py` | LangGraph 상태기계 라우터-리트리버-크리틱 루프의 프로토타입 | 입력: `question` 상태, 출력: `route_decision/context/generation/critic_score` 상태 |
| `AgenticRAG/nodes/router.py` | (현재 비어 있음) 라우팅 노드 분리 구현 예정 | 입력/출력: 미구현 |
| `AgenticRAG/nodes/text_rag.py` | (현재 비어 있음) Text RAG 노드 분리 구현 예정 | 입력/출력: 미구현 |
| `AgenticRAG/nodes/vision_rag.py` | (현재 비어 있음) Vision RAG 노드 분리 구현 예정 | 입력/출력: 미구현 |
| `AgenticRAG/nodes/critic.py` | (현재 비어 있음) 품질평가/재시도 노드 분리 구현 예정 | 입력/출력: 미구현 |
| `AgenticRAG/rl_traning/generate_dpo_data.py` | (현재 비어 있음) 라우터/정책 학습용 DPO 데이터 생성 예정 | 입력/출력: 미구현 |
| `AgenticRAG/rl_traning/train_router.py` | (현재 비어 있음) 라우터 정책 학습 스크립트 예정 | 입력/출력: 미구현 |
| `AgenticRAG/eval/pareto_plot,py` | (현재 비어 있음, 파일명 오타 가능) 비용-성능 파레토 시각화 예정 | 입력/출력: 미구현 |

---

## 현재 코드베이스 관찰 포인트

- `AgenticRAG`는 `main_agent.py`만 프로토타입 코드가 있으며, 나머지 노드/학습/평가 파일은 빈 스켈레톤 상태입니다.
- 학칙 파이프라인은 Text RAG와 Vision RAG를 각각 벤치마크한 뒤 `run_eval_rules.py`, `run_eval_reverse.py`로 성능 차이를 검증하도록 설계되어 있습니다.
- 데이터 폴더(`data/`)는 Git 추적 제외 상태로 보이며, 코드에서 가정하는 경로를 기준으로 전처리/인덱싱/평가 스크립트가 연결됩니다.
