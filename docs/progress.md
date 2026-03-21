# 프로젝트 진행 현황 (2026-03-21 기준)

## 1. 완료된 핵심 작업

- 공지/학칙 RAG 파이프라인 분리 구축
  - 공지: `full_text_extractor` -> `local_slm_refiner` -> `chunker` -> `vector_db` -> `rag_pipeline`
  - 학칙: `md_parser_pdf` -> `rule_data_chunker` -> `vector_db_rules` -> `rag_pipeline_rules`
- Milvus 하이브리드 검색 체계 적용
  - BGE-M3 dense+sparse + RRF + reranker
- 평가 자동화 스크립트 구축
  - QA 생성, 벤치마크, RAGAS 채점, 비교 플롯, 역평가
- 문서 정비
  - `PROJECT_MAP.md`
  - `docs/frontend_srs.md`

## 2. 진행 중

- `AgenticRAG` 구조 정리
  - `graph/main_agent.py` 프로토타입 존재
  - `nodes/*`, `rl_traning/*`, `eval/pareto_plot,py` 고도화 필요

## 3. 다음 우선순위

1. AgenticRAG 노드 분리 구현(`router`, `text_rag`, `vision_rag`, `critic`)
2. 파일명 정리: `pareto_plot,py` -> `pareto_plot.py`, `rl_traning` 오탈자 여부 검토
3. API 서버 코드와 문서 스키마 동기화(`/chat/stream`, history/session)
4. 평가 파이프라인 CI 자동화(최소 smoke benchmark)

## 4. 리스크/체크포인트

- `data/` 미추적 환경이라 재현성 확보를 위해 샘플 데이터셋 버전 관리 필요
- 외부 API 키(`OPENAI_API_KEY`, `SAIFEX_API_KEY`) 의존 스크립트 분리/명시 필요
- GPU 메모리 사용량은 배치 크기 및 모델 동시 구동에 민감하므로 실행 프로파일 축적 필요