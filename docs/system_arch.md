# System Architecture (2026-03)

현재 저장소의 실제 코드 기준 아키텍처 요약 문서입니다.

## 1. 핵심 구성

| 레이어 | 기술 | 역할 |
|---|---|---|
| Frontend | React (예정/연동) | SSE 스트리밍 답변 렌더링 |
| Backend/API | FastAPI (문서 기준), Python 스크립트 기반 실행 | 질의 수신, RAG 호출, 스트리밍 응답 |
| Vector DB | Milvus 2.4 (`docker-compose.yml`) | Dense/Sparse 하이브리드 검색 저장소 |
| Embedding | `BAAI/bge-m3` | 쿼리/문서 임베딩(dense+sparse) |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-Encoder 재정렬 |
| LLM | `gpt-4o-mini` / 로컬 sLM | 최종 답변 생성 |
| Vision | GPT-4o Vision, ColPali/Byaldi 인덱스 실험 | 이미지/PDF 페이지 기반 보강 |

## 2. 데이터 파이프라인

### 2.1 공지사항 트랙
1. `ai_engine/full_text_extractor.py`  
   `data/raw/*`에서 본문/이미지/OCR/PDF/HWP 텍스트 통합
2. `ai_engine/local_slm_refiner.py`  
   통합 텍스트를 구조화 JSON으로 정제
3. `ai_engine/chunker.py`  
   검색용 청크 생성
4. `ai_engine/vector_db.py`  
   Milvus `hoseo_notices` 컬렉션 생성/적재
5. `ai_engine/rag_pipeline.py`  
   검색 + 리랭킹 + 생성

### 2.2 학칙/규정 트랙
1. `ai_engine/md_parser_pdf.py`  
   PDF -> markdown(+page tag)
2. `ai_engine/rule_data_chunker.py`  
   markdown -> chunk JSON
3. `ai_engine/vector_db_rules.py`  
   Milvus `hoseo_rules_v1` 적재
4. `ai_engine/rag_pipeline_rules.py`  
   학칙 질의응답

## 3. 검색/생성 흐름

1. 사용자 질문 입력
2. BGE-M3로 쿼리 dense/sparse 임베딩 생성
3. Milvus hybrid search (`RRFRanker`)로 후보 추출
4. BGE reranker로 상위 컨텍스트 재정렬
5. 컨텍스트 기반으로 LLM 답변 생성
6. (평가 스크립트에서) `contexts`를 함께 저장해 RAGAS 평가

## 4. 멀티모달/에이전틱 확장

- Vision 벤치마크: `evaluation/scripts/run_benchmark_rules_pdf.py`
- ColPali 인덱싱 실험: `ai_engine/colpali.py`, `ai_engine/find.py`
- AgenticRAG 프로토타입: `AgenticRAG/graph/main_agent.py`
  - Router -> Text/Vision -> Critic 루프 구조
  - 일부 노드 파일은 아직 스켈레톤 상태

## 5. 인프라 구성

Milvus는 `docker-compose.yml`로 관리:
- `etcd`
- `minio`
- `milvus-standalone` (`19530`, `9091`)

## 6. 디렉토리 스냅샷

```text
CAPSTONE/
├── ai_engine/                 # 전처리/인덱싱/RAG 코어
├── evaluation/                # QA 생성/벤치마크/RAGAS/플롯
├── AgenticRAG/                # LangGraph 기반 확장 실험
├── docs/                      # 프로젝트 문서
├── docker-compose.yml         # Milvus 스택
└── requirements.txt           # 최소 의존성
```