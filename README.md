# 🎓 호서대학교 학칙/공지사항 멀티모달 RAG

본 프로젝트는 호서대학교 **공지사항 + 학칙/규정 문서**를 대상으로,  
질문 유형에 따라 Text RAG와 Vision RAG를 활용해 답변을 생성하고 성능을 비교/평가하는 캡스톤 코드베이스입니다.

핵심 목표는 다음과 같습니다.
- 공지/학칙 문서 기반의 근거 중심 답변 생성
- 표/양식/시각 정보가 포함된 문서에서 Vision 기반 보완
- RAGAS 기반 정량 평가 및 역평가(Reverse Evaluation)로 정보 유실 검증

---

## 🚀 핵심 스택

- **Vector DB:** Milvus (Hybrid Search: dense + sparse)
- **Embedding/Rerank:** `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`
- **LLM:** `gpt-4o-mini`(OpenAI/SAIFEX), 로컬 sLM(선택)
- **Vision:** PDF 페이지 이미지 기반 멀티모달 질의 (`gpt-4o-mini` Vision 입력)
- **Evaluation:** RAGAS + Matplotlib/Seaborn

---

## 📂 최신 프로젝트 구조

```text
CAPSTONE/
├── ai_engine/                 # 검색/파싱/OCR/벡터화/RAG 실행 핵심 엔진
├── evaluation/
│   ├── scripts/               # 벤치마크, 평가, QA 생성, 플롯 스크립트
│   ├── datasets/              # 평가셋(JSON)
│   └── results/               # 벤치마크/평가 결과(JSON, CSV, PNG)
├── AgenticRAG/                # LangGraph 기반 Agentic RAG 실험(신규)
├── docs/                      # 설계/인프라/API 문서
├── docker-compose.yml         # Milvus(etcd/minio/standalone) 구동
├── requirements.txt           # Python 의존성(최소 목록)
└── PROJECT_MAP.md             # 파일별 역할/입출력 상세 맵
```

> 파일 단위 상세 역할은 `PROJECT_MAP.md`를 참고하세요.

---

## 🔄 실행 파이프라인 요약

### 1) 공지 Text RAG 파이프라인
1. `full_text_extractor.py` -> 원천 데이터 통합 추출  
2. `local_slm_refiner.py` -> 정제 JSON 생성  
3. `chunker.py` -> 컨텍스트 포함 청크 생성  
4. `vector_db.py` -> Milvus 적재  
5. `rag_pipeline.py` / `sLM_RAG_pipeline.py` -> 질의응답

### 2) 학칙 Text vs Vision 비교 파이프라인
1. `md_parser_pdf.py` -> PDF to Markdown(+페이지 태그)  
2. `rule_data_chunker.py` -> 학칙 청크 JSON 생성  
3. `vector_db_rules.py` -> Milvus(`hoseo_rules_v1`) 적재  
4. Text 벤치: `evaluation/scripts/run_benchmark_rules_text.py`  
5. Vision 벤치: `evaluation/scripts/run_benchmark_rules_pdf.py`  
6. 평가: `run_eval_rules.py`, `run_eval_reverse.py`, `plot_*`

---

## 🧪 평가/실험 스크립트

- **QA 생성:** `evaluation/scripts/generate_qa.py`, `generate_qa_rules.py`
- **벤치마크:** `run_benchmark.py`, `run_benchmark_rules_text.py`, `run_benchmark_rules_pdf.py`
- **RAGAS 평가:** `run_eval.py`, `run_eval_rules.py`
- **역평가:** `run_eval_reverse.py`
- **시각화:** `plot_results.py`, `plot_results_rules.py`, `plot_reverse_results.py`

---

## 🧠 AgenticRAG 현황

`AgenticRAG/graph/main_agent.py`에는 LangGraph 프로토타입(라우터 -> Text/Vision -> Critic -> 재시도 루프)이 구현되어 있습니다.  
`AgenticRAG/nodes/*`, `AgenticRAG/rl_traning/*`, `AgenticRAG/eval/pareto_plot,py`는 현재 스켈레톤(비어 있는 파일) 상태입니다.

---

## 📚 문서

- [파일별 역할 맵](PROJECT_MAP.md)
- [시스템 아키텍처](docs/system_arch.md)
- [API 명세서](docs/api_spec.md)
- [인프라 세팅](docs/infra_setup.md)
- [크롤링 로직](docs/crawler_logic.md)
- [프롬프트 규칙](docs/prompt_rules.md)