# 🎓 호서대학교 스마트 캠퍼스 도우미 — 멀티모달 RAG + **Gemma-2B 동적 라우터**

본 프로젝트는 호서대학교 **공지사항·학칙/규정** 질의에 대해, 검색된 근거만으로 답변하는 **RAG(Retrieval-Augmented Generation)** 시스템입니다.  
핵심 차별점은 **사용자 질문을 분석해 `TEXT RAG`와 `VISION RAG` 중 어느 경로로 보낼지 결정하는 동적 라우터**이며, 이 라우터는 **`google/gemma-2-2b-it` 기반의 경량 LLM + LoRA SFT**로 학습된 **`hoseo_router_gemma_2b_sft`** 어댑터로 구현·평가합니다.

단일 **RTX 4090(24GB)** 등 로컬 GPU 환경에서 임베딩·리랭커·(선택) 로컬 생성 모델을 활용하고, **환각 최소화·출처 기반 답변**을 목표로 합니다.

---

## 🧭 왜 Gemma-2B 라우터인가?

- **TEXT RAG**: Milvus에 적재된 텍스트 청크를 BGE-M3 하이브리드 검색 + 리랭커로 가져와 답변합니다. 일반 질의·조문 위주에 적합합니다.
- **VISION RAG**: 표·별표·레이아웃이 중요한 질의는 PDF 페이지 이미지 등 **시각 정보**가 유리합니다.
- **동적 라우터**: 매 질문마다 위 두 경로 중 하나를 고르는 **이진 결정(TEXT / VISION)**을 내립니다.  
  **`hoseo_router_gemma_2b_sft`**는 그 결정을 **짧은 프롬프트만으로 빠르게** 내리도록 SFT된 **LoRA 어댑터**(베이스: `google/gemma-2-2b-it`)입니다.

> 운영 채팅 API(`backend/api_service.py`)는 **학습된 Gemma 라우터**(`AgenticRAG/nodes/router.py`, LoRA `gemma_router_lora_v4`)를 사용합니다. LangGraph `main_agent.py`는 별도 프로토타입입니다.

---

## 🚀 기술 스택 (요약)

| 영역 | 기술 |
|------|------|
| 벡터 DB | **Milvus** (`docker-compose.yml` — etcd / MinIO / standalone) |
| 임베딩 | **BAAI/bge-m3** (dense + sparse) |
| 재정렬 | **BAAI/bge-reranker-v2-m3** |
| 생성·평가 LLM | **gpt-4o-mini** 등 (OpenAI / SAIFEX 등 엔드포인트) |
| 동적 라우터 | **google/gemma-2-2b-it** + **PEFT LoRA** → 산출물 **`hoseo_router_gemma_2b_sft/`** |
| 에이전트 실험 | **LangGraph** (`AgenticRAG/graph/`) |
| 백엔드 연동 | Spring Boot → AI **JSON** `/ask` |
| 프론트(연동) | Flutter + **SSE** (Spring `/api/chat/ask`) |
| 문서 | `docs/` (아키텍처, API, 인프라, 프론트 SRS 등) |

---

## 📂 프로젝트 구조 (현재 저장소 기준)

```text
capstone/
├── backend/
│   └── api_service.py         # FastAPI TV-RAG — POST /ask (Spring 연동)
├── ai_engine/                 # RAG 코어: 전처리, 청킹, Milvus, 파이프라인
├── crawler/                   # hoseo_spider, incremental, crawl_all
├── AgenticRAG/
│   ├── nodes/                 # router.py, text_rag.py, vision_rag.py
│   ├── graph/main_agent.py    # LangGraph 프로토타입
│   └── training/              # 라우터 SFT·DPO
├── evaluation/scripts/
├── docs/
├── docker-compose.yml
├── experience/exp1/gemma_router_lora_v4/  # 라우터 LoRA (기본)
└── PROJECT_MAP.md
```

`data/`·`volumes/`·대용량 결과물·일부 체크포인트는 **`.gitignore`** 대상입니다. 모델 폴더 정책은 팀 규칙에 맞게 유지하세요.

---

## 🤖 AgenticRAG/training/ — SFT 학습 스크립트와 `hoseo_router_gemma_2b_sft` 역할

### 한 줄 요약

**`hoseo_router_gemma_2b_sft`**는 “호서대 RAG에서 이 질문은 텍스트 검색으로 충분한가, 비전(이미지/표) 쪽이 필요한가?”를 **`TEXT` / `VISION` 단답**으로 내리기 위해 **지도학습(SFT)**된 **Gemma-2B-it + LoRA** 어댑터 디렉터리입니다.

### 데이터 → SFT → 저장 흐름

1. **`generate_dpo_datav2.py`**  
   - DPO용 JSONL을 **TEXT:VISION ≈ 5:5**로 밸런싱하고, 라우터 **페르소나·지시문**을 프롬프트에 주입합니다.  
   - 출력 예: `AgenticRAG/rl_training/dpo_dataset_balanced_final.jsonl` (스크립트 내 경로 기준)

2. **`prepare_sft_data.py`**  
   - 위 DPO 데이터에서 **`chosen` 라벨**을 정답으로 삼아, Gemma 챗 형식의 **`messages`** 리스트로 변환합니다.  
   - `model` 역할 응답은 **`TEXT` 또는 `VISION`** 만 남기도록 정규화합니다.  
   - 출력: SFT용 JSONL (스크립트상 `AgenticRAG/rl_training/sft_dataset.jsonl` 등 — **실행 전 `train_router_sft.py`의 `dataset_path`와 경로를 맞출 것**)

3. **`train_router_sft.py`** — **최종 SFT 학습**  
   - 베이스: **`google/gemma-2-2b-it`** (`HUGGING_FACE_HUB_TOKEN` 필요)  
   - **LoRA** (`r=32`, `lora_alpha=64`, 다중 attention/MLP 모듈)  
   - **`trl.SFTTrainer`** + `SFTConfig` (fp16, epoch 평가/저장, `max_seq_length=256` 등)  
   - 학습 중 **`test_dataset_sft.jsonl`** 로 홀드아웃 테스트 분할 저장  
   - **최종 산출물**: 프로젝트 루트 **`hoseo_router_gemma_2b_sft/`** (어댑터 + 토크나이저 저장)

4. **`eval_router_sft.py`**  
   - `PeftModel.from_pretrained(base, "hoseo_router_gemma_2b_sft")` 로 로드 후 **`test_dataset_sft.jsonl`** 전체에 대해 추론  
   - **혼동 행렬**, 전체 정확도, TEXT/VISION별 방어·예측률 출력 (논문/보고용)

5. **`check_raw_answers.py`**  
   - 상위 N개 샘플에 대해 **실제 생성 문자열**을 출력해 라벨 붕괴·말더듬 등 **정성 점검**

### 같은 폴더의 관련 스크립트 (참고)

| 파일 | 역할 |
|------|------|
| `train_routerv2.py` | **DPO** 학습 경로 — 산출 예: `hoseo_router_gemma_2b_v2` (SFT와 별계열) |
| `debug.py`, `confusion_matrix.py` | 초기 **DPO** 어댑터 `hoseo_router_gemma_2b` 기준 디버그·혼동 행렬 |

---

## 📚 문서 허브

| 문서 | 설명 |
|------|------|
| [PROJECT_MAP.md](PROJECT_MAP.md) | 파일·모듈 역할 맵 |
| [docs/system_arch.md](docs/system_arch.md) | 시스템 아키텍처 |
| [docs/api_spec.md](docs/api_spec.md) | API / SSE 규격 |
| [docs/infra_setup.md](docs/infra_setup.md) | Milvus·Python 환경 |
| [docs/frontend_srs.md](docs/frontend_srs.md) | 프론트엔드 SRS |
| [docs/crawler_logic.md](docs/crawler_logic.md) | 수집·전처리 흐름 |
| [docs/prompt_rules.md](docs/prompt_rules.md) | RAG 프롬프트 원칙 |

---

## ▶️ 빠른 시작 (요약)

### TV-RAG 채팅 API (Spring/ngrok 연동)

1. `docker compose up -d` (Milvus)  
2. `conda activate capstone_final` + `SAIFEX_API_KEY` 설정  
3. `python -m backend.api_service` (포트 8000)  
4. `ngrok http 8000` → Spring `rag.server.url`에 등록  
5. API 계약: [docs/api_spec.md](docs/api_spec.md)

### 데이터·인덱싱

1. **증분 공지**: `python -m crawler.incremental_notice_update --once`  
2. **전체 백필**: `python -m crawler.crawl_all --no-webhook`  
3. **학칙 RAG**: `md_parser_pdf` → `rule_data_chunker` → `vector_db_rules`  
4. **라우터 SFT**: `AgenticRAG/training/` — [docs/infra_setup.md](docs/infra_setup.md) 참고  

---

## 라이선스·데이터

- **Gemma** 사용 시 Google/Hugging Face **이용 약관·라이선스**를 준수해야 합니다.  
- `data/` 등 원본 문서·크롤 데이터는 **학교/저작권 정책**에 맞게 취급하세요.
