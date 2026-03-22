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

> LangGraph 에이전트(`AgenticRAG/graph/main_agent.py`)의 `router_node`는 현재 **룰베이스 예시**이며, 추후 동일 슬롯에 **학습된 Gemma 라우터 추론**을 연결하는 것을 전제로 합니다.

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
| 프론트(연동 예정) | React + **SSE** 스트리밍 |
| 문서 | `docs/` (아키텍처, API, 인프라, 프론트 SRS 등) |

---

## 📂 프로젝트 구조 (현재 저장소 기준)

```text
capstone/
├── ai_engine/                 # RAG 코어: 전처리, 청킹, Milvus 적재, 파이프라인
│   ├── rag_pipeline.py        # 공지 Milvus RAG + OpenAI 생성
│   ├── rag_pipeline_rules.py  # 학칙 Milvus RAG + 스트리밍 생성
│   ├── vector_db.py / vector_db_rules.py
│   ├── chunker.py, rule_data_chunker.py, md_parser_pdf.py
│   ├── full_text_extractor.py, local_slm_refiner.py, vision_processor.py
│   └── …
├── evaluation/                # 벤치마크, RAGAS, 플롯
│   ├── scripts/
│   └── datasets/ / results/   # 대부분 .gitignore (용량)
├── AgenticRAG/                # LangGraph + 라우터 학습·평가
│   ├── graph/
│   │   ├── main_agent.py      # Router → Text/Vision RAG → Critic 그래프
│   │   └── state.py
│   ├── nodes/
│   │   ├── text_rag.py, vision_rag.py, critic.py
│   │   └── router.py          # (확장용)
│   ├── training/              # ⬇︎ SFT·DPO·데이터 스크립트 (아래 상세)
│   └── eval/
├── docs/                      # system_arch, api_spec, infra_setup, frontend_srs, …
├── docker-compose.yml         # Milvus 스택
├── requirements.txt
├── PROJECT_MAP.md             # 파일별 역할 맵
├── hoseo_router_gemma_2b_sft/ # ✅ SFT 완료 LoRA 어댑터 (프로젝트 루트)
│   # adapter_config.json, adapter_model.safetensors, tokenizer* …
└── …                          # temp_*_checkpoints, 기타 실험 산출물 (로컬)
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

1. **Milvus**: 저장소 루트에서 `docker compose up -d`  
2. **의존성**: `pip install -r requirements.txt` 및 Milvus/RAG 스크립트에 필요한 패키지 추가 설치 (`pymilvus`, `FlagEmbedding`, `openai` 등 — [docs/infra_setup.md](docs/infra_setup.md) 참고)  
3. **공지 RAG**: `ai_engine/` 파이프라인으로 청크 적재 후 `rag_pipeline.py`  
4. **학칙 RAG**: `md_parser_pdf` → `rule_data_chunker` → `vector_db_rules` → `rag_pipeline_rules.py`  
5. **라우터 SFT**: `AgenticRAG/training/`에서 데이터 준비 후 `train_router_sft.py` → **`hoseo_router_gemma_2b_sft`** 생성 → `eval_router_sft.py`로 검증  

---

## 라이선스·데이터

- **Gemma** 사용 시 Google/Hugging Face **이용 약관·라이선스**를 준수해야 합니다.  
- `data/` 등 원본 문서·크롤 데이터는 **학교/저작권 정책**에 맞게 취급하세요.
