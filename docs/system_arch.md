# 🏛️ System Architecture - Advanced Multimodal RAG Backend

본 문서는 호서대학교 공지사항 및 학사 정보를 실시간으로 제공하는 챗봇 시스템의 전반적인 아키텍처를 정의합니다. 
단일 로컬 GPU(NVIDIA RTX 4090, 24GB) 자원을 극대화하여 활용하며, **FastAPI 기반의 단일 백엔드**와 **Redis Stack을 활용한 하이브리드 데이터베이스** 구조를 채택하여 라이브 서비스의 실시간성과 안정성을 보장합니다.



---

## 1. Core Tech Stack (핵심 기술 스택)

| 계층 (Layer) | 기술 스택 | 주요 역할 |
| :--- | :--- | :--- |
| **Frontend** | React | 사용자 웹 UI, SSE(Server-Sent Events) 스트리밍 응답 렌더링 |
| **Backend & API** | FastAPI (Python) | 비동기 API 게이트웨이, RAG 파이프라인 제어, 세션 관리 |
| **Database** | Redis Stack | 사용자 대화 세션(Memory), 메타데이터(JSON), 벡터 검색(Vector) |
| **AI Serving** | vLLM, PyTorch | 대형 언어 모델(LLM) 고속 추론 및 VRAM 메모리 관리 |
| **Data Pipeline** | Selenium, LLaVA | 웹 크롤링, 원본 데이터(PDF/Img) 수집 및 멀티모달 텍스트 변환 |

---

## 2. 데이터 수집 및 전처리 파이프라인 (Offline Ingestion)
라이브 서비스의 부하를 막기 위해, 데이터 수집은 백그라운드 스케줄러를 통해 오프라인으로 진행됩니다.

* **2.1 수집 파이프라인 (`crawler` 모듈):**
  * 대상: 호서대학교 공지사항 웹페이지 및 첨부파일.
  * 중복 방지: 공지사항 고유 ID를 기준으로 Redis를 조회하여 신규 데이터만 수집 (Idempotency 보장).
  * Raw 데이터 보존: 원본 형태를 유지하기 위해 수집된 PDF와 이미지는 로컬 `data/raw/[공지ID]/` 경로에 물리적으로 저장.

* **2.2 멀티모달 변환 (Vision-to-Text):**
  * 포스터, 안내문 등 텍스트 추출이 불가능한 이미지는 로컬 Vision LLM(LLaVA 등)을 통해 상세한 상황 묘사 텍스트로 변환.
  * 변환된 텍스트는 원본 게시글의 텍스트와 병합되어 `info.json`에 메타데이터로 기록됨.

* **2.3 텍스트 분할 (Chunking):**
  * `langchain`의 `RecursiveCharacterTextSplitter` 사용.
  * `chunk_size`: 800자 (적절한 문맥 포함)
  * `chunk_overlap`: 100자 (청크 간 문맥 단절 방지)

---

## 3. 하이브리드 데이터베이스 설계 (Redis Stack)
단순 벡터 검색을 넘어, 필터링과 세션 관리를 동시에 수행하기 위해 Redis를 메인 DB로 사용합니다.

* **3.1 메타데이터 및 정형 데이터 (Redis JSON):**
  * 제목, 작성일, 카테고리, 원본 파일 로컬 경로 등을 JSON 형태로 저장.
  * 챗봇이 답변할 때 "원본 문서 보기" 링크를 즉시 제공하기 위한 근거 데이터.
* **3.2 임베딩 및 벡터 검색 (Redis Vector Search):**
  * 임베딩 모델: `BAAI/bge-m3` (다국어 및 한국어 검색 성능 우수).
  * 청크(Chunk)된 텍스트를 벡터로 변환하여 저장하고, 사용자의 질의와 코사인 유사도(Cosine Similarity)를 계산하여 검색.
* **3.3 대화 세션 관리 (Session Memory):**
  * 사용자의 `session_id`를 Key로 하여 최근 대화 이력(K=5~10)을 List 또는 Hash로 저장.
  * 멀티턴(Multi-turn) 대화 시 문맥 유지를 위해 LLM 프롬프트에 동적으로 주입됨.

---

## 4. 실시간 RAG 검색 및 답변 생성 (Real-time Generation)
단순 검색의 한계를 극복하기 위해 **2-Stage 검색(Reranking)**과 **스트리밍(Streaming)**을 도입합니다.

* **4.1 1차 검색 (Retrieval):**
  * Redis Vector Search를 통해 질문과 유사한 문서 10개(K=10)를 빠르게 1차 추출.
* **4.2 2차 검색 (Reranking):**
  * 모델: `Dongjin-kr/ko-reranker` (Cross-Encoder 연산).
  * 1차로 찾은 10개의 문서와 질문 간의 실제 문맥적 연관성을 GPU로 재계산하여, 가장 정확한 최상위 3개 문서만 필터링. (환각 현상 최소화)
* **4.3 LLM 추론 및 스트리밍 (Generation):**
  * 생성 모델: `EEVE-Korean-10.8B` 또는 `Llama-3-8B-Instruct`.
  * 프레임워크: `vLLM`을 활용한 PagedAttention 기술 적용.
  * **SSE (Server-Sent Events):** 답변이 완성될 때까지 기다리지 않고, 생성되는 즉시 토큰 단위로 프론트엔드에 스트리밍 전송하여 체감 대기 시간을 0초에 가깝게 구현.

---

## 5. RTX 4090 VRAM 예산 관리 (Resource Allocation)
단일 24GB VRAM 안에서 파이프라인이 병목 없이 동작하도록 메모리를 엄격히 분할합니다.

* **Total VRAM:** 24.0 GB
* **LLM (EEVE-10.8B):** ~ 10.0 GB (AWQ 또는 GGUF 양자화 적용 필수)
* **Embedding (bge-m3):** ~ 4.0 GB
* **Reranker (ko-reranker):** ~ 3.0 GB
* **Buffer & Context Window:** ~ 7.0 GB (동시 접속자 요청 처리 및 KV Cache 용도)

---

## 6. 디렉토리 구조 (Directory Layout)
```text
CAPSTONE/
├── ai_engine/                 # AI 코어 로직
│   ├── loader.py              # 데이터 로드 및 텍스트 병합
│   ├── vector_db.py           # Redis Stack 연결 및 임베딩 관리
│   ├── reranker.py            # Cross-Encoder 기반 재정렬 로직
│   ├── vision_processor.py    # 멀티모달 이미지 텍스트 변환 엔진
│   └── chain.py               # RAG 파이프라인 (검색 -> 생성) 총괄
├── api/                       # 백엔드 서버
│   └── main.py                # FastAPI 엔드포인트 (스트리밍, 세션 관리)
├── crawler/                   # 데이터 수집 (Offline)
│   ├── hoseo_spider.py        # 호서대 공지사항 크롤러
│   └── utils.py               # 중복 검사, 파일 다운로드 등 공통 함수
├── data/                      
│   └── raw/                   # [Git 제외] 수집된 PDF, 이미지 물리적 저장소
├── docs/                      # 시스템 아키텍처 및 API 명세서
└── requirements.txt           # [파일] 파이썬 패키지 의존성 명세