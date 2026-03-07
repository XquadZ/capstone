# 🏛️ System Architecture - Advanced Hybrid Multimodal RAG Backend

본 문서는 호서대학교 공지사항 및 학사 정보를 실시간으로 제공하는 챗봇 시스템의 전반적인 아키텍처를 정의합니다.
연구실 서버의 단일 최고 사양 GPU(**NVIDIA RTX 4090, 24GB VRAM**) 자원을 활용하여 텍스트와 비전 모델을 메모리에 상주(Always-on)시키며, 문서의 특성에 따라 파이프라인을 분기하는 **하이브리드 RAG 접근법**과 **Elasticsearch 8.x 기반의 검색 엔진**을 채택하여 압도적인 처리 속도와 라이브 서비스의 안정성을 보장합니다.

---

## 1. Core Tech Stack (핵심 기술 스택)

| 계층 (Layer) | 기술 스택 | 주요 역할 |
| :--- | :--- | :--- |
| **Frontend** | React | 사용자 웹 UI, SSE(Server-Sent Events) 스트리밍 응답 렌더링 |
| **Backend & API** | FastAPI (Python) | 비동기 API 게이트웨이, RAG 파이프라인 제어, 세션 관리 |
| **Search Engine** | Elasticsearch 8.x | 하이브리드 검색(BM25 + Dense kNN + Late Interaction), 메타데이터 필터링 |
| **Text Preprocessing & Generation** | Qwen2.5-7B-Instruct | 텍스트 요약, 메타데이터 추출 및 최종 답변 생성 (vLLM 활용 고속 추론) |
| **Vision Preprocessing** | ColQwen2.5 | 복잡한 시각 문서(표/차트)의 멀티벡터(Multi-vector) 변환 |
| **Text Embedding** | BAAI/bge-m3 | 요약된 텍스트의 고품질 다국어/한국어 벡터화 |
| **Data Pipeline** | BeautifulSoup, PyMuPDF | 웹 크롤링, 문서 레이아웃 파싱 및 Text/Vision 라우팅 |

---

## 2. 하이브리드 전처리 파이프라인 (Offline Ingestion)
풍부한 24GB VRAM을 활용하여, 모델 스와핑(Swapping) 없이 Text Track과 Vision Track을 동시에 빠르고 유연하게 전처리합니다.

* **2.1 데이터 수집 및 라우팅 (`crawler` & `loader`):**
  * 호서대학교 공지사항 및 첨부파일(PDF/HWP)을 수집합니다.
  * 문서의 레이아웃을 분석하여, 줄글 위주의 페이지는 **Text Track**으로, 표/다단 등 시각적 복잡도가 높은 페이지는 **Vision Track**으로 라우팅합니다.

* **2.2 Text Track (sLM 기반 요약 및 단일 벡터화):**
  * 메모리에 상주하는 Qwen2.5-7B를 거쳐 핵심 내용 요약 및 마크다운으로 구조화됩니다.
  * 요약된 텍스트는 `bge-m3` 모델을 통해 단일 임베딩 벡터(`text_vector`)로 변환됩니다.

* **2.3 Vision Track (ColPali 방식의 멀티벡터 변환):**
  * 메모리에 상주하는 ColQwen2.5 모델을 통해 고해상도 이미지의 패치(Patch) 단위 시각 정보를 128차원의 멀티 벡터(`colpali_vectors`)로 즉시 변환하여, 정보 유실 없는 세밀한 검색(Late Interaction)을 준비합니다.

---

## 3. 검색 엔진 설계 (Elasticsearch 8.x)
단일 인덱스 내에서 텍스트와 이미지 벡터를 모두 수용하며, 하이브리드 검색을 수행합니다.

* **3.1 멀티 인덱스(Multi-Index) 구조:**
  * `univ_rules` (학칙 등 정적 데이터), `univ_notices` (공지 등 동적 데이터)로 인덱스 분리.
* **3.2 하이브리드 매핑 (Mapping):**
  * 문서 타입(`content_type`: text/image) 플래그를 둡니다.
  * 텍스트 요약본은 일반 `dense_vector`(bge-m3)로, 이미지 데이터는 `nested` 형태의 `dense_vector`(ColQwen2.5) 배열로 저장합니다.
* **3.3 하이브리드 검색 (Retrieval):**
  * 질문 유입 시 메타데이터(카테고리, 날짜)로 **Pre-filtering**을 수행합니다.
  * 이후 **BM25(키워드) + Dense(의미) + Late Interaction(비전 상세 대조)** 점수를 합산(RRF)하여 최적의 문서를 도출합니다.

---

## 4. 실시간 RAG 검색 및 답변 생성 (Online Querying)
정확한 정보 탐색과 빠른 응답 속도를 위해 쿼리 라우팅과 스트리밍을 도입합니다.

* **4.1 Query Routing (`chain.py`):**
  * 사용자의 질문 의도를 분석하여 검색할 특정 인덱스 및 필터를 동적으로 결정합니다.
* **4.2 통합 검색 (Hybrid Search in ES):**
  * Elasticsearch에 질의를 던져 최상위 컨텍스트(텍스트 요약본 + 이미지 멀티벡터가 가리키는 원문) K개를 추출합니다.
* **4.3 LLM 추론 및 스트리밍 (Generation):**
  * `vLLM`의 PagedAttention 기술이 적용된 Qwen2.5-7B가 추출된 컨텍스트를 바탕으로 답변을 작성합니다.
  * 응답은 SSE(Server-Sent Events)를 통해 프론트엔드로 즉시 스트리밍 전송됩니다.

---

## 5. RTX 4090 VRAM 예산 관리 (Resource Allocation - 24GB Limit)
모델들을 메모리에 동시 적재(Concurrent Loading)하여 오프라인/온라인 파이프라인 전환 없이 매끄럽게 동작하도록 24GB VRAM을 배분합니다.

* **Total VRAM:** 24.0 GB
* **LLM (Text Processing & Generation):** Qwen2.5-7B (8-bit 양자화 적용 시) -> **~8.0 GB**
  * *비고: `vLLM`을 통한 서빙으로 추론 속도 극대화*
* **Vision Encoder:** ColQwen2.5 -> **~4.5 GB**
* **Text Embedding:** BAAI/bge-m3 -> **~1.5 GB**
* **Buffer & KV Cache:** 잔여 메모리 **~10.0 GB**
  * *비고: 넉넉한 KV Cache 확보로 다수 사용자의 긴 문맥(Context) 멀티턴 대화를 안정적으로 처리*

---

## 6. 디렉토리 구조 (Directory Layout)
```text
CAPSTONE/
├── ai_engine/                 # AI 코어 로직
│   ├── loader.py              # 데이터 로드, 레이아웃 분석 및 Text/Vision 라우팅
│   ├── text_processor.py      # 로컬 sLM(vLLM) 연동 텍스트 요약, 메타데이터 추출 및 임베딩
│   ├── vision_processor.py    # ColQwen2.5 기반 시각 이미지 멀티벡터 변환
│   ├── es_client.py           # Elasticsearch 연동, 인덱스 생성 및 하이브리드 검색 쿼리
│   └── chain.py               # 쿼리 라우팅 및 최종 생성(Generation) 체인 관리
├── api/                       # 백엔드 서버
│   └── main.py                # FastAPI 엔드포인트 (스트리밍, 세션 관리)
├── crawler/                   # 데이터 수집 (Offline)
│   ├── hoseo_spider.py        # 호서대 공지사항 및 파일 크롤러
│   └── utils.py               # 중복 검사, 파일 다운로드 등 공통 함수
├── data/                      
│   └── raw/                   # [Git 제외] 원본 PDF, 이미지 물리적 저장소
├── docs/                      # 시스템 아키텍처 및 API 명세서
└── requirements.txt           # 파이썬 패키지 의존성 명세