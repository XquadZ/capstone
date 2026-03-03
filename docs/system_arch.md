# 🏛️ System Architecture - Advanced Multimodal RAG Backend (Local GPU Optimized)

이 문서는 RTX 4090 GPU 자원을 최대한 활용하여 구동되는 학교 챗봇의 고급 검색, 생성, 그리고 **멀티모달 데이터(이미지) 처리 로직**을 정의합니다.

## 1. 데이터 수집 및 전처리 (Data Ingestion & Multimodal Processing)
- **수집 파이프라인:** `crawler` 모듈을 통해 호서대 공지사항, 첨부파일(PDF/HWP), 본문 삽입 이미지를 `data/raw/` 로컬 폴더로 수집.
- **멀티모달 변환 (Vision-to-Text):** - 본문에 포함된 이미지(포스터, 안내문 등)는 멀티모달 LLM(GPT-4o API 또는 로컬 LLaVA 등)을 거쳐 상세한 텍스트 묘사로 변환됨.
  - 변환된 텍스트는 원본 게시글의 메타데이터(`info.json`)와 결합되어 RAG의 지식 베이스로 편입.
- **텍스트 분할 (Chunking):** - `RecursiveCharacterTextSplitter` 사용
  - `chunk_size`: 800자
  - `chunk_overlap`: 100자 (문맥 유지를 위해 앞뒤 100자씩 겹침)

## 2. 벡터 데이터베이스 & 임베딩 (Vector DB & Embedding)
- **라이브러리:** `ChromaDB` (로컬 디스크 저장 `database/` 폴더)
- **임베딩 모델:** `BAAI/bge-m3` 또는 `jhgan/ko-sroberta-multitask` (HuggingFace)
- **GPU 활용:** 텍스트 및 변환된 이미지 데이터의 임베딩 추출 시 RTX 4090의 CUDA 코어를 활용하여 처리 속도 극대화.
- **🔥 파인튜닝 목표:** 추후 학교 특화 용어(학사경고, 채플 등)를 학습시켜 검색 정확도를 높이기 위한 대조학습(Contrastive Learning) 대상.

## 3. 검색 및 답변 생성 (Advanced Retrieval & Generation)
단순 검색의 한계를 극복하기 위해 **2-Stage 검색(Reranking)**을 도입합니다.

- **1차 검색 (Retrieval):** 임베딩 기반 유사도 검색 (K=10, 넉넉하게 10개의 문서 추출)
- **2차 검색 (Reranking):** - **모델:** `Dongjin-kr/ko-reranker` (Cross-Encoder)
  - **역할:** 1차로 찾은 10개의 문서 중 질문과 진짜 관련 있는 최상위 3개 문서를 GPU 연산으로 재정렬 및 필터링
- **LLM 모델 (생성기):** - `EEVE-Korean-10.8B` 또는 `Llama-3-8B-Instruct` 등 한국어 특화 SLM(Small Language Model) 사용
  - `vLLM` 또는 `Ollama` 프레임워크를 통해 로컬 GPU(RTX 4090) 메모리에 올려서 API 형태로 서빙

## 4. 디렉토리 역할 (모듈화)
- `crawler/hoseo_spider.py`: (신규) 호서대 공지사항 웹 크롤링 및 이미지/PDF 로컬 다운로드
- `ai_engine/vision_processor.py`: (신규) 다운로드된 이미지를 멀티모달 LLM으로 분석하여 텍스트로 변환하는 모듈
- `ai_engine/loader.py`: PDF 파일 로드 및 메타데이터, 텍스트(이미지 변환 텍스트 포함) 병합 로직
- `ai_engine/vector_db.py`: HuggingFace 임베딩 로드 및 ChromaDB 저장/관리
- `ai_engine/reranker.py`: 검색된 문서를 재정렬(Reranking)하여 정확도를 높이는 로직
- `ai_engine/chain.py`: [검색 -> 재정렬 -> 로컬 LLM 생성]으로 이어지는 파이프라인 총괄
- `api/main.py`: FastAPI 서버 실행 (`api_spec.md` 규격 구현)

## 5. 웹 서비스 및 실시간성 최적화 (New)
- **비동기 메시지 큐 (Redis + Celery):** - 크롤링이나 대규모 임베딩 작업은 FastAPI와 분리하여 백그라운드에서 실행. 사용자의 질의 응답 속도에 영향을 주지 않도록 설계.
- **세션 및 캐싱 (Redis):**
  - **대화 이력 관리:** 사용자별 최근 대화 5~10개를 Redis에 저장하여 멀티턴(Multi-turn) 대화 구현.
  - **답변 캐싱:** 빈번한 동일 질문에 대해 LLM 추론 없이 즉시 응답하여 GPU 부하 감소.
- **서빙 엔진 최적화:**
  - `vLLM`의 연속 배치(Continuous Batching) 기능을 활용하여 동시 접속 시 처리량(Throughput) 극대화.