# 🏛️ System Architecture - Advanced RAG Backend (Local GPU Optimized)

이 문서는 RTX 4090 GPU 자원을 최대한 활용하여, 외부 API 의존 없이 로컬 환경에서 구동되는 학교 챗봇의 고급 검색 및 생성 로직(Advanced RAG)을 정의합니다.

## 1. 데이터 전처리 (Data Ingestion)
- **대상:** `data/` 폴더 내 학교 규정집 및 공지사항 (`.pdf`, `.txt`)
- **텍스트 분할 (Chunking):** - `RecursiveCharacterTextSplitter` 사용
  - `chunk_size`: 800자
  - `chunk_overlap`: 100자 (문맥 유지를 위해 앞뒤 100자씩 겹침)

## 2. 벡터 데이터베이스 & 임베딩 (Vector DB & Embedding)
- **라이브러리:** `ChromaDB` (로컬 디스크 저장 `database/` 폴더)
- **임베딩 모델:** `BAAI/bge-m3` 또는 `jhgan/ko-sroberta-multitask` (HuggingFace)
- **GPU 활용:** 임베딩 추출 시 RTX 4090의 CUDA 코어 활용 처리 속도 극대화
- **🔥 파인튜닝 목표:** 추후 학교 특화 용어(학사경고, 채플 등)를 학습시켜 검색 정확도를 높이기 위한 대조학습(Contrastive Learning) 대상.

## 3. 검색 및 답변 생성 (Advanced Retrieval & Generation)
단순 검색의 한계를 극복하기 위해 **2-Stage 검색(Reranking)**을 도입합니다.

- **1차 검색 (Retrieval):** 임베딩 기반 유사도 검색 (K=10, 넉넉하게 10개의 문서 추출)
- **2차 검색 (Reranking):** - **모델:** `Dongjin-kr/ko-reranker` (Cross-Encoder)
  - **역할:** 1차로 찾은 10개의 문서 중 질문과 진짜 관련 있는 최상위 3개 문서를 GPU 연산으로 재정렬 및 필터링
- **LLM 모델 (생성기):** - `EEVE-Korean-10.8B` 또는 `Llama-3-8B-Instruct` 등 한국어 특화 SLM(Small Language Model) 사용
  - `vLLM` 또는 `Ollama` 프레임워크를 통해 로컬 GPU(RTX 4090) 메모리에 올려서 API 형태로 서빙

## 4. 디렉토리 역할 (모듈화)
- `ai_engine/loader.py`: PDF 파일 로드 및 텍스트 추출 (PyMuPDF 등 활용)
- `ai_engine/vector_db.py`: HuggingFace 임베딩 로드 및 ChromaDB 저장/관리
- `ai_engine/reranker.py`: 검색된 문서를 재정렬(Reranking)하여 정확도를 높이는 로직
- `ai_engine/chain.py`: [검색 -> 재정렬 -> 로컬 LLM 생성]으로 이어지는 파이프라인 총괄
- `api/main.py`: FastAPI 서버 실행 (`api_spec.md` 규격 구현)