# 🏛️ System Architecture - RAG Backend

이 문서는 학교 챗봇의 데이터 처리 및 검색 로직(RAG)을 정의합니다.

## 1. 데이터 전처리 (Data Ingestion)
- **대상:** `data/` 폴더 내 모든 `.pdf`, `.txt` 파일
- **텍스트 분할 (Chunking):** - `RecursiveCharacterTextSplitter` 사용
  - `chunk_size`: 800자
  - `chunk_overlap`: 100자 (문맥 유지를 위해 앞뒤 100자씩 겹침)

## 2. 벡터 데이터베이스 (Vector DB)
- **라이브러리:** `ChromaDB`
- **저장 위치:** `database/` 폴더 내에 로컬 저장
- **임베딩 모델:** OpenAI `text-embedding-3-small` (또는 HuggingFace 무료 모델)

## 3. 검색 및 답변 생성 (Retrieval & Generation)
- **Search Type:** 유사도 기반 검색 (Similarity Search)
- **K-value:** 3 (질문과 가장 유사한 문서 조각 3개를 가져옴)
- **LLM 모델:** `gpt-4o` 또는 `gpt-3.5-turbo`
- **Chain:** `create_retrieval_chain`을 사용하여 검색된 문서를 바탕으로 답변 생성

## 4. 디렉토리 역할
- `ai_engine/loader.py`: PDF 파일 로드 및 텍스트 추출 담당
- `ai_engine/vector_db.py`: 임베딩 및 ChromaDB 저장/로드 담당
- `ai_engine/chain.py`: 검색 및 LLM 답변 생성 프로세스 총괄
- `api/main.py`: FastAPI 서버 실행 및 `api_spec.md` 규격 구현