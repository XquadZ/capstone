# Infrastructure Setup Guide (Current)

본 문서는 현재 저장소에서 실제로 사용 중인 인프라(Milvus + Python 실행 스크립트) 기준으로 정리한다.

## 1. 권장 환경

- GPU: RTX 4090 (권장)
- OS: Windows 10/11 또는 Ubuntu 22.04+
- Python: 3.10+
- Docker Desktop 또는 Docker Engine

## 2. Milvus 구동

저장소 루트에서 아래 명령으로 Milvus 스택을 실행한다.

```bash
docker compose up -d
```

구성 서비스(`docker-compose.yml`):
- `milvus-etcd`
- `milvus-minio`
- `milvus-standalone` (port `19530`, `9091`)

정상 확인:
```bash
docker ps
```

## 3. Python 의존성 설치

`requirements.txt`는 최소 패키지만 포함하므로, 실제 실행 스크립트 기준 추가 설치가 필요할 수 있다.

```bash
pip install -r requirements.txt
pip install pymilvus FlagEmbedding openai python-dotenv
```

필요 시(파일/비전 처리):
```bash
pip install pymupdf easyocr pillow olefile
```

## 4. 환경 변수

실행 시나리오에 따라 아래 키를 사용한다.

- `OPENAI_API_KEY` : OpenAI 직접 호출 스크립트
- `SAIFEX_API_KEY` : SAIFEX endpoint 호출 스크립트

PowerShell 예시:
```powershell
$env:OPENAI_API_KEY="..."
$env:SAIFEX_API_KEY="..."
```

## 5. 데이터 경로 전제

코드가 기대하는 기본 경로:
- `data/raw/`
- `data/processed/`
- `data/rules_regulations/raw_pdfs/`
- `data/rules_regulations/markdown_parsed/`
- `data/rules_regulations/chunks/`

`data/`는 보통 Git 추적 제외이므로 별도 준비가 필요하다.

## 6. 실행 순서 (권장)

### 공지 파이프라인
1. `python ai_engine/full_text_extractor.py`
2. `python ai_engine/local_slm_refiner.py`
3. `python ai_engine/chunker.py`
4. `python ai_engine/vector_db.py`
5. `python ai_engine/rag_pipeline.py`

### 학칙 파이프라인
1. `python ai_engine/md_parser_pdf.py`
2. `python ai_engine/rule_data_chunker.py`
3. `python ai_engine/vector_db_rules.py`
4. `python ai_engine/rag_pipeline_rules.py`

## 7. 트러블슈팅

- Milvus 연결 실패: `localhost:19530` 포트 점유/컨테이너 상태 확인
- 인코딩 깨짐(Windows): 스크립트 내 UTF-8 래핑 사용 또는 터미널 UTF-8 설정
- OOM: batch size 축소, retrieve_k/top_k 축소, 비전 모델 동시 실행 지양