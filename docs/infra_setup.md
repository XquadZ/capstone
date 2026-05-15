# Infrastructure Setup Guide (2026-05)

Milvus + TV-RAG API + (선택) ngrok + 크롤러 실행 가이드입니다.

## 1. 권장 환경

- GPU: RTX 4090 (권장, 라우터·BGE-M3·OCR)
- OS: Windows 10/11
- Python: 3.10+ (conda `capstone_final` 등)
- Docker Desktop
- Chrome (Selenium 크롤)

## 2. Milvus

```powershell
cd C:\Users\DMLAB_Server1\capstone
docker compose up -d
docker ps
```

- 포트: **19530** (gRPC), **9091** (metrics)
- 컨테이너: `milvus-etcd`, `milvus-minio`, `milvus-standalone`

## 3. Python 의존성

```powershell
pip install -r requirements.txt
pip install pymilvus FlagEmbedding openai python-dotenv fastapi uvicorn schedule selenium webdriver-manager
```

비전·첨부 처리:

```powershell
pip install pymupdf easyocr pillow olefile pdf2image
```

## 4. 환경 변수

| 변수 | 필수 | 용도 |
|------|------|------|
| `SAIFEX_API_KEY` | **TV-RAG API 필수** | `backend/api_service.py` |
| `OPENAI_API_KEY` | 일부 스크립트 | 정제·평가 등 |
| `NOTICE_EVENT_WEBHOOK_URL` | 선택 | 증분 크롤 → Spring 웹훅 |
| `NOTICE_EVENT_API_KEY` | 선택 | 웹훅 인증 |

PowerShell:

```powershell
$env:SAIFEX_API_KEY = "..."
```

`.env`를 프로젝트 루트에 두면 `GPTRefiner` 등이 로드합니다.

## 5. TV-RAG 서버 실행 (3터미널)

### 터미널 1 — Milvus

```powershell
cd C:\Users\DMLAB_Server1\capstone
docker compose up -d
```

### 터미널 2 — AI API

```powershell
cd C:\Users\DMLAB_Server1\capstone
conda activate capstone_final
python -m backend.api_service
```

- 워밍업: BGE-M3 + Gemma 라우터 + (첫 추론) — 수 분 소요 가능
- 확인: `http://localhost:8000/health`

### 터미널 3 — ngrok (Spring 연동 시)

```powershell
ngrok http 8000
```

Spring `rag.server.url` = `https://xxxx.ngrok-free.dev` (경로 `/ask` 제외)

### 로컬 테스트

```powershell
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"테스트\",\"domain\":\"notice\",\"use_tv_rag\":true}"
```

## 6. 크롤러·증분 업데이트

```powershell
conda activate capstone_final
cd C:\Users\DMLAB_Server1\capstone

# 최신 공지만 (1회)
python -m crawler.incremental_notice_update --once

# 스케줄 (10:00~17:30, 30분 간격)
python -m crawler.incremental_notice_update

# 전체 백필 (미등록 schIdx만)
python -m crawler.crawl_all --no-webhook
```

## 7. 데이터 경로

```text
data/raw/{notice_id}/
data/processed/integrated_text/
data/processed/text/
data/processed/chunks/
data/rules_regulations/   # 학칙 PDF·청크
```

`data/`, `volumes/` — Git 미추적.

## 8. 라우터 어댑터

기본 경로: `experience/exp1/gemma_router_lora_v4/`  
(`AgenticRAG/nodes/router.py`의 `ADAPTER_PATH`)

## 9. 트러블슈팅

| 증상 | 조치 |
|------|------|
| Milvus 연결 실패 | `docker compose ps`, 19530 |
| `SAIFEX_API_KEY` 없음 | API 서버 기동 실패 |
| `domain은 notice 또는 rules만` | Spring에서 `category` 매핑 |
| Milvus `offset+limit` 16384 | `query_iterator` 사용 (이미 반영) |
| 크롬 `InvalidSessionId` | `crawl_all` 세션 재시작 로직 |
| ngrok HTML 응답 | `ngrok-skip-browser-warning` 헤더 |

## 10. 관련 문서

- [api_spec.md](api_spec.md)
- [crawler_logic.md](crawler_logic.md)
- [system_arch.md](system_arch.md)
