# API Spec (2026-05)

프론트·Spring·AI 팀 연동용 API 계약입니다.

---

## A. AI 서버 (FastAPI) — `backend/api_service.py`

Spring이 **직접** 호출하는 엔드포인트입니다. (Flutter는 Spring만 호출)

### A.1 기본

- Base URL 예: `https://{ngrok-host}` (**끝에 `/ask` 없이**)
- Path: `POST /ask`
- Content-Type: `application/json`
- ngrok 무료 도메인: 요청 헤더 `ngrok-skip-browser-warning: true` 권장

### A.2 Request

```json
{
  "question": "사용자 질문",
  "domain": "notice",
  "use_tv_rag": true
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `question` | string | O | - | 1자 이상 |
| `domain` | string | X | `notice` | **`notice`** \| **`rules`** 만 |
| `use_tv_rag` | boolean | X | `true` | TV-RAG(라우터+TEXT/VISION) vs 단일 텍스트 RAG |

**Spring `category` → AI `domain` 매핑 (권장)**

| 프론트/백 `category` | AI `domain` |
|---------------------|-------------|
| `rules`, `학칙` | `rules` |
| 그 외 (`일반`, 공지 등) | `notice` |

### A.3 Response (200)

```json
{
  "domain": "notice",
  "question": "...",
  "answer": "최종 답변 전문",
  "route": "TEXT",
  "latency_sec": 12.34,
  "contexts": ["..."],
  "sources": ["[공지-학사] 97080"],
  "meta": {
    "pipeline": "agentic_tv_rag",
    "domain": "notice",
    "use_tv_rag": true,
    "provider": "saifex",
    "router_raw": "TEXT"
  }
}
```

### A.4 Errors

| HTTP | 내용 |
|------|------|
| 400 | `domain`이 `notice`/`rules`가 아님 |
| 422 | `question` 누락 등 |
| 500 | Milvus/LLM/RAG 내부 오류 |

### A.5 Health

- `GET /health` → `{"status":"ok","service":"tv-rag-api","ts":...}`

### A.6 Preview (개발용)

- `GET /ask-preview?q=...&domain=notice&use_tv_rag=true`

---

## B. Spring 백엔드 — Flutter 연동

### B.1 Request (앱 → Spring)

`POST /api/chat/ask` (SSE)

```json
{
  "user_id": "deviceId",
  "session_id": "sessionId",
  "question": "질문",
  "category": "일반"
}
```

### B.2 Response (Spring → 앱, SSE)

```text
data: {"chunk":"단어 "}
data: {"chunk":"", "sources":["..."]}
data: [DONE]
```

- 3초마다 comment `ping` (heartbeat) 가능
- 실패 시: `{"error":{"code":"CONNECTION_FAILED","message":"..."}}`

### B.3 Spring → AI 내부 호출

위 **A절** `POST /ask` 사용. `answer`를 받은 뒤 단어 단위로 SSE 분할(약 30ms 간격).

---

## C. 환경 변수 (AI)

| 변수 | 용도 |
|------|------|
| `SAIFEX_API_KEY` | **필수** — `api_service` 부팅 시 설정 |
| `API_HOST` / `API_PORT` | 기본 `0.0.0.0:8000` |
| `NOTICE_EVENT_WEBHOOK_URL` | 증분 크롤 후 Spring 알림 (선택) |

---

## D. 프론트 체크리스트

- SSE URL은 **Spring** (`/api/chat/ask`), AI ngrok URL 아님
- `[DONE]` 및 `{"chunk":...}` JSON 파싱
- `sources` 이벤트 형식 처리
- 긴 SSE 시 타임아웃·백그라운드 연결 유지

## E. 장애 구간 판별

| Spring 로그 | 구간 |
|-------------|------|
| `RAG 서버 오류` | Spring → AI |
| `RAG 응답 수신` + `청크 전송 중 클라이언트 연결 끊김` | AI OK, 앱/SSE |
| AI 터미널 `[ASK] A:` | AI 답변 생성 완료 |

---

## F. 관련 문서

- [frontend_srs.md](frontend_srs.md) — UI 요구사항
- [system_arch.md](system_arch.md) — 아키텍처
- [infra_setup.md](infra_setup.md) — TV-RAG 3터미널 실행
