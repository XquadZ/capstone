# API Spec (Current Baseline)

이 문서는 현재 저장소 기준으로 프론트엔드/백엔드 연동 시 사용할 최소 API 계약을 정의한다.  
현재 코드에는 API 서버 구현 파일이 포함되어 있지 않으므로, 아래는 `docs/frontend_srs.md`와 평가 스크립트에서 사용한 포맷을 기준으로 한 **권장 표준**이다.

## 1. 기본 정책

- Base URL: `http://<AI_ENGINE_HOST>:8000/api/v1`
- Content-Type: `application/json`
- 인증: 운영 환경에서는 `Authorization` 또는 `X-API-Key` 중 1개 표준화 필요

## 2. 스트리밍 채팅 API

### 2.1 Request
- Method: `POST`
- Path: `/chat/stream`

요청 바디:
```json
{
  "user_id": "student_123",
  "session_id": "session_abc998",
  "question": "이번 2026학년도 장학금 신청 기한이 언제야?"
}
```

### 2.2 Response (SSE)
응답은 `text/event-stream`으로 내려가며, `data:` 라인 단위로 파싱한다.

```text
data: {"chunk":"2026학년도 "}
data: {"chunk":"1학기 장학금 신청 기한은 ..."}
data: {"chunk":"", "sources":[{"doc_id":"notice_123","title":"2026 장학금 안내","file_url":"https://..."}]}
data: [DONE]
```

### 2.3 SSE 이벤트 규칙

- `chunk`는 누적 가능한 문자열 조각이어야 한다.
- 마지막 이벤트 전후로 `sources`를 1회 이상 제공한다.
- 종료는 `data: [DONE]`으로 명시한다.
- 오류 발생 시 표준 JSON 이벤트 제공을 권장:

```text
data: {"error":{"code":"UPSTREAM_TIMEOUT","message":"LLM 응답 지연"}}
data: [DONE]
```

## 3. 세션 API (권장)

현재 저장소에는 세션 API 구현이 없으므로 아래 엔드포인트를 권장 표준으로 정의한다.

### 3.1 대화 이력 조회
- Method: `GET`
- Path: `/chat/history/{session_id}`

응답 예시:
```json
{
  "status": "success",
  "session_id": "session_abc998",
  "history": [
    {
      "role": "user",
      "content": "성적장학금 기준이 뭐야?",
      "timestamp": "2026-03-21T12:00:00Z"
    },
    {
      "role": "assistant",
      "content": "직전 학기 평점 3.5 이상입니다.",
      "timestamp": "2026-03-21T12:00:02Z"
    }
  ]
}
```

### 3.2 세션 초기화
- Method: `POST`
- Path: `/chat/session/new`

응답 예시:
```json
{
  "status": "success",
  "session_id": "session_new_001"
}
```

## 4. 헬스체크 API (권장)

- Method: `GET`
- Path: `/health`

응답 예시:
```json
{
  "status": "healthy",
  "milvus_connected": true,
  "llm_available": true
}
```

## 5. 프론트엔드 구현 체크포인트

- SSE 파서에서 `[DONE]`와 오류 이벤트를 모두 처리할 것
- 스트리밍 중단 시 부분 응답을 보존할 것
- `sources`가 누락될 수 있는 상황을 UI에서 허용할 것