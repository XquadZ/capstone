# 📄 API Specification (v2.0) - Advanced RAG Backend

본 문서는 호서대학교 스마트 캠퍼스 도우미 챗봇의 프론트엔드(React)와 백엔드(FastAPI) 간의 통신 규격을 정의합니다. 
라이브 시연의 체감 속도를 극대화하기 위해 메인 질의응답은 **SSE(Server-Sent Events) 기반의 스트리밍 방식**을 채택했습니다.

## 1. 기본 설정 (Global Settings)
- **Base URL:** `http://localhost:8000/api/v1`
- **CORS Policy:** React 개발 서버(`http://localhost:3000` 등)에서의 접근을 허용하도록 FastAPI 미들웨어 설정 필수.

---

## 2. 챗봇 질의응답 (Streaming API)
사용자의 질문을 받아 RAG 파이프라인을 거친 후, LLM의 생성 결과를 한 글자씩 실시간으로 전송합니다.

- **Endpoint:** `POST /chat/stream`
- **Content-Type:** `application/json` (Request) / `text/event-stream` (Response)

### Request Body
멀티턴(Multi-turn) 대화를 위해 반드시 `session_id`를 포함해야 합니다.
```json
{
  "user_id": "student_01",
  "session_id": "session_9982",
  "question": "이번 2026학년도 장학금 신청 기한이 언제야?"
}



{
  "status": "success",
  "session_id": "session_9982",
  "history": [
    {
      "role": "user",
      "content": "올해 성적장학금 기준이 뭐야?",
      "timestamp": "2026-03-03T23:45:12Z"
    },
    {
      "role": "assistant",
      "content": "성적장학금 기준은 직전 학기 평점 3.5 이상입니다. [출처: 장학규정 시행세칙]",
      "timestamp": "2026-03-03T23:45:15Z"
    }
  ]
}



{
  "status": "empty",
  "message": "새로운 대화를 시작합니다.",
  "history": []
}



{
  "status": "healthy",
  "gpu_vram_usage": "14.2GB / 24.0GB",
  "redis_connected": true,
  "active_sessions": 3
}