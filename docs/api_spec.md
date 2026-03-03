# 📄 API Specification (v2.0) - Advanced RAG Backend
# 📄 Internal API Specification - AI Engine (FastAPI)

본 문서는 호서대학교 스마트 캠퍼스 도우미의 **메인 백엔드(Spring Boot, VM1)**와 **AI 엔진 서버(FastAPI, VM2)** 간의 내부 서버 통신(Server-to-Server) 규격을 정의합니다.

## 1. 기본 설정 (Global Settings)
- **Base URL:** `http://<VM2_Internal_IP>:8000/api/v1` (내부망 IP 사용)
- **Content-Type:** `application/json`
- **보안 (Authentication):** 외부의 직접 접근을 막기 위해 모든 요청 헤더에 내부 통신용 Secret Key를 포함해야 합니다.
  - `X-API-Key: {SHARED_SECRET_KEY}`

---

## 2. 챗봇 질의응답 (Streaming API)
Spring Boot가 사용자의 질문을 전달하면, FastAPI가 RAG 파이프라인을 거쳐 생성된 답변을 스트리밍(SSE) 방식으로 반환합니다. 
*(주의: Spring Boot는 이 SSE 응답을 WebFlux 등을 활용해 React로 릴레이(Relay)해야 합니다.)*

- **Endpoint:** `POST /chat/stream`
- **Headers:** - `X-API-Key: string`
- **Request Body:**
```json
{
  "user_id": "student_123",
  "session_id": "session_abc998",
  "question": "이번 2026학년도 장학금 신청 기한이 언제야?"
}


data: {"chunk": "2026학년도 "}
data: {"chunk": "1학기 "}
data: {"chunk": "장학금 신청 기한은 "}
data: {"chunk": "3월 15일까지입니다. "}
data: {"chunk": "", "sources": [{"doc_id": "notice_123", "title": "2026 장학금 안내", "file_url": "https://<VM1_Domain>/docs/123/안내문.pdf"}]}
data: [DONE]


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