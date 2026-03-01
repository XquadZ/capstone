# 📄 API Specification (v1.0) - 학교 챗봇 프로젝트

이 문서는 프론트엔드(React)와 백엔드(Python/FastAPI) 간의 통신 규격을 정의합니다.

## 1. 기본 설정
- **Base URL:** `http://localhost:8000`
- **Content-Type:** `application/json`

---

## 2. 질문하기 (Ask Question)
사용자가 챗봇에게 질문을 던지고 답변을 받는 메인 API입니다.

### [POST] /ask

**Request Body:**
```json
{
  "question": "2026년 장학금 신청 기간이 언제야?",
  "user_id": "test_user_01"
}


{
  "answer": "2026년 1학기 장학금 신청 기간은 3월 2일부터 3월 15일까지입니다.",
  "status": "success",
  "sources": [
    {
      "file_name": "2026_학사일정.pdf",
      "page": 3
    },
    {
      "file_name": "장학금_안내문.txt",
      "page": 1
    }
  ],
  "timestamp": "2026-03-01T21:30:00Z"
}


{
  "status": "error",
  "message": "죄송합니다. 관련 문서를 찾을 수 없어 답변이 어렵습니다. 학사지원팀(041-xxx-xxxx)으로 문의해주세요."
}