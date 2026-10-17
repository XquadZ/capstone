# 신규 크롤 공지 상단 토스트 연동

## 목표

증분 크롤이 **새로 성공한 공지**만 Spring 웹앱([Hoseo LENS test.html](https://wrecker-motivator-overall.ngrok-free.dev/test.html))에서:

1. 페이지 상단 슬라이드 토스트
2. **Windows 시스템 알림** (브라우저 Notification API, 사용자 허용 필요)

로 잠깐 보여 주고 사라지게 함.

## 데이터 흐름

```text
증분 크롤 (신규만)
  ├─ POST Spring /api/notices/new     ← DB·키워드 알림용
  └─ POST AI    /notices/alerts/publish
                    └─ SSE /notices/alerts/stream
                           └─ test.html 상단 토스트
```

표시 형식: `[대분류] 공지 제목`  
대분류 폐쇄집합: 학사/수업, 장학/등록금/근로, 취업/진로/채용, 창업/산학, 국제/교류/연수, 비교과/특강/행사, 모집/선발/신청, 생활/복지, 안전/보안/시설, 연구/사업/행정, 기타/일반  
- 로컬/AI 토스트: `category`=대분류, `display`=`[대분류] 제목`, 세부=`sub_category`
- **Spring 웹훅**: `category`에만 대분류 정확 일치, `major_category`는 빈 문자열  
  (둘 다 넣으면 백엔드가 `major/category`로 합쳐 `비교과/특강/행사/비교과/특강/행사`가 됨)

## AI 쪽 (이미 적용)

- `.env`
  - `NOTICE_EVENT_WEBHOOK_URL=https://wrecker-motivator-overall.ngrok-free.dev/api/notices/new`
  - `NOTICE_ALERT_PUBLISH_URL=http://127.0.0.1:8000/notices/alerts/publish`
- AI API 엔드포인트
  - `POST /notices/alerts/publish`
  - `GET  /notices/alerts/stream` (SSE)

AI ngrok이 떠 있어야 브라우저가 SSE를 받음 (예: `https://wriggle-script-caption.ngrok-free.dev`).

## Spring / 정적 파일 쪽 (배포 필요)

1. 이 폴더의 `test.html` + `sw-notice.js` 를 Spring `static/` (또는 현재 서빙 경로)에 **같이** 배포  
   (`sw-notice.js`는 Windows 알림 클릭 시 공지 URL을 여는 Service Worker)
2. AI ngrok 주소가 바뀌면 `test.html` 안의 `AI_TOAST_SSE_URL` 기본값을 수정  
   또는 브라우저에서:
   ```js
   localStorage.setItem('hoseo_toast_sse', 'https://새AI엔그록/notices/alerts/stream')
   ```
   또는 URL: `test.html?toast_sse=https://새AI엔그록/notices/alerts/stream`

## 확인 방법

1. AI API + AI ngrok + 증분 크롤 실행
2. 브라우저에서 test.html 연 상태로 두기
3. 크롤이 신규 공지를 잡으면 상단에 NEW 토스트가 내려옴
4. 수동 테스트:
   ```bash
   curl -X POST http://127.0.0.1:8000/notices/alerts/publish \
     -H "Content-Type: application/json" \
     -d "{\"items\":[{\"notice_id\":\"1\",\"category\":\"학사\",\"title\":\"테스트\",\"display\":\"[학사] 테스트\",\"url\":\"https://www.hoseo.ac.kr/\"}]}"
   ```
