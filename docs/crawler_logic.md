# 🕷️ Crawler Module (데이터 수집 엔진)

이 모듈은 RAG(검색 증강 생성) 시스템 구성을 위한 대학 공지사항 및 첨부파일(PDF 등) 데이터를 수집하는 역할을 담당합니다.

## 🎯 수집 대상 (Target)
- **대상 사이트:** 호서대학교 공지사항 게시판
- **대상 URL:** `https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_1708240139`
- **수집 범위:** 게시글 제목, 작성일, 본문 내용, 첨부파일(PDF) 링크 및 다운로드

## 🛠️ 사용 기술 및 라이브러리 (Dependencies)
- `requests`: HTTP 요청 및 웹페이지 가져오기
- `beautifulsoup4`: HTML 파싱 및 데이터 추출
- (`추가 예정`): 동적 페이지 렌더링이 필요한 경우 Selenium 적용 고려

## 📂 데이터 저장 구조 (Output Schema)
수집된 데이터는 `../data/raw/` 경로에 JSON 형태로 저장되며, 구조는 다음과 같습니다.
```json
{
  "id": "게시글 고유 번호",
  "title": "공지사항 제목",
  "date": "작성일 (YYYY-MM-DD)",
  "content": "본문 텍스트",
  "attachments": [
    {
      "file_name": "첨부파일 이름.pdf",
      "download_url": "다운로드 링크"
    }
  ]
}