# 🕷️ Crawler Module (데이터 수집 및 멀티모달 전처리 엔진)

이 모듈은 RAG(검색 증강 생성) 시스템 구성을 위한 대학 공지사항, 첨부파일(PDF 등), 그리고 **본문 삽입 이미지** 데이터를 수집하는 역할을 담당합니다. 수집된 이미지는 향후 멀티모달 LLM(GPT-4o 등)을 통해 텍스트로 변환되어 지식 베이스에 통합됩니다.

## 🎯 수집 대상 (Target)
- **대상 사이트:** 호서대학교 공지사항 게시판
- **대상 URL:** `https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_17082401398`
- **수집 범위:** 게시글 제목, 작성일, 본문 텍스트, 첨부파일(PDF/HWP) 링크 및 다운로드, **본문 내 이미지 파일 로컬 저장**

## 🛠️ 사용 기술 및 라이브러리 (Dependencies)
- **Selenium**: JavaScript 렌더링 대응 및 동적 데이터 추출 (메인 엔진)
- **WebDriver Manager**: 크롬 드라이버 자동 관리 및 버전 동기화
- **BeautifulSoup4**: Selenium으로 로드된 HTML의 세부 파싱 및 태그 추출
- **Requests/Urllib**: 이미지 및 첨부파일의 바이너리 다운로드 (성능 최적화)
- (`추가 예정`): 이미지 내 텍스트 추출(Vision LLM)을 위한 API 연동 로직

## 📂 데이터 저장 구조 (Output Schema & Directory)
각 공지사항은 관리의 용이성과 멀티모달 처리를 위해 **고유 ID를 이름으로 하는 개별 폴더**에 저장됩니다.

**디렉토리 구조 예시:**
```text
../data/raw/
└── [게시글 고유 번호]/
    ├── info.json       # 메타데이터 및 텍스트
    ├── images/         # 본문 추출 이미지 모음
    │   ├── img_01.jpg
    │   └── img_02.png
    └── attachments/    # 다운로드된 첨부파일
        └── 안내문.pdf