# 🕷️ Crawler & Ingestion Module (Data Pipeline)

본 모듈은 호서대학교 스마트 캠퍼스 도우미(RAG 시스템)의 지식 베이스를 구축하기 위한 데이터 수집 및 전처리 엔진입니다. 
**라이브 시연 및 웹 서비스의 안정성을 위해 본 모듈은 사용자 API 요청과 완전히 분리되어 백그라운드(배치 작업)로만 실행됩니다.**

[Image of a web scraping and data ingestion pipeline for RAG systems integrating multimodal vision to text]

## 1. 수집 대상 및 핵심 전략 (Target & Strategy)

- **대상 사이트:** 호서대학교 웹사이트 공지사항 게시판
- **타겟 URL:** `https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_17082401398`
- **핵심 전략 (Idempotency):** - 맹목적인 전체 크롤링을 방지합니다. 
  - 공지사항 목록에서 **고유 번호(ID)**를 먼저 추출한 뒤, `Redis`에 이미 존재하는 ID인지 확인합니다. 
  - 존재하지 않는 **신규 게시글만 상세 페이지로 진입**하여 수집 속도를 높이고 서버 부하를 최소화합니다.

---

## 2. 기술 스택 (Dependencies)

| 라이브러리/도구 | 역할 |
| :--- | :--- |
| **Selenium** | JavaScript로 렌더링되는 동적 페이지 탐색 및 페이지네이션(페이징) 처리 |
| **BeautifulSoup4** | 로드된 HTML 문서의 세부 파싱, 본문 텍스트 및 이미지/첨부파일 태그(`<img>`, `<a>`) 추출 |
| **Requests / urllib** | 파싱된 URL을 통해 PDF, HWP, 이미지 파일을 로컬로 바이너리 다운로드 |
| **로컬 LLaVA / GPT-4o** | 수집된 본문 이미지(포스터 등)를 멀티모달 처리하여 텍스트 형태의 시각적 묘사(Description)로 변환 |

---

## 3. 데이터 처리 워크플로우 (Step-by-Step)

1. **탐색 및 중복 검사 (Discovery):**
   - 게시판 목록 1페이지부터 순회하며 공지글 ID 추출.
   - `Redis.exists(f"notice:{id}")` 검사를 통해 신규 글 식별.
2. **다운로드 (Scraping & Download):**
   - 본문 텍스트 크롤링.
   - 게시글에 첨부된 `.pdf`, `.hwp` 파일과 본문에 삽입된 이미지 파일을 `data/raw/[ID]/` 폴더에 다운로드.
3. **멀티모달 변환 (Vision-to-Text):**
   - 다운로드된 이미지를 `vision_processor` 모듈로 전달.
   - 모델이 이미지를 분석하여 "2026학년도 장학금 신청 안내 포스터입니다. 신청 기간은..." 형태의 텍스트로 변환.
4. **텍스트 병합 및 청킹 (Merge & Chunking):**
   - 원본 텍스트 + 이미지 변환 텍스트 + PDF 파싱 텍스트를 하나로 병합.
   - `RecursiveCharacterTextSplitter`를 사용해 800자 단위(Overalp 100자)로 분할(Chunking).
5. **DB 적재 (Ingestion to Redis):**
   - 메타데이터(제목, 작성일, 원본 경로)는 **Redis JSON**으로 저장.
   - 분할된 텍스트 청크는 임베딩(`bge-m3`) 후 **Redis Vector** 필드에 적재.

---

## 4. 로컬 저장소 및 메타데이터 구조 (Schema)

크롤링이 완료되면 로컬 디스크에는 원본 파일이, Redis에는 가공된 검색용 데이터가 저장됩니다.

### 4.1 로컬 물리 디렉토리 구조
데이터 유실을 막고 프론트엔드에서 원본을 열람할 수 있도록 고유 ID별 폴더를 생성합니다.
```text
CAPSTONE/data/raw/
└── 170824/                 # 공지사항 고유 ID
    ├── info.json           # 크롤링된 메타데이터 원본 백업
    ├── images/             # 본문 삽입 이미지 보관
    │   └── poster_01.jpg
    └── attachments/        # 다운로드된 첨부파일
        └── 장학금안내.pdf



{
  "notice_id": "170824",
  "title": "2026학년도 1학기 장학금 신청 안내",
  "date": "2026-03-04",
  "author": "교무처",
  "category": "장학",
  "content_text": "병합된 전체 텍스트 내용...",
  "source_links": [
    "/docs/raw/170824/attachments/장학금안내.pdf",
    "/docs/raw/170824/images/poster_01.jpg"
  ]
}