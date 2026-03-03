# 🎓 호서대학교 스마트 캠퍼스 도우미 (Advanced Multimodal RAG)

본 프로젝트는 호서대학교의 공지사항, 학사 일정, 장학금 정보를 실시간으로 수집하고, 사용자의 질문에 정확한 출처와 함께 답변을 제공하는 **실시간 스트리밍 RAG 챗봇 시스템**입니다. 
단일 RTX 4090(24GB) 로컬 서버 환경에서 VRAM을 극대화하여 활용하며, 환각(Hallucination) 없는 고성능 AI 서비스를 제공하는 것을 목표로 합니다.

[Image of a modern GitHub README landing page with tech stack badges for React, FastAPI, Redis, and vLLM]

## 🚀 기술 스택 (Core Tech Stack)
기존의 단순 검색(ChromaDB) 방식에서 벗어나, 속도와 세션 관리를 위한 **Redis 하이브리드 아키텍처**로 고도화되었습니다.

- **Frontend:** React (SSE 스트리밍 응답 처리)
- **Backend:** FastAPI (비동기 API 및 세션 관리)
- **Database:** Redis Stack (세션 캐싱, JSON 메타데이터, Vector 검색 통합)
- **AI & ML:** vLLM (EEVE-10.8B-AWQ), BAAI/bge-m3, ko-reranker, 로컬 Vision LLM
- **Data Pipeline:** Selenium, BeautifulSoup4, LangChain

---

## 📚 공식 문서 허브 (Documentation)
프로젝트의 상세한 설계 및 세팅 가이드는 아래의 문서를 확인해 주세요.

| 문서명 | 설명 | 링크 |
| :--- | :--- | :--- |
| **시스템 아키텍처** | 전체 시스템 구조, VRAM 분배 및 RAG 파이프라인 | [system_arch.md](docs/system_arch.md) |
| **API 명세서** | 프론트-백엔드 간 SSE 스트리밍 통신 규격 | [api_spec.md](docs/api_spec.md) |
| **인프라 세팅 가이드** | RTX 4090 환경의 Docker 및 패키지 설치 방법 | [infra_setup.md](docs/infra_setup.md) |
| **데이터 파이프라인** | 공지사항 크롤링 및 멀티모달 이미지 전처리 로직 | [crawler_logic.md](docs/crawler_logic.md) |
| **프롬프트 및 페르소나** | 환각 방지 및 대화 문맥 유지를 위한 시스템 프롬프트 | [prompt_rules.md](docs/prompt_rules.md) |

---

## 📂 프로젝트 구조 (Project Structure)
```text
CAPSTONE/
├── ai_engine/          # 핵심 AI 로직 (임베딩, 리랭킹, 멀티모달 변환)
├── api/                # FastAPI 서버 엔드포인트 및 RAG 라우팅
├── crawler/            # 오프라인 데이터 수집 모듈 (중복 방지 로직 포함)
├── data/               # [Git Ignored] 원본 수집 데이터 (PDF, HWP, Images)
├── docs/               # 설계 문서 및 API 규격서
├── venv/               # [Git Ignored] Python 로컬 가상환경
├── .gitignore          # Git 제외 목록 (보안 키, 대용량 데이터 등)
└── requirements.txt    # 의존성 패키지 목록 (pip install -r)