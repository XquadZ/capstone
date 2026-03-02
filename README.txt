# 🎓 Hoseo RAG Project: 공지사항 질의응답 시스템

이 프로젝트는 호서대학교 공지사항 데이터를 수집하여, 사용자의 질문에 정확한 정보를 제공하는 RAG(Retrieval-Augmented Generation) 기반 AI 에이전트를 구축하는 프로젝트입니다.

## 🚀 연구실 서버 환경 (Lab Setup)
- **GPU:** NVIDIA GeForce RTX 4090
- **Python:** 3.12 (Conda 환경명: `capstone_4090`)
- **Key Stack:** PyTorch (CUDA 12.1), LangChain, BeautifulSoup4, ChromaDB

## 📂 프로젝트 구조 (Project Structure)
```text
CAPSTONE/
├── ai_engine/          # 핵심 AI 로직 (Chain, VectorDB 관리)
├── crawler/            # 데이터 수집 모듈 (호서대 공지사항 크롤러)
│   └── README.md       # 크롤러 상세 명세서
├── data/               # [Ignored] 수집된 데이터 (JSON, PDF)
├── docs/               # 시스템 설계 및 API 문서
├── requirements.txt    # 의존성 패키지 목록
└── .gitignore          # Git 제외 목록 (가상환경, 데이터 등)