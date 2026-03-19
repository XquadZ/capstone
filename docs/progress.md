# 🚀 캡스톤 프로젝트 진행 상황

## 📅 2026-03-03 (오늘 완료된 작업)
- [x] **호서대 크롤러(`hoseo_spider.py`) 고도화**
    - Iframe 구조 분석 및 상대 경로(`urljoin`) 문제 해결.
    - 본문 내 포스터 이미지(ThumbnailPrint) 및 첨부파일(PDF, HWP) 로컬 저장 기능 완성.
    - `data/raw/{notice_id}/` 구조로 데이터 물리적 저장 성공.
- [x] **디버깅 및 환경 설정**
    - Selenium 타임아웃 예외 처리 및 유연한 선택자 적용.
    - `SSL InsecureRequestWarning` 대응 및 세션 쿠키 활용 다운로드 로직 구현.

## 📅 2026-03-04 (내일부터 이어할 작업)
- [ ] **파일명 및 구조 정규화**
    - `ai_engine/vision_preocessor.py` 오타 수정 → `vision_processor.py`.
- [ ] **Vision AI 연동 (OCR)**
    - 로컬에 저장된 `img_*.jpg` 파일에서 텍스트 추출 로직 구현.
    - 추출된 텍스트를 기존 `info.json`에 병합하는 기능 추가.
- [ ] **문서 파싱 기능 확장**
    - `attachments/` 폴더 내 PDF 및 HWP 파일의 텍스트 추출 방안 검토 (PyMuPDF 등 활용).
- [ ] **데이터 정제(Preprocessing)**
    - 수집된 원본 데이터를 모델 학습 또는 검색에 용이한 형태로 가공.

---
## 💡 개발자 조언 (놓치기 쉬운 부분)
1. **오타 수정**: `preocessor` 같은 오타는 나중에 외부 라이브러리에서 이 파일을 참조할 때 `ImportError`를 유발하므로 가장 먼저 수정하는 것이 좋습니다.
2. **이미지 품질 체크**: 저장된 이미지들이 OCR을 돌리기에 해상도가 충분한지 육안으로 한 번 확인해 보세요.
3. **git ignore**: 만약 Git을 사용 중이라면, `data/raw/` 내부의 대용량 파일들이 원격 저장소에 올라가지 않도록 `.gitignore` 설정을 확인해야 합니다. 