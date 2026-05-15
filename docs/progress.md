# 프로젝트 진행 현황 (2026-05-15 기준)

## 1. 완료된 핵심 작업

### RAG·인덱싱
- 공지: 크롤 → 통합 텍스트 → GPT 정제 → 청크 → `hoseo_notices`
- 학칙: PDF → 청크 → `hoseo_rules_v1`
- BGE-M3 hybrid + Reranker + Milvus 2.4

### TV-RAG (운영 API)
- `backend/api_service.py` — `POST /ask` (`domain`: notice | rules)
- Gemma LoRA 라우터 (`AgenticRAG/nodes/router.py` → `gemma_router_lora_v4`)
- `text_rag` / `vision_rag` — **domain별** 공지·학칙 분기
- SAIFEX 게이트웨이 고정, 부팅 시 GPU warm-up

### 크롤러
- `crawler/hoseo_spider.py` — 카테고리·상세 URL
- `crawler/incremental_notice_update.py` — 증분 + 웹훅
- `crawler/crawl_all.py` — 전체 백필, 세션 복구·모델 재사용
- Milvus 대량 `parent_id` 조회 → `query_iterator`

### 평가·실험
- `evaluation/scripts/` — QA, RAGAS, 벤치마크
- `experience/exp1/` — 라우터 SFT step1~10

### 문서
- `docs/*`, `PROJECT_MAP.md`, `README.md` (2026-05 갱신)

## 2. 연동 (Spring / Flutter)

- Spring: SSE `/api/chat/ask` → AI JSON `/ask`
- `category` → `domain` 매핑 (`학칙`→`rules`, 그 외→`notice`)
- 이슈: SSE 중 클라이언트 연결 끊김 → AI 통과 후 프론트·ngrok 점검

## 3. 진행 중 / 예정

- [ ] Spring–AI 연동 안정화 (타임아웃·SSE)
- [ ] `crawl_all` 전체 공지 백필 완료
- [ ] Agentic `main_agent.py` vs `api_service` 단일 진입점 정리
- [ ] 평가 CI smoke benchmark

## 4. 리스크

- `data/` 미추적 → 재현성·샘플 버전 관리
- API 키: SAIFEX / OpenAI / Ahoseo 혼재 — 스크립트별 확인
- GPU OOM: 배치·동시 모델 로드 주의
- ngrok 무료: 장시간 SSE 불안정 가능

## 5. 빠른 실행

```powershell
docker compose up -d
conda activate capstone_final
python -m backend.api_service
# 별도: ngrok http 8000
```

자세한 내용: [infra_setup.md](infra_setup.md)
