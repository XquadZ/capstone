"""
FastAPI 진입점: 공지(notice)·학칙(rules) 모두 동일한 TV-RAG 플로우
(라우터 → TEXT=text_rag_node / VISION=vision_rag_node).

도메인별 차이는 state["domain"]으로 노드 내부에서 Milvus 컬렉션·RAW 경로만 분기합니다.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# SAIFEX 고정 (서버 프로세스 내 OpenAI SDK 호출을 게이트웨이로 통일)
# ---------------------------------------------------------------------------
SAIFEX_BASE_URL = "https://ahoseo.saifex.ai/v1"
SAIFEX_API_KEY = os.getenv("SAIFEX_API_KEY")
if not SAIFEX_API_KEY:
    raise RuntimeError(
        "SAIFEX_API_KEY가 필요합니다. 현재 API 서버는 SAIFEX 전용으로 동작하도록 설정되어 있습니다."
    )

os.environ["OPENAI_API_KEY"] = SAIFEX_API_KEY
os.environ["OPENAI_BASE_URL"] = SAIFEX_BASE_URL

# 로컬 자가 테스트: /ask 시 서버 터미널에 질문·답변 미리보기 출력 (운영 배포 시 False)
ASK_LOG_TO_TERMINAL: bool = True
ASK_LOG_ANSWER_MAX_CHARS: int = 8000


def _print_ask_terminal_preview(question: str, route: str, latency_sec: float, answer: str) -> None:
    if not ASK_LOG_TO_TERMINAL:
        return
    body = (answer or "").strip()
    if len(body) > ASK_LOG_ANSWER_MAX_CHARS:
        tail = len(answer) - ASK_LOG_ANSWER_MAX_CHARS
        body = body[:ASK_LOG_ANSWER_MAX_CHARS] + f"\n... ({tail}자 생략)"
    print("\n" + "=" * 60, flush=True)
    print(f"[ASK] Q: {question}", flush=True)
    print(f"[ASK] route={route}  latency_sec={latency_sec}", flush=True)
    print("[ASK] A:", flush=True)
    print(body, flush=True)
    print("=" * 60 + "\n", flush=True)


# --- 요청/응답 스키마 ---------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="사용자 질문")
    domain: str = Field("notice", description="notice | rules")
    use_tv_rag: bool = Field(
        True,
        description="True: 라우터+TV-RAG(공통). False: 라우터 없이 텍스트 RAG만 (도메인별 단일 파이프라인)",
    )


class SourceItem(BaseModel):
    doc_id: str = ""
    title: str = ""
    file_url: str = ""
    category: str = ""
    page: Optional[int] = None
    source_type: str = "notice"


class AskResponse(BaseModel):
    domain: str
    question: str
    answer: str
    route: str
    latency_sec: float
    contexts: List[str]
    sources: List[SourceItem] = Field(default_factory=list)
    meta: Dict[str, Any]


# --- 글로벌 파이프라인 변수 ---------------------------------------------------

_notice_pipeline = None

def _get_notice_pipeline():
    global _notice_pipeline
    if _notice_pipeline is None:
        from ai_engine.rag_pipeline_notice import HoseoRAGPipeline
        _notice_pipeline = HoseoRAGPipeline()
    return _notice_pipeline


# --- 🚀 Lifespan (GPU 사전 적재 / Warm-up) -----------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("🔥 [서버 부팅] AI 모델 GPU 사전 적재(Warm-up)를 시작합니다...")
    start_time = time.time()
    
    # 1. 텍스트 RAG 파이프라인 (임베딩 & 리랭커 모델) VRAM 적재
    try:
        print("📦 [1/2] 텍스트 RAG 모델(임베딩/리랭커) 로딩 중...")
        _get_notice_pipeline()
    except Exception as e:
        print(f"❌ 텍스트 RAG 로딩 에러: {e}")

    # 2. SLM 라우터 노드 로딩 및 '더미(Dummy) 질문' 던지기
    try:
        print("🧠 [2/2] SLM 라우터 모델(LoRA) 로딩 및 엔진 예열 중...")
        from AgenticRAG.nodes.router import slm_router_node
        # 깡통 질문을 던져서 모델이 실제로 추론을 1회 돌게 만듭니다. (진짜 워밍업)
        dummy_state = {"question": "시스템 워밍업용 더미 질문입니다.", "domain": "notice", "retry_count": 0}
        slm_router_node(dummy_state)
    except Exception as e:
        print(f"❌ 라우터 워밍업 에러: {e}")
        
    print(f"✅ [부팅 완료] 모든 AI 모델 GPU 적재 끝! (소요시간: {time.time() - start_time:.2f}초)")
    print("="*60 + "\n")
    
    yield # 서버가 돌아가는 동안 대기
    
    # 서버 종료 시 리소스 정리
    print("🛑 서버가 종료됩니다. GPU 메모리를 해제합니다.")


# --- FastAPI 앱 --------------------------------------------------------------

app = FastAPI(
    title="Capstone TV-RAG API",
    version="0.2.0",
    description="notice / rules 공통 TV-RAG · Spring·ngrok 연동용",
    lifespan=lifespan  # 👈 여기에 lifespan이 연결됩니다!
)


# --- 헬퍼 함수 ---------------------------------------------------------------

def _safe_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def _normalize_domain(raw: str) -> str:
    d = raw.strip().lower()
    if d not in {"notice", "rules"}:
        raise HTTPException(status_code=400, detail="domain은 notice 또는 rules만 허용됩니다.")
    return d


def _sources_from_node(node_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = node_result.get("sources_structured")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("file_url") or item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if not title and not url:
            continue
        out.append(
            {
                "doc_id": str(item.get("doc_id") or item.get("notice_id") or "").strip(),
                "title": title,
                "file_url": url,
                "category": str(item.get("category") or "").strip(),
                "page": item.get("page"),
                "source_type": str(item.get("source_type") or "notice").strip(),
            }
        )
    return out


def _dedupe_sources(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = it.get("file_url") or f"{it.get('source_type')}:{it.get('doc_id')}:{it.get('title')}"
        if key in seen:
            continue
        seen.add(str(key))
        out.append(it)
    return out


def _tv_rag_answer_and_meta(
    domain: str, question: str
) -> Tuple[str, str, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    공통 TV-RAG: Gemma 라우터 → text_rag_node | vision_rag_node.
    노드는 state['domain']으로 hoseo_notices vs hoseo_rules_v1 및 RAW 경로를 선택합니다.
    """
    from AgenticRAG.nodes.router import slm_router_node
    from AgenticRAG.nodes.text_rag import text_rag_node
    from AgenticRAG.nodes.vision_rag import vision_rag_node

    state: Dict[str, Any] = {"question": question, "retry_count": 0, "domain": domain}
    route_decision = str(slm_router_node(state).get("route_decision", "TEXT")).upper()
    route = "VISION" if "VISION" in route_decision else "TEXT"

    node_result = vision_rag_node(state) if route == "VISION" else text_rag_node(state)
    answer = str(node_result.get("generation", "")).strip()

    contexts = _safe_list_str(node_result.get("retrieved_chunk_texts", []))
    if not contexts:
        contexts = _safe_list_str(node_result.get("context", []))

    sources = _dedupe_sources(_sources_from_node(node_result))

    meta = {
        "pipeline": "agentic_tv_rag",
        "domain": domain,
        "use_tv_rag": True,
        "provider": "saifex",
        "router_raw": route_decision,
    }
    return answer, route, contexts, sources, meta


def _notice_text_only(question: str) -> Tuple[str, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """라우터 미사용: 공지 단일 텍스트 RAG (HoseoRAGPipeline)."""
    from ai_engine.notice_source_resolver import notice_sources_from_parent_ids

    pipeline = _get_notice_pipeline()
    hits = pipeline.search_and_rerank(question, retrieve_k=50, final_k=5)
    answer = pipeline.generate_answer(question)

    contexts: List[str] = []
    parent_ids: List[str] = []
    for h in hits:
        ent = h.get("entity", {})
        txt = str(ent.get("chunk_text", "")).strip()
        if txt:
            contexts.append(txt)
        pid = str(ent.get("parent_id", "")).strip()
        if pid:
            parent_ids.append(pid)

    sources = notice_sources_from_parent_ids(parent_ids)
    from ai_engine.notice_source_resolver import append_notice_links_to_answer

    answer = append_notice_links_to_answer(answer, sources)

    meta = {
        "pipeline": "rag_pipeline_notice",
        "use_tv_rag": False,
        "provider": "saifex",
        "chunk_count": len(hits),
    }
    return answer, contexts, sources, meta


def _rules_text_only(question: str) -> Tuple[str, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """라우터 미사용: 학칙 텍스트 RAG (hoseo_rules_v1)."""
    from ai_engine.rag_pipeline_rules import generate_answer, retrieve_documents

    chunks = retrieve_documents(question, top_k_milvus=10, final_top_k=3)
    answer = generate_answer(question, chunks)
    contexts = [str(c.get("text", "")).strip() for c in chunks if c.get("text")]
    sources: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for c in chunks:
        src = str(c.get("source", "")).strip()
        if not src or src in seen:
            continue
        seen.add(src)
        pn = c.get("page_num")
        sources.append(
            {
                "doc_id": str(c.get("doc_id", "")).strip(),
                "title": src,
                "file_url": "",
                "category": "",
                "page": int(pn) if pn is not None and str(pn).isdigit() else None,
                "source_type": "rules",
            }
        )
    meta = {
        "pipeline": "rag_pipeline_rules",
        "use_tv_rag": False,
        "provider": "saifex",
        "chunk_count": len(chunks),
    }
    return answer, contexts, sources, meta


# --- API 엔드포인트 ----------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "tv-rag-api", "ts": time.time()}


@app.get("/ask-preview", response_model=AskResponse)
def ask_preview(
    q: str = Query(..., min_length=1, description="질문 (브라우저·curl 자가 테스트용)"),
    domain: str = Query("notice", description="notice | rules"),
    use_tv_rag: bool = Query(True),
):
    """POST /ask 와 동일 파이프라인. 터미널 미리보기 로그도 동일하게 출력됩니다."""
    return ask(AskRequest(question=q.strip(), domain=domain, use_tv_rag=use_tv_rag))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    domain = _normalize_domain(req.domain)
    q = req.question.strip()
    started = time.time()

    if req.use_tv_rag:
        answer, route, contexts, sources, meta = _tv_rag_answer_and_meta(domain, q)
    else:
        route = "TEXT"
        if domain == "notice":
            answer, contexts, sources, meta = _notice_text_only(q)
        else:
            answer, contexts, sources, meta = _rules_text_only(q)

    latency_sec = round(time.time() - started, 4)

    if domain == "notice" and sources:
        from ai_engine.notice_source_resolver import append_notice_links_to_answer

        answer = append_notice_links_to_answer(answer, sources)

    _print_ask_terminal_preview(q, route, latency_sec, answer)

    source_models = [SourceItem(**s) for s in sources]

    return AskResponse(
        domain=domain,
        question=q,
        answer=answer,
        route=route,
        latency_sec=latency_sec,
        contexts=contexts,
        sources=source_models,
        meta=meta,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("backend.api_service:app", host=host, port=port, reload=True)