import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="사용자 질문")
    domain: str = Field("notice", description="notice | rules")
    use_tv_rag: bool = Field(
        True, description="notice 도메인에서만 TV-RAG(라우터) 사용"
    )


class AskResponse(BaseModel):
    domain: str
    question: str
    answer: str
    route: str
    latency_sec: float
    contexts: List[str]
    sources: List[str]
    meta: Dict[str, Any]


app = FastAPI(
    title="Capstone TV-RAG API",
    version="0.1.0",
    description="Spring/ngrok 연동 테스트용 최소 API",
)

_notice_pipeline = None


def _get_notice_pipeline():
    global _notice_pipeline
    if _notice_pipeline is None:
        from ai_engine.rag_pipeline_notice import HoseoRAGPipeline

        _notice_pipeline = HoseoRAGPipeline()
    return _notice_pipeline


def _safe_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


@app.get("/health")
def health():
    return {"status": "ok", "service": "tv-rag-api", "ts": time.time()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    domain = req.domain.strip().lower()
    q = req.question.strip()
    if domain not in {"notice", "rules"}:
        raise HTTPException(status_code=400, detail="domain은 notice 또는 rules만 허용됩니다.")

    started = time.time()

    if domain == "rules":
        # 학칙은 현재 rules 전용 텍스트 RAG 경로를 직접 사용
        from ai_engine.rag_pipeline_rules import generate_answer, retrieve_documents

        chunks = retrieve_documents(q, top_k_milvus=10, final_top_k=3)
        answer = generate_answer(q, chunks)
        contexts = [str(c.get("text", "")).strip() for c in chunks if c.get("text")]
        sources = [str(c.get("source", "")).strip() for c in chunks if c.get("source")]

        return AskResponse(
            domain=domain,
            question=q,
            answer=answer,
            route="TEXT",
            latency_sec=round(time.time() - started, 4),
            contexts=contexts,
            sources=list(dict.fromkeys(sources)),
            meta={"pipeline": "rag_pipeline_rules", "chunk_count": len(chunks)},
        )

    # ---------------- notice domain ----------------
    if req.use_tv_rag:
        # notice + TV-RAG: 라우터 판단 후 text/vision 노드 실행
        from AgenticRAG.nodes.router import slm_router_node
        from AgenticRAG.nodes.text_rag import text_rag_node
        from AgenticRAG.nodes.vision_rag import vision_rag_node

        state = {"question": q, "retry_count": 0}
        route_decision = str(slm_router_node(state).get("route_decision", "TEXT")).upper()
        route = "VISION" if "VISION" in route_decision else "TEXT"

        node_result = vision_rag_node(state) if route == "VISION" else text_rag_node(state)
        answer = str(node_result.get("generation", "")).strip()
        contexts = _safe_list_str(node_result.get("retrieved_chunk_texts", []))
        if not contexts:
            contexts = _safe_list_str(node_result.get("context", []))
        sources = _safe_list_str(node_result.get("context", []))

        return AskResponse(
            domain=domain,
            question=q,
            answer=answer,
            route=route,
            latency_sec=round(time.time() - started, 4),
            contexts=contexts,
            sources=sources,
            meta={"pipeline": "agentic_notice_tv_rag", "use_tv_rag": True},
        )

    # notice + 단일 텍스트 RAG (라우터 미사용)
    pipeline = _get_notice_pipeline()
    hits = pipeline.search_and_rerank(q, retrieve_k=50, final_k=5)
    answer = pipeline.generate_answer(q)
    contexts = []
    sources = []
    for h in hits:
        ent = h.get("entity", {})
        txt = str(ent.get("chunk_text", "")).strip()
        if txt:
            contexts.append(txt)
        pid = str(ent.get("parent_id", "")).strip()
        cat = str(ent.get("category", "")).strip() or "일반"
        if pid:
            sources.append(f"[공지-{cat}] {pid}")

    return AskResponse(
        domain=domain,
        question=q,
        answer=answer,
        route="TEXT",
        latency_sec=round(time.time() - started, 4),
        contexts=contexts,
        sources=list(dict.fromkeys(sources)),
        meta={"pipeline": "rag_pipeline_notice", "use_tv_rag": False, "chunk_count": len(hits)},
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("backend.api_service:app", host=host, port=port, reload=True)
