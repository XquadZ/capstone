import os
from openai import OpenAI
from AgenticRAG.graph.state import AgentState

try:
    from ai_engine.rag_pipeline_notice import get_shared_notice_pipeline
except ImportError:
    print("⚠️ [Text RAG] ai_engine.rag_pipeline_notice 를 불러오지 못했습니다.")
    get_shared_notice_pipeline = None  # type: ignore


def _entity_to_dict(ent):
    if isinstance(ent, dict):
        return ent
    if hasattr(ent, "to_dict"):
        return ent.to_dict()
    return dict(ent)


def _notice_hits_as_docs(hits):
    from ai_engine.notice_source_resolver import notice_context_header

    docs = []
    for h in hits:
        ent = _entity_to_dict(h["entity"])
        pid = str(ent.get("parent_id", "unknown")).strip()
        txt = ent.get("chunk_text", "") or ""
        label = notice_context_header(pid) if pid and pid != "unknown" else "공지"
        docs.append(
            {
                "parent_id": pid,
                "source": label,
                "text": txt,
                "page_content": txt,
            }
        )
    return docs


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _text_rag_rules(question: str) -> dict:
    """학칙(hoseo_rules_v1): rag_pipeline_rules 검색 + 텍스트 생성."""
    try:
        from ai_engine.rag_pipeline_rules import retrieve_documents
    except ImportError as e:
        print(f"⚠️ [Text RAG] rag_pipeline_rules 로드 실패: {e}")
        return {
            "generation": "학칙 검색 모듈을 불러오지 못했습니다.",
            "context": [],
            "retrieved_chunk_texts": [],
        }

    search_results = []
    try:
        chunks = retrieve_documents(question, top_k_milvus=10, final_top_k=5)
        for c in chunks:
            search_results.append(
                {
                    "source": str(c.get("source", "unknown")),
                    "text": c.get("text", "") or "",
                    "page_content": c.get("text", "") or "",
                    "page_num": c.get("page_num"),
                }
            )
    except Exception as e:
        print(f"❌ [Text RAG] 학칙 검색 실패: {e}")
        search_results = []

    if not search_results:
        print("❌ [Text RAG] 학칙 검색 결과가 없습니다.")
        return {
            "generation": "관련 학칙·규정 문서를 찾지 못했습니다.",
            "context": [],
            "retrieved_chunk_texts": [],
        }

    retrieved_chunk_texts = [
        (d.get("page_content") or d.get("text") or "").strip()
        for d in search_results
        if (d.get("page_content") or d.get("text") or "").strip()
    ]

    context_text = ""
    sources_used = []
    for i, doc in enumerate(search_results):
        src = doc.get("source", "unknown")
        pn = doc.get("page_num", "")
        sources_used.append(f"{src} (p.{pn})" if pn != "" else src)
        content = doc.get("page_content", doc.get("text", ""))
        context_text += f"\n### 참고 문서 {i+1} [출처: {src}, 페이지: {pn}] ###\n{content}\n"

    print(f"📄 [Text RAG·Rules] {len(search_results)}개의 학칙 청크를 확보했습니다.")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 호서대학교 학칙 및 규정을 안내하는 전문 AI 어시스턴트입니다.\n"
                        "아래 [참고 문서]만을 근거로 답변하세요.\n\n"
                        "### 답변 규칙 ###\n"
                        "1. 문서에 없는 내용은 추측하지 마세요.\n"
                        "2. 근거가 된 문서명·페이지를 명시하세요.\n"
                        "3. 답변 끝에 출처 목록을 붙이지 마세요(앱에서 규정명·페이지를 링크로 표시합니다).\n"
                        "4. 사용자 질문에 명시적인 연도/날짜 표현이 없으면 2026년 기준으로 해석해 답변하세요.\n"
                        "5. 답변에 마크다운 강조(**)를 사용하지 마세요."
                    ),
                },
                {"role": "user", "content": f"사용자 질문: {question}\n\n[참고 문서]\n{context_text}"},
            ],
            max_tokens=1500,
            temperature=0.0,
        )
        generation = response.choices[0].message.content
        print("✅ [Text RAG·Rules] 텍스트 기반 답변 생성 완료!")
    except Exception as e:
        print(f"❌ [Text RAG·Rules] API 호출 실패: {e}")
        generation = "AI 분석 서버와의 통신 중 오류가 발생했습니다."

    rules_sources = []
    seen_rules: set[str] = set()
    for doc in search_results:
        src = str(doc.get("source", "")).strip()
        if not src or src in seen_rules:
            continue
        seen_rules.add(src)
        pn = doc.get("page_num")
        rules_sources.append(
            {
                "doc_id": "",
                "title": src,
                "file_url": "",
                "category": "",
                "page": int(pn) if pn is not None and str(pn).isdigit() else None,
                "source_type": "rules",
            }
        )

    return {
        "generation": generation,
        "context": [],
        "sources_structured": rules_sources,
        "retrieved_chunk_texts": retrieved_chunk_texts,
    }


def text_rag_node(state: AgentState) -> dict:
    question = state["question"]
    domain = str(state.get("domain", "notice") or "notice").strip().lower()
    if domain == "rules":
        print(f"\n--- [NODE: Text RAG] 학칙·규정 텍스트 RAG (gpt-4o-mini) domain=rules ---")
        return _text_rag_rules(question)

    print(f"\n--- [NODE: Text RAG] 공지사항 텍스트 RAG (Model: gpt-4o-mini, OpenAI) ---")

    if get_shared_notice_pipeline is None:
        return {
            "generation": "공지사항 검색 모듈을 불러오지 못했습니다.",
            "context": [],
            "retrieved_chunk_texts": [],
        }

    search_results = []
    try:
        pipe = get_shared_notice_pipeline()
        hits = pipe.search_and_rerank(question, retrieve_k=50, final_k=5)
        search_results = _notice_hits_as_docs(hits)
    except Exception as e:
        print(f"❌ [Text RAG] 공지 검색 실패: {e}")
        search_results = []

    if not search_results:
        print("❌ [Text RAG] 검색 결과가 없습니다.")
        return {
            "generation": "관련 공지사항 문서를 찾지 못했습니다.",
            "context": [],
            "retrieved_chunk_texts": [],
        }

    retrieved_chunk_texts = [
        (d.get("page_content") or d.get("text") or "").strip()
        for d in search_results
        if (d.get("page_content") or d.get("text") or "").strip()
    ]

    from ai_engine.notice_source_resolver import (
        append_notice_links_to_answer,
        notice_sources_from_parent_ids,
    )

    context_text = ""
    parent_ids: list[str] = []

    for i, doc in enumerate(search_results):
        source = doc.get("source", "unknown")
        pid = str(doc.get("parent_id", "")).strip()
        if pid and pid != "unknown":
            parent_ids.append(pid)
        content = doc.get("page_content", doc.get("text", ""))
        context_text += f"\n### 문서 조각 {i+1}\n{source}\n---\n{content}\n"

    sources_structured = notice_sources_from_parent_ids(parent_ids)

    print(f"📄 [Text RAG] {len(search_results)}개의 공지 청크를 확보했습니다.")

    print("🚀 [Text RAG] gpt-4o-mini (OpenAI) 호출 중...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 호서대학교 행정·학사 공지 안내 전문가입니다. 제공된 공지사항 발췌만을 바탕으로 답변하세요.\n\n"
                        "### 답변 원칙 ###\n"
                        "1. 근거가 되는 연도·부서·분류 등 메타 정보와 본문 요지를 명확히 드러내세요.\n"
                        "2. 답변 본문에 공지번호(schIdx)·parent_id만 단독으로 나열하지 마세요.\n"
                        "3. 근거 공지를 언급할 때는 제목과 함께 컨텍스트에 있는 원문 URL을 답변에 반드시 적으세요.\n"
                        "4. 문서에 없는 내용은 추측하지 말고 '제공된 공지에서 확인이 어렵습니다'라고 답하세요.\n"
                        "5. 사용자 질문에 명시적인 연도/날짜 표현이 없으면 2026년 기준으로 해석해 답변하세요.\n"
                        "6. 답변에 마크다운 강조(**)를 사용하지 마세요."
                    ),
                },
                {
                    "role": "user",
                    "content": f"사용자 질문: {question}\n\n[제공된 공지사항 컨텍스트]\n{context_text}",
                },
            ],
            max_tokens=1500,
            temperature=0.0,
        )

        generation = append_notice_links_to_answer(
            response.choices[0].message.content or "",
            sources_structured,
        )
        print("✅ [Text RAG] 텍스트 기반 답변 생성 완료!")

    except Exception as e:
        print(f"❌ [Text RAG] API 호출 실패: {e}")
        generation = "AI 분석 서버와의 통신 중 오류가 발생했습니다."

    return {
        "generation": generation,
        "context": [],
        "sources_structured": sources_structured,
        "retrieved_chunk_texts": retrieved_chunk_texts,
    }
