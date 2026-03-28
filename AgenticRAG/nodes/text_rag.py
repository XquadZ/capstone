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
    docs = []
    for h in hits:
        ent = _entity_to_dict(h["entity"])
        pid = ent.get("parent_id", "unknown")
        cat = ent.get("category", "") or "일반"
        txt = ent.get("chunk_text", "") or ""
        label = f"[공지-{cat}] {pid}"
        docs.append({"source": label, "text": txt, "page_content": txt})
    return docs


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def text_rag_node(state: AgentState) -> dict:
    question = state["question"]
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
        hits = pipe.search_and_rerank(question, retrieve_k=50, final_k=3)
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

    context_text = ""
    sources_used = []

    for i, doc in enumerate(search_results):
        metadata = doc
        source = metadata.get("source", "unknown")
        sources_used.append(source)
        content = metadata.get("page_content", metadata.get("text", ""))
        context_text += f"\n### 문서 조각 {i+1} [출처: {source}] ###\n{content}\n"

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
                        "2. 답변 마지막에 반드시 '📚 [참고 문헌 및 근거 자료]' 섹션을 두고, 사용한 출처 라벨을 중복 없이 나열하세요.\n"
                        "3. 문서에 없는 내용은 추측하지 말고 '제공된 공지에서 확인이 어렵습니다'라고 답하세요."
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

        generation = response.choices[0].message.content

        if "📚" not in generation:
            source_footer = "\n\n📚 **[참고 문헌 및 근거 자료]**\n"
            source_footer += "\n".join([f"- {s}" for s in set(sources_used)])
            generation += source_footer

        print("✅ [Text RAG] 텍스트 기반 답변 생성 완료!")

    except Exception as e:
        print(f"❌ [Text RAG] API 호출 실패: {e}")
        generation = "AI 분석 서버와의 통신 중 오류가 발생했습니다."

    return {
        "generation": generation,
        "context": [f"Text Sources (Notice): {', '.join(set(sources_used))}"],
        "retrieved_chunk_texts": retrieved_chunk_texts,
    }
