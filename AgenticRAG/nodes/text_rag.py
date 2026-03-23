import os
from openai import OpenAI
from AgenticRAG.graph.state import AgentState

# 1. 하이브리드 검색 모듈 임포트 (Milvus 연동)
try:
    from ai_engine.rag_pipeline_rules import retrieve_documents 
except ImportError:
    print("⚠️ [Text RAG] ai_engine에서 검색 함수를 불러오지 못했습니다.")
    def retrieve_documents(q, *args, **kwargs): return [] 

# 2. SAIFEX 전용 OpenAI 클라이언트 (공식 문서 기반 세팅) [cite: 7, 20]
client = OpenAI(
    api_key=os.getenv("SAIFEX_API_KEY"),
    base_url="https://ahoseo.saifex.ai/v1" # [cite: 7, 19]
)

def text_rag_node(state: AgentState) -> dict:
    question = state["question"]
    print(f"\n--- [NODE: Text RAG] TV-RAG 텍스트 정밀 분석 시작 (Model: gpt-4o-mini) ---")

    # ==========================================
    # STEP 1: 안전한 하이브리드 검색 호출 (Top-3)
    # ==========================================
    search_results = []
    try:
        # 다양한 파라미터 규격에 대응하는 방어적 호출
        search_results = retrieve_documents(question, 3) 
    except TypeError:
        try:
            search_results = retrieve_documents(question, k=3)
        except TypeError:
            search_results = retrieve_documents(question)

    if not search_results:
        print("❌ [Text RAG] 검색 결과가 없습니다.")
        return {"generation": "관련 텍스트 문서를 찾지 못했습니다.", "context": []}

    # ==========================================
    # STEP 2: 검색 결과에서 텍스트 및 출처 메타데이터 추출
    # ==========================================
    context_text = ""
    sources_used = []
    
    for i, doc in enumerate(search_results):
        # LangChain Document 또는 Dict 호환 처리
        metadata = getattr(doc, 'metadata', doc) if hasattr(doc, 'metadata') else doc
        source = metadata.get("source", "unknown")
        sources_used.append(source)
        
        # 실제 텍스트 내용 가져오기 (page_content 혹은 text)
        content = getattr(doc, 'page_content', metadata.get('text', str(doc)))
        
        # 💡 각 컨텍스트 조각에 출처 태그를 명시적으로 삽입하여 LLM이 인지하게 함
        context_text += f"\n### 문서 조각 {i+1} [출처: {source}] ###\n{content}\n"

    print(f"📄 [Text RAG] 총 {len(set(sources_used))}개 파일에서 {len(search_results)}개의 텍스트 청크를 확보했습니다.")

    # ==========================================
    # STEP 3: GPT-4o-mini API 호출 (출처 명시 프롬프트 강화) [cite: 23, 110]
    # ==========================================
    print(f"🚀 [Text RAG] gpt-4o-mini API 호출 중...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "당신은 호서대학교 학사/행정 전문가입니다. 제공된 문서 내용만을 바탕으로 답변하세요.\n\n"
                        "### 답변 원칙 ###\n"
                        "1. 답변 내용의 근거가 되는 구체적인 조항이나 수치를 반드시 포함하세요.\n"
                        "2. 답변의 마지막 부분에 반드시 '📚 [참고 문헌 및 근거 자료]' 섹션을 구성하세요.\n"
                        "3. 해당 섹션에는 답변에 사용된 모든 파일명을 중복 없이 리스트 형태로 나열하세요.\n"
                        "4. 문서에 없는 내용은 추측하지 말고 '제공된 문서에서 확인이 어렵습니다'라고 답하세요."
                    )
                },
                {"role": "user", "content": f"사용자 질문: {question}\n\n[제공된 문서 컨텍스트]\n{context_text}"}
            ],
            max_tokens=1500,
            temperature=0.0 # 일관된 정보 제공을 위해 창의성 배제
        )
        
        generation = response.choices[0].message.content # [cite: 67, 260]
        
        # 💡 시스템 로그 차원에서도 출처를 하단에 강제 추가 (이중 안전장치)
        if "📚" not in generation:
            source_footer = "\n\n📚 **[참고 문헌 및 근거 자료]**\n"
            source_footer += "\n".join([f"- {s}" for s in set(sources_used)])
            generation += source_footer
            
        print("✅ [Text RAG] 텍스트 기반 정밀 답변 생성 완료!")
        
    except Exception as e:
        print(f"❌ [Text RAG] API 호출 실패: {e}")
        generation = "AI 분석 서버와의 통신 중 오류가 발생했습니다."

    return {
        "generation": generation, 
        "context": [f"Text Sources: {', '.join(set(sources_used))}"]
    }