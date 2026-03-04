import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. 초기 설정 (환경변수에서 API 키 로드)
print("⏳ RAG 시스템 엔진 로딩 중...")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ 환경변수 'OPENAI_API_KEY'가 설정되지 않았습니다. 터미널에서 확인해 주세요.")

model = SentenceTransformer('jhgan/ko-sbert-sts')
es = Elasticsearch("http://localhost:9200")
client = OpenAI(api_key=api_key)

def get_answer(user_query):
    # STEP 1: 질문 벡터화 및 검색 (상위 2개)
    query_vector = model.encode(user_query).tolist()
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'notice_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    
    response = es.search(index="hoseo_notices", query=script_query, size=2)
    hits = response['hits']['hits']
    
    if not hits:
        return "죄송합니다. 관련 공지사항을 찾을 수 없습니다."

    # STEP 2: 검색된 내용을 컨텍스트로 변환
    context = ""
    for hit in hits:
        source = hit['_source']
        context += f"\n[공지 제목: {source['title']}]\n내용 요약: {source['ai_summary']}\n"

    # STEP 3: OpenAI API 호출
    print("🧠 OpenAI가 답변을 생성하고 있습니다...")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 호서대학교 공지사항 전문 안내원입니다. 제공된 공지 내용을 바탕으로 친절하게 답변하세요."},
            {"role": "user", "content": f"[공지 내용]:\n{context}\n\n질문: {user_query}"}
        ]
    )
    
    return completion.choices[0].message.content

if __name__ == "__main__":
    query = "우리학교 ROTC 모집해? 지원 자격이랑 기간 좀 알려줘"
    answer = get_answer(query)
    
    print(f"\n🙋 질문: {query}")
    print(f"\n🤖 AI 답변:\n{answer}")