import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. 초기 설정
print("⏳ 하이브리드 RAG 시스템 엔진 로딩 중...")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ 'OPENAI_API_KEY'가 설정되지 않았습니다.")

model = SentenceTransformer('jhgan/ko-sbert-sts')

# 연구실 서버 연결 설정 (vector_db와 동일)
es = Elasticsearch(
    "http://localhost:9200",
    meta_header=False,
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)
client = OpenAI(api_key=api_key)

def get_answer(user_query):
    query_vector = model.encode(user_query).tolist()
    
    # [핵심] ES 8.x 하이브리드 검색 (BM25 키워드 + k-NN 벡터)
    # 의미상으로 비슷하거나(벡터), 정확히 일치하는 키워드(ROTC 등)가 있으면 최상단으로 올림
    print(f"🔍 '{user_query}' 관련 공지 검색 중...")
    
    try:
        response = es.search(
            index="hoseo_notices",
            query={
                # 1. 키워드 매칭 (제목에 있는 키워드는 3배 가중치)
                "multi_match": {
                    "query": user_query,
                    "fields": ["title^3", "ai_summary"],
                    "boost": 0.5
                }
            },
            knn={
                # 2. 벡터 유사도 매칭
                "field": "notice_vector",
                "query_vector": query_vector,
                "k": 3,
                "num_candidates": 100,
                "boost": 0.5
            },
            size=3, # 최종 상위 3개 문서 추출
            source=["title", "ai_summary", "url"]
        )
        hits = response['hits']['hits']
        
        # 디버깅 출력
        print(f"📊 최종 하이브리드 검색된 문서 개수: {len(hits)}개")
        for i, hit in enumerate(hits):
            print(f"   [{i+1}] 제목: {hit['_source']['title']} (최종 점수: {hit['_score']:.4f})")
            
    except Exception as e:
        print(f"⚠️ 검색 중 에러 발생: {e}")
        return "검색 엔진 연결에 실패했습니다."
    
    if not hits:
        return "죄송합니다. 관련 공지사항을 찾을 수 없습니다."

    # STEP 3: 컨텍스트 구성
    context = ""
    for hit in hits:
        source = hit['_source']
        context += f"\n[공지 제목: {source['title']}]\n핵심 내용: {source['ai_summary']}\n링크: {source['url']}\n"

    # STEP 4: OpenAI GPT-4o 답변 생성
    print("🧠 OpenAI가 최적의 답변을 생성하고 있습니다...")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 호서대학교 공지사항 전문 AI입니다. 제공된 [공지 내용]을 분석하여 질문에 매우 정확하고 친절하게 답변하세요. 기간, 자격 요건 등의 수치는 틀리지 않게 주의하세요."},
            {"role": "user", "content": f"[공지 내용]:\n{context}\n\n질문: {user_query}"}
        ]
    )
    
    return completion.choices[0].message.content

if __name__ == "__main__":
    query = "지금 교내근로 모집하는 거 있어?"
    answer = get_answer(query)
    
    print(f"\n🙋 질문: {query}")
    print("-" * 50)
    print(f"🤖 AI 답변:\n{answer}")