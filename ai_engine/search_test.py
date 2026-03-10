from byaldi import RAGMultiModalModel

print("⏳ Byaldi 비전 RAG 인덱스 로딩 중... (최초 로딩 시 시간이 조금 걸립니다)")
# 우리가 방금 전까지 고생해서 만든 그 인덱스를 불러옵니다.
RAG = RAGMultiModalModel.from_index("hoseo_vision_index")

# 올려주신 첫 번째 이미지 속 질문을 그대로 타이핑했습니다!
query = "2026년 호서대 학생예비군 훈련 일정좀 알려줘"

print(f"\n🔍 질문: '{query}'")
print("🧠 수천 장의 이미지 중에서 정답이 있는 이미지를 찾고 있습니다...")

# k=3 : 가장 관련성 높은 이미지 3장을 가져옵니다.
results = RAG.search(query, k=3)

print("\n🎉 [검색 결과]")
print("=" * 50)
for i, res in enumerate(results):
    print(f"🏆 [{i+1}위] 문서 ID (doc_id): {res.doc_id}")
    # metadata에 파일명이 저장되어 있다면 출력, 없으면 기본값
    # 문서가 이미지인지 PDF의 특정 페이지인지 확인 가능
    print(f"⭐ 매칭 점수 (Score): {res.score:.4f}")
    print("-" * 50)