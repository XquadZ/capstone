from pymilvus import connections, utility

# 1. Milvus 연결
connections.connect("default", host="localhost", port="19530")

# 2. 삭제할 컬렉션 이름 확인
COLLECTION_NAME = "rules_collection"

if utility.has_collection(COLLECTION_NAME):
    print(f"🗑️ '{COLLECTION_NAME}' 컬렉션을 삭제합니다. (모든 데이터 초기화)")
    utility.drop_collection(COLLECTION_NAME)
    print("✅ 삭제 완료. 이제 새로운 스키마로 다시 생성할 수 있습니다.")
else:
    print("❌ 삭제할 컬렉션이 존재하지 않습니다.")