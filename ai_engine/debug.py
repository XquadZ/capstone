from pymilvus import connections, Collection, utility

# 1. Milvus 연결
try:
    connections.connect("default", host="localhost", port="19530")
    print("🔌 Milvus 서버 연결 성공!")

    # 2. 확인할 컬렉션 리스트
    target_collections = ["hoseo_rules_v1", "hoseo_notices"]

    for c_name in target_collections:
        if not utility.has_collection(c_name):
            print(f"\n❌ '{c_name}' 컬렉션이 존재하지 않습니다.")
            continue
            
        col = Collection(c_name)
        print(f"\n📊 [{c_name}] 상세 스키마 정보")
        print("-" * 50)
        
        # 필드별 상세 정보 출력
        for field in col.schema.fields:
            pk_label = "[PK]" if field.is_primary else ""
            print(f"🔹 필드명: {field.name:18} | 타입: {str(field.dtype):10} {pk_label}")
            
        print(f"📝 설명: {col.description}")
        print(f"🔢 현재 적재된 엔티티 수: {col.num_entities}")
        print("-" * 50)

except Exception as e:
    print(f"❌ 오류 발생: {e}")