from pymilvus import connections, Collection, utility

def check_schema():
    # 1. Milvus 연결
    try:
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus 연결 성공!")
    except Exception as e:
        print(f"❌ Milvus 연결 실패: {e}")
        return

    # 2. 존재하는 모든 컬렉션 목록 확인
    collections = utility.list_collections()
    print(f"📦 현재 존재하는 컬렉션: {collections}")

    for col_name in collections:
        print(f"\n" + "="*50)
        print(f"🔍 컬렉션명: [{col_name}] 정보")
        print("="*50)
        
        col = Collection(col_name)
        schema = col.schema
        
        print(f"📝 설명: {schema.description}")
        print("\n📌 필드 리스트:")
        print(f"{'필드명':<20} | {'타입':<15} | {'기본키':<10}")
        print("-" * 50)
        
        for field in schema.fields:
            pk_status = "YES" if field.is_primary else "NO"
            print(f"{field.name:<20} | {str(field.dtype):<15} | {pk_status:<10}")
        
        # 인덱스 정보 확인 (검색 가능 여부 확인)
        indexes = col.indexes
        if indexes:
            print("\n⚡ 인덱스 정보:")
            for idx in indexes:
                print(f"- 필드: {idx.field_name}, 인덱스 타입: {idx.index_type}")
        else:
            print("\n⚠️ 인덱스가 생성되지 않았습니다.")

if __name__ == "__main__":
    check_schema()