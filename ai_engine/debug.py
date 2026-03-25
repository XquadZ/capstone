from pymilvus import connections, Collection

def check_fields(collection_name="hoseo_notices"):
    try:
        # 1. 연결
        connections.connect("default", host="localhost", port="19530")
        
        # 2. 컬렉션 로드
        col = Collection(collection_name)
        schema = col.schema

        print(f"\n🚀 [{collection_name}] 컬렉션 필드 리스트")
        print("=" * 60)
        print(f"{'필드명':<20} | {'데이터 타입':<15} | {'설명'}")
        print("-" * 60)

        # 3. 필드 정보만 순회 (데이터는 출력 안 함)
        for field in schema.fields:
            # 벡터 필드인지 체크해서 표시해주기
            field_type = str(field.dtype)
            desc = field.description if field.description else "-"
            print(f"{field.name:<20} | {field_type:<15} | {desc}")
        
        print("=" * 60)
        
        # 4. 데이터 샘플 (벡터 제외하고 텍스트 필드만 1건 확인)
        col.load()
        # 출력 필드에서 벡터(dense_vector, sparse_vector)를 제외하고 요청
        text_fields = [f.name for f in schema.fields if "vector" not in f.name.lower()]
        res = col.query(expr="", limit=1, output_fields=text_fields)
        
        if res:
            print("\n🧪 [텍스트 데이터 샘플 1건]")
            import json
            print(json.dumps(res[0], indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    check_fields()