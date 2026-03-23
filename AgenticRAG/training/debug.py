from pymilvus import connections, utility

# 1. Milvus 서버 연결 (Docker 19530 포트)
try:
    connections.connect("default", host="localhost", port="19530")
    print("🔌 Milvus 서버 연결 성공!")

    # 2. 현재 생성된 모든 컬렉션 리스트 가져오기
    collections = utility.list_collections()

    print("\n" + "="*40)
    print(f"📦 현재 DB에 있는 컬렉션 목록 (총 {len(collections)}개)")
    print("="*40)
    
    if not collections:
        print("⚠️ 생성된 컬렉션이 하나도 없습니다.")
    else:
        for i, name in enumerate(collections):
            print(f"  ({i+1}) {name}")
    print("="*40)

except Exception as e:
    print(f"❌ 연결 실패: {e}")