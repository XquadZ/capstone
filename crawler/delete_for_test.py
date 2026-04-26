from pymilvus import Collection, connections

# 1. Milvus 연결
connections.connect("default", host="localhost", port="19530")
col = Collection("hoseo_notices")
col.load()

# 2. 삭제할 타겟 ID 설정 (최신 공지 2개)
target_ids = ["96841", "96840"]

# 3. 삭제 실행 (parent_id 필드 기준)
# 문자열인지 숫자인지 확인 필요하나, 보통 string으로 처리되므로 리스트로 전달
expr = f"parent_id in {target_ids}"
res = col.delete(expr)

print(f"🗑️ 테스트 삭제 완료: {target_ids}")
print(f"결과: {res}")

# 4. 제대로 지워졌는지 카운트 확인
col.flush()
print(f"📊 현재 남은 총 엔티티 수: {col.num_entities}")