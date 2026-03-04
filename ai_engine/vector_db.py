import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# 1. 모델 및 DB 연결 설정
# [수정] 허깅페이스에서 정상 작동하는 한국어 모델로 교체
print("⏳ SBERT 모델 로딩 중 (최초 실행 시 시간이 걸릴 수 있습니다)...")
model = SentenceTransformer('jhgan/ko-sbert-sts') 

# Docker로 실행 중인 Elasticsearch 연결
es = Elasticsearch("http://localhost:9200")

def create_index(index_name="hoseo_notices"):
    """Elasticsearch 인덱스 및 매핑 설정 (벡터 검색용)"""
    # 인덱스가 이미 있으면 삭제하고 새로 만들거나, 그냥 유지 (여기선 유지)
    if not es.indices.exists(index=index_name):
        settings = {
            "mappings": {
                "properties": {
                    "notice_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "standard"}, 
                    "content": {"type": "text"},
                    "ai_summary": {"type": "text"},
                    "notice_vector": {"type": "dense_vector", "dims": 768},
                    "url": {"type": "keyword"},
                    "target_date": {"type": "date"}
                }
            }
        }
        es.indices.create(index=index_name, body=settings)
        print(f"📂 인덱스 '{index_name}' 생성 완료")
    else:
        print(f"ℹ️ 인덱스 '{index_name}'가 이미 존재합니다.")

def index_all_processed_data(index_name="hoseo_notices"):
    processed_root = os.path.join(os.getcwd(), "data", "processed")
    
    if not os.path.exists(processed_root):
        print("❌ 'data/processed' 폴더를 찾을 수 없습니다.")
        return

    actions = []
    # processed 폴더 내의 모든 공지 ID 폴더 순회
    target_folders = [f for f in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, f))]
    
    for notice_id in target_folders:
        file_path = os.path.join(processed_root, notice_id, "ai_extracted_info.json")
        if not os.path.exists(file_path): continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # [제목 보정] 제목이 "호서광장" 등 의미 없는 경우 AI 요약본 첫 줄 사용
        title = data.get('title', '제목 없음')
        if title in ["제목 없음", "호서광장", "공지사항"] and data.get('ai_summary_content'):
            title = data['ai_summary_content'].split('\n')[0][:50]

        # [데이터 통합] 검색을 위해 본문 + 이미지 요약 + PDF 요약 합치기
        full_text = data.get("ai_summary_content", "")
        for img in data.get("ai_image_summaries", []):
            full_text += "\n" + img['summary']
        for pdf in data.get("ai_attachment_summaries", []):
            full_text += "\n" + pdf['summary']
        
        # 텍스트가 너무 짧으면 임베딩 건너뜀
        if len(full_text.strip()) < 5: continue

        # 벡터 변환
        embedding = model.encode(full_text).tolist()

        doc = {
            "_index": index_name,
            "_id": notice_id,
            "_source": {
                "notice_id": notice_id,
                "title": title,
                "content": data.get('content', ''),
                "ai_summary": full_text,
                "notice_vector": embedding,
                "url": f"https://www.hoseo.ac.kr/Home/BBSView.mbz?action=MAPP_1708240139&schIdx={notice_id}",
                "target_date": data.get('target_date', '2026-03-04')
            }
        }
        actions.append(doc)
        print(f"📦 준비 완료: {title[:30]}... (ID: {notice_id})")

    # Bulk API 실행
    if actions:
        helpers.bulk(es, actions)
        print(f"🚀 총 {len(actions)}개의 데이터를 Elasticsearch에 성공적으로 저장했습니다!")
    else:
        print("ℹ️ 인덱싱할 데이터가 없습니다.")

if __name__ == "__main__":
    create_index()
    index_all_processed_data()