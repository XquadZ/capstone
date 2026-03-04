import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# 1. 모델 로딩
print("⏳ SBERT 모델 로딩 중...")
model = SentenceTransformer('jhgan/ko-sbert-sts') 

# 2. [수정] Elasticsearch 연결 설정 보강
# vector_db.py 상단 연결 부분 수정
es = Elasticsearch(
    "http://localhost:9200",
    verify_certs=False,
    request_timeout=30,
    # [추가] 서버가 이해할 수 있는 v8 헤더를 강제로 보냄
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8",
             "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"}
)

def create_index(index_name="hoseo_notices"):
    """Elasticsearch 인덱스 생성"""
    try:
        # 인덱스 존재 여부 확인 전 예외 처리 추가
        if es.indices.exists(index=index_name):
            print(f"ℹ️ 인덱스 '{index_name}'가 이미 존재합니다.")
            return

        mappings = {
            "properties": {
                "notice_id": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "ai_summary": {"type": "text"},
                "notice_vector": {
                    "type": "dense_vector", 
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                },
                "url": {"type": "keyword"},
                "target_date": {"type": "date"}
            }
        }
        
        es.indices.create(index=index_name, mappings=mappings)
        print(f"📂 인덱스 '{index_name}' 생성 완료")
    except Exception as e:
        print(f"⚠️ 인덱스 체크/생성 중 오류(무시 가능): {e}")

def index_all_processed_data(index_name="hoseo_notices"):
    processed_root = os.path.join(os.getcwd(), "data", "processed")
    if not os.path.exists(processed_root):
        print("❌ 'data/processed' 폴더가 없습니다.")
        return

    actions = []
    target_folders = [f for f in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, f))]
    
    for notice_id in target_folders:
        file_path = os.path.join(processed_root, notice_id, "ai_extracted_info.json")
        if not os.path.exists(file_path): continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = data.get('title', '제목 없음')
        if title in ["제목 없음", "호서광장", "공지사항"] and data.get('ai_summary_content'):
            title = data['ai_summary_content'].split('\n')[0][:50]

        full_text = data.get("ai_summary_content", "")
        for img in data.get("ai_image_summaries", []):
            full_text += "\n" + img['summary']
        for pdf in data.get("ai_attachment_summaries", []):
            full_text += "\n" + pdf['summary']
        
        if len(full_text.strip()) < 5: continue
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

    if actions:
        helpers.bulk(es, actions)
        print(f"🚀 총 {len(actions)}개의 데이터를 Elasticsearch에 저장했습니다!")

if __name__ == "__main__":
    create_index()
    index_all_processed_data()