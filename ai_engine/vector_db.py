import os
import json
import torch
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# 1. 보안 가드레일 우회 설정
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "True"

print(f"✅ 현재 PyTorch 버전: {torch.__version__}")
print("⏳ SBERT 모델 로딩 중 (ko-sbert-sts)...")

try:
    model = SentenceTransformer('jhgan/ko-sbert-sts') 
    print("✅ 모델 로딩 완료!")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit()

# 2. [연구실 환경] Elasticsearch 연결 설정 (강제 호환 모드)
es = Elasticsearch(
    "http://localhost:9200",
    verify_certs=False,
    request_timeout=60,
    meta_header=False, 
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)

def create_index(index_name="hoseo_notices"):
    """하이브리드 검색을 위한 인덱스 재생성"""
    try:
        # 기존 인덱스가 있다면 삭제하고 깔끔하게 다시 만듭니다. (구조 변경을 위해)
        if es.indices.exists(index=index_name):
            print(f"♻️ 기존 인덱스 '{index_name}'를 삭제하고 새로 생성합니다.")
            es.indices.delete(index=index_name)

        # [핵심] 키워드 검색(title, ai_summary)과 벡터 검색(notice_vector)을 동시 지원하는 구조
        mappings = {
            "properties": {
                "notice_id": {"type": "keyword"},
                "title": {"type": "text"},          # BM25 키워드 검색용
                "content": {"type": "text"},
                "ai_summary": {"type": "text"},     # BM25 키워드 검색용
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
        print(f"📂 하이브리드 인덱스 '{index_name}' 생성 완료")
    except Exception as e:
        print(f"⚠️ 인덱스 생성 중 오류: {e}")

def index_all_processed_data(index_name="hoseo_notices"):
    processed_root = os.path.join(os.getcwd(), "data", "processed")
    if not os.path.exists(processed_root):
        print(f"❌ '{processed_root}' 폴더가 없습니다.")
        return

    actions = []
    target_folders = [f for f in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, f))]
    
    print(f"🔍 총 {len(target_folders)}개의 공지사항 재구성 중...")

    for notice_id in target_folders:
        file_path = os.path.join(processed_root, notice_id, "ai_extracted_info.json")
        if not os.path.exists(file_path): continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = data.get('title', '제목 없음')
        if title in ["제목 없음", "호서광장", "공지사항"] and data.get('ai_summary_content'):
            title = data['ai_summary_content'].split('\n')[0][:50]

        # [고도화] 임베딩용 텍스트 재구성
        # SBERT는 보통 512토큰에서 잘리므로, 가장 중요한 '제목'과 'PDF 요약'을 앞쪽으로 배치합니다.
        embed_text = f"제목: {title}\n"
        
        pdf_texts = [pdf['summary'] for pdf in data.get("ai_attachment_summaries", [])]
        if pdf_texts:
            embed_text += "[첨부파일 핵심 내용]\n" + "\n".join(pdf_texts) + "\n"
            
        embed_text += f"[본문 요약]\n{data.get('ai_summary_content', '')}\n"
        
        for img in data.get("ai_image_summaries", []):
            embed_text += f"[이미지 정보]\n{img['summary']}\n"
            
        if len(embed_text.strip()) < 5: continue
        
        # SBERT 벡터 임베딩 생성 (자동으로 Max Length까지 잘라서 처리됨)
        embedding = model.encode(embed_text).tolist()

        doc = {
            "_index": index_name,
            "_id": notice_id,
            "_source": {
                "notice_id": notice_id,
                "title": title,
                "content": data.get('content', ''),
                "ai_summary": embed_text,  # 원본 대신 재구성된 텍스트를 저장하여 검색 효율 극대화
                "notice_vector": embedding,
                "url": f"https://www.hoseo.ac.kr/Home/BBSView.mbz?action=MAPP_1708240139&schIdx={notice_id}",
                "target_date": data.get('target_date', '2026-03-04')
            }
        }
        actions.append(doc)
        print(f"📦 벡터화 완료: {title[:25]}... (ID: {notice_id})")

    if actions:
        helpers.bulk(es, actions)
        print(f"🚀 총 {len(actions)}개의 하이브리드 데이터를 ES에 저장했습니다!")

if __name__ == "__main__":
    create_index()
    index_all_processed_data()