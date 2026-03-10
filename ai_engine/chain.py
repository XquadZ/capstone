import os
import gzip
import json

# 1. 경로 설정
INDEX_PATH = ".byaldi/hoseo_vision_index"
# 추적하고 싶은 검색 결과 ID들
target_ids = ["319", "1225", "843"] 

def track_vision_ids(ids):
    gz_path = os.path.join(INDEX_PATH, "doc_ids_to_file_names.json.gz")
    
    if not os.path.exists(gz_path):
        print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {gz_path}")
        return

    print(f"📂 인덱스 맵 분석 중 (Gzip 압축 해제)...")
    
    try:
        with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
            metadata = json.load(f)
            
        print("\n🎯 [문서 추적 결과]")
        print("=" * 70)
        
        for tid in ids:
            # Byaldi는 내부적으로 정보를 문자열 키로 관리할 수 있습니다.
            info = metadata.get(str(tid))
            
            if info:
                # Byaldi 데이터 구조에 따른 분기 처리
                if isinstance(info, list):
                    file_path = info[0]
                    page_num = info[1]
                elif isinstance(info, dict):
                    file_path = info.get('path', 'unknown')
                    page_num = info.get('page_num', 1)
                else:
                    file_path = info
                    page_num = "단일 이미지"

                print(f"🆔 문서 ID : {tid}")
                print(f"📄 파일명  : {os.path.basename(file_path)}")
                print(f"📍 페이지  : {page_num}")
                print(f"📁 절대경로: {os.path.abspath(file_path)}")
                print("-" * 70)
            else:
                print(f"⚠️ ID {tid}번에 대한 정보를 찾을 수 없습니다.")

    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    track_vision_ids(target_ids)