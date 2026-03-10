import os
from PIL import Image
from byaldi import RAGMultiModalModel

# 🚀 [필수] 초고해상도 이미지 제한 해제
Image.MAX_IMAGE_PIXELS = None 

def run_qwen_indexing():
    input_path = "data/byaldi_input"
    index_name = "hoseo_vision_index"
    
    # 전체 파일 개수 파악
    all_files = os.listdir(input_path)
    total_count = len(all_files)

    print(f"🚀 한국어 특화 Qwen 기반 ColPali 모델 로딩 중...")
    RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v0.1")

    print(f"\n📊 총 {total_count}개 파일 대상 인덱싱을 시작합니다.")
    print("-" * 50)

    # 인덱싱 실행 (Byaldi 내부 로그가 document 번호를 보여줍니다)
    RAG.index(
        input_path=input_path,
        index_name=index_name,
        overwrite=True, 
        store_collection_with_index=True
    )
    
    print("-" * 50)
    print(f"✨ 완료! 이제 모든 공지사항이 4090 GPU에 의해 비전 DB로 변환되었습니다.")

if __name__ == "__main__":
    run_qwen_indexing()