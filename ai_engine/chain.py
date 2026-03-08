from byaldi import RAGMultiModalModel

# 모델 로드 및 인덱싱 실행
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
RAG.index(
    input_path="data/byaldi_input",
    index_name="hoseo_vision_index", # 이 이름으로 .byaldi 안에 폴더가 생깁니다.
    overwrite=True
)