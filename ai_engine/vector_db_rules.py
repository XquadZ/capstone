import os
import json
import time
import numpy as np
from pathlib import Path
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from FlagEmbedding import BGEM3FlagModel

class RuleMilvusIndexer:
    # 💡 에러 회피를 위해 컬렉션 이름을 새롭게 지정했습니다.
    def __init__(self, collection_name="hoseo_rules_v1"):
        self.collection_name = collection_name
        
        # 1. BGE-M3 모델 로드 (4090 GPU 활용)
        print("🤖 BGE-M3 모델 로드 중...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        # 2. Milvus 서버 연결
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus DB 연결 완료!")

    def create_collection(self):
        """기존 컬렉션을 삭제하고, 학칙 스키마로 새 컬렉션 생성"""
        if utility.has_collection(self.collection_name):
            print(f"🗑️ 기존 컬렉션 '{self.collection_name}' 삭제 중... (초기화 후 재생성)")
            utility.drop_collection(self.collection_name)

        # 학칙 데이터 구조에 맞춘 필드 스키마 정의
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="categories", dtype=DataType.JSON), # LLM이 뽑은 3개 카테고리 (배열 형태)
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]

        schema = CollectionSchema(fields, description="호서대 학칙/규정 하이브리드 RAG 컬렉션")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        print("⚙️ 하이브리드 인덱스 생성 중...")
        # Dense Vector Index (의미 기반 검색용)
        self.collection.create_index(
            field_name="dense_vector", 
            index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
        )
        # Sparse Vector Index (키워드 기반 검색용)
        self.collection.create_index(
            field_name="sparse_vector", 
            index_params={"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}}
        )
        print(f"✅ 학칙용 하이브리드 스키마 및 인덱스({self.collection_name}) 준비 완료!")

    def insert_chunks(self, json_path):
        """단일 JSON 파일을 읽어 Batch 단위로 임베딩 후 삽입"""
        if not os.path.exists(json_path):
            print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        self.collection.load()
        total_chunks = len(chunks)
        print(f"🚀 인덱싱 시작 (대상: 총 {total_chunks}개 청크)")

        batch_size = 32 # 4090 GPU에 맞춘 적정 배치 사이즈
        total_inserted = 0
        start_time = time.time()

        for i in range(0, total_chunks, batch_size):
            try:
                batch = chunks[i : i + batch_size]
                
                # 데이터 추출
                doc_ids = [c.get("doc_id", "") for c in batch]
                page_nums = [c.get("page_num", 0) for c in batch]
                texts = [c.get("text", "") for c in batch]
                sources = [c.get("source", "") for c in batch]
                categories = [c.get("categories", []) for c in batch]

                # 🧠 BGE-M3 하이브리드 임베딩 생성
                embeddings = self.model.encode(texts, return_dense=True, return_sparse=True)

                # float32 변환 (Milvus 요구사항)
                dense_vecs = embeddings['dense_vecs'].astype(np.float32)
                sparse_vecs = embeddings['lexical_weights']

                # Milvus 삽입 (스키마 순서: doc_id, page_num, text, source, categories, dense, sparse)
                data = [
                    doc_ids, page_nums, texts, sources, categories, 
                    dense_vecs, sparse_vecs
                ]

                self.collection.insert(data)
                total_inserted += len(batch)
                
                print(f"🔄 진행 상황: {total_inserted}/{total_chunks} 완료...", end='\r')

            except Exception as e:
                print(f"\n❌ 배치 삽입 중 에러 발생 (Index {i}): {e}")

        self.collection.flush()
        elapsed = time.time() - start_time
        print(f"\n🎉 모든 인덱싱 완료! (소요 시간: {elapsed:.1f}초)")
        print(f"✅ 최종 저장된 엔티티 수: {self.collection.num_entities}")


if __name__ == "__main__":
    # 파일 경로 설정
    BASE_DIR = Path(__file__).parent.parent
    INPUT_JSON = BASE_DIR / "data" / "rules_regulations" / "chunks" / "all_rules_chunks_meta.json"
    
    # 💡 만약 경로를 못 찾는다는 에러가 나면 아래의 절대경로 주석을 해제하고 사용하세요.
    # INPUT_JSON = r"C:\Users\DMLAB_Server1\capstone\data\rules_regulations\chunks\all_rules_chunks_meta.json"
    
    indexer = RuleMilvusIndexer() # 기본값이 hoseo_rules_v1 으로 설정됨
    indexer.create_collection()
    indexer.insert_chunks(INPUT_JSON)