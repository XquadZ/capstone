import os
import json
import time
import numpy as np
from pathlib import Path
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from FlagEmbedding import BGEM3FlagModel

class MilvusIndexer:
    def __init__(self, collection_name="hoseo_notices"):
        self.collection_name = collection_name
        
        # 1. BGE-M3 모델 로드 (4090 GPU 활용)
        print("🤖 BGE-M3 모델 로드 중...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        # 2. Milvus 서버 연결
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus DB 연결 완료!")

    def create_collection(self):
        """기존 컬렉션을 삭제하고, 확장된 필드 길이로 새 컬렉션 생성"""
        if utility.has_collection(self.collection_name):
            print(f"🗑️ 기존 컬렉션 '{self.collection_name}' 삭제 중... (초기화 후 재생성)")
            utility.drop_collection(self.collection_name)

        # 필드 스키마 정의 - 길이를 대폭 늘려 에러 방지
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="year", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=500), # 50 -> 500
            FieldSchema(name="target", dtype=DataType.VARCHAR, max_length=1000),  # 100 -> 1000 (에러 핵심 해결)
            FieldSchema(name="entity", dtype=DataType.VARCHAR, max_length=500),   # 100 -> 500
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]

        schema = CollectionSchema(fields, description="호서대 공지사항 하이브리드 RAG 컬렉션")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        print("⚙️ 하이브리드 인덱스 생성 중...")
        # Dense Vector Index
        self.collection.create_index(
            field_name="dense_vector", 
            index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )
        # Sparse Vector Index
        self.collection.create_index(
            field_name="sparse_vector", 
            index_params={"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}}
        )
        print("✅ 확장된 스키마로 컬렉션 준비 완료!")

    def insert_chunks(self, chunks_dir):
        """데이터 삽입 루프"""
        chunks_path = Path(chunks_dir)
        files = list(chunks_path.glob("*_chunks.json"))
        
        self.collection.load()
        print(f"🚀 인덱싱 시작 (대상 파일: {len(files)}개)")

        total_inserted = 0
        start_time = time.time()

        for i, file_path in enumerate(files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                if not chunks: continue

                # 데이터 추출
                chunk_ids = [c["chunk_id"] for c in chunks]
                parent_ids = [c["parent_id"] for c in chunks]
                texts = [c["chunk_text"] for c in chunks]
                years = [str(c["metadata"].get("year", "")) for c in chunks]
                categories = [str(c["metadata"].get("category", "")) for c in chunks]
                targets = [str(c["metadata"].get("target", "")) for c in chunks]
                entities = [str(c["metadata"].get("entity", "")) for c in chunks]

                # BGE-M3 임베딩 생성
                embeddings = self.model.encode(texts, return_dense=True, return_sparse=True, batch_size=12)

                # float32 변환
                dense_vecs = embeddings['dense_vecs'].astype(np.float32)

                # Milvus 삽입 (리스트 순서 주의)
                data = [
                    chunk_ids, parent_ids, years, categories,
                    targets, entities, texts, dense_vecs,
                    embeddings['lexical_weights']
                ]

                self.collection.insert(data)
                total_inserted += len(chunks)

                if (i+1) % 50 == 0 or (i+1) == len(files):
                    elapsed = time.time() - start_time
                    print(f"[{i+1}/{len(files)}] {file_path.name} 완료 (누적: {total_inserted}, {elapsed:.1f}s)")

            except Exception as e:
                print(f"❌ {file_path.name} 에러: {e}")

        self.collection.flush()
        print(f"\n🎉 모든 에러 해결 및 인덱싱 완료!")
        print(f"✅ 최종 저장된 청크 수: {self.collection.num_entities}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"
    
    indexer = MilvusIndexer()
    indexer.create_collection() # 여기서 삭제 후 재생성함
    indexer.insert_chunks(CHUNKS_DIR)