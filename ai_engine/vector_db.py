import os
import json
from pathlib import Path
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from FlagEmbedding import BGEM3FlagModel

class MilvusIndexer:
    def __init__(self, collection_name="hoseo_notices"):
        self.collection_name = collection_name
        
        # 1. BGE-M3 모델 로드 (4090 GPU 활용, FP16으로 메모리/속도 최적화)
        print("🤖 BGE-M3 모델 로드 중... (시간이 조금 걸릴 수 있습니다)")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        # 2. Milvus 서버 연결 (로컬 도커 기준 기본 포트)
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus DB 연결 완료!")

    def create_collection(self):
        """Milvus 컬렉션(테이블) 스키마 정의 및 생성"""
        if utility.has_collection(self.collection_name):
            print(f"⚠️ '{self.collection_name}' 컬렉션이 이미 존재합니다. 삭제하고 새로 만드시겠습니까? (이 코드는 덮어쓰지 않고 기존 컬렉션을 씁니다)")
            self.collection = Collection(self.collection_name)
            return

        # 필드 정의
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="year", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="target", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="entity", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535), # 원문
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024), # 의미 검색용
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)   # 키워드 검색용
        ]

        schema = CollectionSchema(fields, description="호서대 공지사항 하이브리드 RAG 컬렉션")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # 인덱스 생성 (검색 속도를 위해 필수)
        print("⚙️ 인덱스 생성 중...")
        # Dense 인덱스
        self.collection.create_index(
            field_name="dense_vector", 
            index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )
        # Sparse 인덱스
        self.collection.create_index(
            field_name="sparse_vector", 
            index_params={"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}}
        )
        print("✅ 컬렉션 및 인덱스 생성 완료!")

    def insert_chunks(self, chunks_dir):
        """로컬 폴더의 청크 JSON들을 읽어 벡터화 후 Milvus에 삽입"""
        chunks_path = Path(chunks_dir)
        files = list(chunks_path.glob("*_chunks.json"))
        
        self.collection.load() # 메모리에 컬렉션 적재
        print(f"🚀 인덱싱 시작 (대상 파일: {len(files)}개)")

        for i, file_path in enumerate(files):
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not chunks:
                continue

            # 데이터를 담을 리스트 준비
            data = {
                "chunk_id": [], "parent_id": [], "year": [], "category": [],
                "target": [], "entity": [], "chunk_text": [],
                "dense_vector": [], "sparse_vector": []
            }

            # 텍스트만 모아서 한 번에 임베딩(배치 처리)하면 속도가 훨씬 빠름
            texts_to_embed = [chunk["chunk_text"] for chunk in chunks]
            
            # BGE-M3 임베딩 실행 (Dense, Sparse 동시 추출)
            embeddings = self.model.encode(texts_to_embed, return_dense=True, return_sparse=True)

            for idx, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                
                data["chunk_id"].append(chunk["chunk_id"])
                data["parent_id"].append(chunk["parent_id"])
                data["year"].append(meta.get("year", ""))
                data["category"].append(meta.get("category", ""))
                data["target"].append(meta.get("target", ""))
                data["entity"].append(meta.get("entity", ""))
                data["chunk_text"].append(chunk["chunk_text"])
                
                data["dense_vector"].append(embeddings['dense_vecs'][idx])
                data["sparse_vector"].append(embeddings['lexical_weights'][idx])

            # Milvus에 삽입
            self.collection.insert([
                data["chunk_id"], data["parent_id"], data["year"], data["category"],
                data["target"], data["entity"], data["chunk_text"],
                data["dense_vector"], data["sparse_vector"]
            ])

            if (i+1) % 50 == 0 or (i+1) == len(files):
                print(f"[{i+1}/{len(files)}] {file_path.name} 삽입 완료 (현재 청크 누적 삽입 중...)")

        # 삽입 후에는 반드시 flush를 해줘야 검색이 가능해집니다
        self.collection.flush()
        print(f"🎉 모든 데이터 인덱싱 완료! 총 {self.collection.num_entities}개의 청크가 Milvus에 저장되었습니다.")

if __name__ == "__main__":
    # 청크 데이터가 들어있는 폴더 경로
    CHUNKS_DIR = "./data/processed/chunks"
    
    indexer = MilvusIndexer()
    indexer.create_collection()
    indexer.insert_chunks(CHUNKS_DIR)