import os
import json
import re
from pathlib import Path

class ContextualChunker:
    def __init__(self):
        # 쪼개진 텍스트가 너무 짧으면 의미가 없으므로 합치기 위한 최소 길이
        self.min_chunk_length = 50 

    def create_global_context(self, data):
        """JSON 데이터에서 글로벌 꼬리표(Context) 생성"""
        title = data.get("title", "제목 없음")
        meta = data.get("metadata", {})
        category = meta.get("category", "분류없음")
        entity = meta.get("entity", "부서없음")
        
        # 꼬리표 포맷: [카테고리 - 부서명] 공지사항 제목
        return f"[{category} - {entity}] {title}"

    def split_markdown_content(self, content):
        """마크다운 본문을 의미 단위(단락/헤더)로 분할"""
        # \n\n 이나 마크다운 헤더(###, 1. 등)를 기준으로 텍스트를 나눔
        # 정제된 데이터는 빈 줄(\n\n)로 구분이 잘 되어 있으므로 이를 적극 활용
        raw_blocks = re.split(r'\n\n+', content.strip())
        
        chunks = []
        current_chunk = ""

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            # 만약 블록이 너무 짧으면(예: 50자 이하) 다음 블록과 합침 (파편화 방지)
            if len(current_chunk) + len(block) < self.min_chunk_length:
                current_chunk += "\n" + block if current_chunk else block
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = block
        
        # 마지막 남은 덩어리 처리
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def process_file(self, filepath):
        """단일 JSON 파일을 읽어 청크 리스트로 반환"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        parent_id = filepath.stem # 파일명 (예: '78776')
        global_context = self.create_global_context(data)
        refined_content = data.get("refined_content", "")

        raw_chunks = self.split_markdown_content(refined_content)
        
        chunked_data_list = []
        for idx, chunk_text in enumerate(raw_chunks):
            # 조각의 맨 앞에 글로벌 꼬리표 부착 (핵심 로직!!!)
            contextualized_text = f"문서정보: {global_context}\n\n내용:\n{chunk_text}"
            
            chunk_obj = {
                "chunk_id": f"{parent_id}_chunk_{idx+1}",
                "parent_id": parent_id, # Big 데이터와 연결될 고리
                "metadata": data.get("metadata", {}), # 원본 메타데이터 상속
                "chunk_text": contextualized_text # 임베딩 및 검색될 실제 텍스트
            }
            chunked_data_list.append(chunk_obj)

        return chunked_data_list

def run_chunking_pipeline(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunker = ContextualChunker()
    files = list(input_path.glob("*.json"))
    
    total_original_files = len(files)
    total_chunks_created = 0

    print(f"🚀 청킹 파이프라인 시작 (대상: {total_original_files}개 파일)")

    for i, file in enumerate(files):
        # 청킹 작업 수행
        chunk_list = chunker.process_file(file)
        
        if not chunk_list:
            continue

        # 결과 저장: 78776.json -> 78776_chunks.json (리스트 형태로 저장)
        target_file = output_path / f"{file.stem}_chunks.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(chunk_list, f, ensure_ascii=False, indent=2)
            
        total_chunks_created += len(chunk_list)
        
        if (i+1) % 100 == 0 or (i+1) == total_original_files:
            print(f"[{i+1}/{total_original_files}] 처리 중... (현재 누적 청크: {total_chunks_created}개)")

    print("="*50)
    print("✅ 청킹 완료!")
    print(f"📄 원본 파일 수: {total_original_files}개")
    print(f"🧩 생성된 총 청크 수: {total_chunks_created}개")
    print(f"📂 저장 위치: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    # 경로 설정 (사진의 구조에 맞춤)
    INPUT_DIR = "./data/processed/text"
    OUTPUT_DIR = "./data/processed/chunks"
    
    run_chunking_pipeline(INPUT_DIR, OUTPUT_DIR)