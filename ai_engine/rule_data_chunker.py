import os
import json
import re
from pathlib import Path

class RuleContextualChunker:
    def __init__(self):
        self.min_chunk_length = 100  # 학칙은 문맥이 중요하므로 최소 길이를 조금 늘림

    def split_by_page_and_paragraph(self, content):
        """페이지 태그를 기준으로 먼저 나누고, 그 안에서 문단 단위로 분할"""
        chunks = []
        
        # 1. 페이지 태그()를 기준으로 텍스트 분할
        # 정규식을 써서 페이지 번호와 해당 페이지의 텍스트를 발췌
        page_splits = re.split(r'', content)
        
        # page_splits[0]은 첫 태그 이전의 빈 공간이므로 무시
        # page_splits[1] = 페이지 번호, page_splits[2] = 텍스트, page_splits[3] = 페이지 번호...
        for i in range(1, len(page_splits), 2):
            page_num = int(page_splits[i])
            page_text = page_splits[i+1].strip()
            
            if not page_text:
                continue

            # 2. 해당 페이지 안에서 문단(\n\n) 단위로 다시 쪼개기 (표 구조는 유지됨)
            raw_blocks = re.split(r'\n\n+', page_text)
            current_chunk = ""

            for block in raw_blocks:
                block = block.strip()
                if not block:
                    continue

                if len(current_chunk) + len(block) < self.min_chunk_length:
                    current_chunk += "\n\n" + block if current_chunk else block
                else:
                    if current_chunk:
                        # 청크 저장 시 소속 '페이지 번호'를 반드시 함께 저장
                        chunks.append({"page": page_num, "text": current_chunk})
                    current_chunk = block
            
            if current_chunk:
                chunks.append({"page": page_num, "text": current_chunk})

        return chunks

    def process_file(self, filepath):
        """마크다운 파일을 읽어 청크 리스트로 반환"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = filepath.stem  # 예: '1-1-1-1. 학칙'
        
        # 문서 전체의 꼬리표 (학칙은 파일명 자체가 가장 중요한 컨텍스트임)
        global_context = f"[학칙 및 규정] {filename}"

        # 페이지 및 문단 단위로 쪼개기
        raw_chunks = self.split_by_page_and_paragraph(content)
        
        chunked_data_list = []
        for idx, item in enumerate(raw_chunks):
            page_num = item['page']
            chunk_text = item['text']
            
            # 검색 성능 향상을 위해 텍스트 앞에 출처 명시
            contextualized_text = f"문서: {global_context}\n페이지: {page_num}p\n\n내용:\n{chunk_text}"
            
            chunk_obj = {
                "chunk_id": f"{filename}_p{page_num}_{idx+1}",
                "source_file": f"{filename}.pdf",
                "page_number": page_num,  # 🚨 Vision 실험을 위한 가장 중요한 메타데이터!
                "image_filename": f"{filename}_page_{page_num}.png", # 연결될 이미지 파일명 미리 세팅
                "chunk_text": contextualized_text
            }
            chunked_data_list.append(chunk_obj)

        return chunked_data_list

def run_chunking_pipeline(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunker = RuleContextualChunker()
    md_files = list(input_path.glob("*.md"))
    
    total_original_files = len(md_files)
    total_chunks_created = 0
    all_chunks = []

    print(f"🚀 학칙 청킹 파이프라인 시작 (대상: {total_original_files}개 파일)")

    for i, file in enumerate(md_files):
        chunk_list = chunker.process_file(file)
        if not chunk_list:
            continue
            
        all_chunks.extend(chunk_list)
        total_chunks_created += len(chunk_list)
        
        if (i+1) % 50 == 0 or (i+1) == total_original_files:
            print(f"[{i+1:03d}/{total_original_files:03d}] 처리 중... (현재 누적 청크: {total_chunks_created}개)")

    # Vector DB에 넣기 편하도록 모든 청크를 하나의 거대한 JSON으로 병합 저장
    target_file = output_path / "all_rules_chunks.json"
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("✅ 학칙 청킹 완료!")
    print(f"📄 원본 파일 수: {total_original_files}개")
    print(f"🧩 생성된 총 청크 수: {total_chunks_created}개")
    print(f"💾 통합 저장 위치: {target_file}")
    print("="*60)

if __name__ == "__main__":
    # 경로 설정
    INPUT_DIR = "./data/rules_regulations/markdown_parsed"
    OUTPUT_DIR = "./data/rules_regulations/chunks"
    
    run_chunking_pipeline(INPUT_DIR, OUTPUT_DIR)