import os
import re
import json

def chunk_markdown_file(filepath, chunk_size=800):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # <page doc_id="..." num="..."></page> 태그를 기준으로 텍스트 분리
    # 텍스트와 메타데이터를 짝지어 추출합니다.
    parts = re.split(r'<page doc_id="(.*?)" num="(.*?)"></page>', content)
    
    chunks = []
    
    # re.split 결과는 [텍스트1, doc_id1, num1, 텍스트2, doc_id2, num2, ...] 형태로 나옵니다.
    # 3개씩 묶어서 처리합니다.
    for i in range(0, len(parts) - 1, 3):
        page_text = parts[i].strip()
        if not page_text:
            continue
            
        doc_id = parts[i+1]
        
        # 페이지 번호가 숫자가 아닐 경우(예: 로마자 등)를 대비한 예외 처리
        try:
            page_num = int(parts[i+2])
        except ValueError:
            page_num = parts[i+2] # 숫자가 아니면 문자열 그대로 유지
        
        # 💡 +-1 페이지 계산 (문자열 페이지 번호일 경우 계산 생략)
        prev_page = page_num - 1 if isinstance(page_num, int) else None
        next_page = page_num + 1 if isinstance(page_num, int) else None

        # 1. 문단(\n\n) 단위로 쪼개기 (마크다운 표는 내부에 빈 줄이 없으므로 표 전체가 하나의 문단으로 묶여 보호됨!)
        paragraphs = page_text.split('\n\n')
        current_chunk_text = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 현재 문단(표 포함)을 더했을 때 청크 사이즈를 넘어가면 잘라냄
            # 단, 문단 하나(예: 거대한 표)가 통째로 chunk_size를 넘는 경우 강제로 찢지 않고 그대로 유지!
            if len(current_chunk_text) > 0 and (len(current_chunk_text) + len(para) > chunk_size):
                chunks.append({
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "prev_page": prev_page,  # 🎯 잊어버리지 않게 미리 세팅!
                    "next_page": next_page,  # 🎯 잊어버리지 않게 미리 세팅!
                    "text": current_chunk_text.strip(),
                    "source": os.path.basename(filepath)
                })
                current_chunk_text = para + "\n\n"
            else:
                current_chunk_text += para + "\n\n"
                
        # 마지막 남은 조각 털어넣기
        if current_chunk_text.strip():
            chunks.append({
                "doc_id": doc_id,
                "page_num": page_num,
                "prev_page": prev_page,
                "next_page": next_page,
                "text": current_chunk_text.strip(),
                "source": os.path.basename(filepath)
            })
            
    return chunks

def process_all_markdowns(input_dir, output_filepath):
    if not os.path.exists(input_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {input_dir}")
        return
        
    md_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.md')]
    print(f"🔍 총 {len(md_files)}개의 Markdown 파일 청킹 시작...\n" + "-"*40)
    
    all_chunks = []
    for md_file in md_files:
        filepath = os.path.join(input_dir, md_file)
        try:
            file_chunks = chunk_markdown_file(filepath)
            all_chunks.extend(file_chunks)
            print(f"✅ {md_file} -> {len(file_chunks)}개 청크 생성 완료")
        except Exception as e:
            print(f"❌ 에러 발생 ({md_file}): {e}")
            
    # 결과를 JSON으로 저장
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
    print("-" * 40)
    print(f"🎉 전체 작업 완료! 총 {len(all_chunks)}개의 청크가 생성되었습니다.")
    print(f"📁 저장 위치: {output_filepath}")

if __name__ == "__main__":
    base_path = os.getcwd()
    
    INPUT_MD_DIR = os.path.join(base_path, "data", "rules_regulations", "markdown_parsed")
    OUTPUT_JSON_PATH = os.path.join(base_path, "data", "rules_regulations", "chunks", "all_rules_chunks.json")
    
    process_all_markdowns(INPUT_MD_DIR, OUTPUT_JSON_PATH)