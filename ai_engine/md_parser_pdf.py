import os
import time
from pathlib import Path
import pymupdf4llm

def convert_pdfs_to_paged_markdown(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 출력 폴더 생성 (없으면 자동 생성)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_path.glob("*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        print(f"❌ {input_dir} 에 PDF 파일이 없습니다.")
        return

    print(f"🚀 총 {total_files}개의 학칙 PDF ➡️ '페이지 꼬리표(PAGE_X)' 삽입 파싱 시작")
    print("멀티모달(Vision) RAG 실험을 위한 핵심 전처리 작업입니다.\n")

    success_count = 0
    start_time = time.time()

    for idx, pdf_file in enumerate(pdf_files, 1):
        md_filename = f"{pdf_file.stem}.md"
        md_filepath = output_path / md_filename
        
        print(f"[{idx:03d}/{total_files:03d}] 🔄 꼬리표 삽입 중: {pdf_file.name}", end=" ... ")
        
        try:
            # 💡 핵심 로직 1: page_chunks=True 옵션으로 텍스트를 페이지 단위로 쪼개서 받음
            md_chunks = pymupdf4llm.to_markdown(str(pdf_file), page_chunks=True)
            
            full_md_text = ""
            
            # 💡 핵심 로직 2: 각 페이지 텍스트 위에 기계가 인식할 태그 강제 삽입
            for page_idx, chunk in enumerate(md_chunks, 1):
                page_text = chunk.get('text', '')
                
                # 이 태그가 나중에 Vision 모델이 볼 이미지(page_1.png)와 연결해 줍니다!
                full_md_text += f"\n\n\n\n"
                full_md_text += page_text
                
            # 최종 마크다운 파일로 덮어쓰기 저장
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(full_md_text.strip())
                
            print("✅ 완료")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 에러 발생: {str(e)}")

    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 페이지 매핑 마크다운 파싱 대장정 완료!")
    print(f"📄 성공: {success_count}/{total_files}개")
    print(f"⏱️ 소요 시간: {elapsed_time:.2f}초")
    print(f"💾 저장 위치: {output_path}")
    print("="*60)

if __name__ == "__main__":
    # 작성자님의 폴더 구조에 맞춘 경로
    INPUT_DIR = "./data/rules_regulations/raw_pdfs"
    OUTPUT_DIR = "./data/rules_regulations/markdown_parsed"
    
    convert_pdfs_to_paged_markdown(INPUT_DIR, OUTPUT_DIR)