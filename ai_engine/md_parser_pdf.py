import os
import time
from pathlib import Path
import pymupdf4llm  # 표를 마크다운으로 완벽하게 바꿔주는 LLM/RAG 전용 라이브러리

def convert_pdfs_to_markdown(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 출력 폴더가 없으면 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # PDF 파일 목록 가져오기
    pdf_files = list(input_path.glob("*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        print(f"❌ {input_dir} 에 PDF 파일이 없습니다.")
        return

    print(f"🚀 총 {total_files}개의 학칙 PDF ➡️ Markdown 파싱을 시작합니다.")
    print("이 작업은 표(Table) 구조를 추출하므로 시간이 조금 걸릴 수 있습니다.\n")

    success_count = 0
    start_time = time.time()

    for idx, pdf_file in enumerate(pdf_files, 1):
        # 저장할 마크다운 파일명 (예: 1-1-1-1. 학칙.pdf -> 1-1-1-1. 학칙.md)
        md_filename = f"{pdf_file.stem}.md"
        md_filepath = output_path / md_filename
        
        print(f"[{idx:03d}/{total_files:03d}] 🔄 파싱 중: {pdf_file.name}", end=" ... ")
        
        try:
            # 💡 핵심 로직: PDF를 통째로 마크다운으로 변환 (표 구조 보존)
            md_text = pymupdf4llm.to_markdown(str(pdf_file))
            
            # 마크다운 파일로 저장
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(md_text)
                
            print("✅ 완료")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 에러 발생: {str(e)}")

    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 마크다운 파싱 대장정 완료!")
    print(f"📄 성공: {success_count}/{total_files}개")
    print(f"⏱️ 소요 시간: {elapsed_time:.2f}초")
    print(f"💾 저장 위치: {output_path}")
    print("="*60)

if __name__ == "__main__":
    # 작성자님의 프로젝트 구조에 맞춘 경로 설정
    INPUT_DIR = "./data/rules_regulations/raw_pdfs"
    OUTPUT_DIR = "./data/rules_regulations/markdown_parsed"
    
    convert_pdfs_to_markdown(INPUT_DIR, OUTPUT_DIR)