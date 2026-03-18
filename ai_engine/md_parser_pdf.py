import os
import fitz  # PyMuPDF
import re

def parse_pdf_to_md_with_smart_tags(pdf_path, output_dir):
    filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(filename)[0]
    
    # 1. 파일명에서 문서 ID 추출 (예: "1-2-1-3. 대학원 장학 규정" -> "1-2-1-3")
    id_match = re.search(r"(\d+(?:-\d+)+)", base_name)
    fallback_id = id_match.group(1) if id_match else "unknown"
    
    # 2. 문서 본문에서 ID와 페이지 번호를 찾는 정규표현식 (예: 1-1-1-1∼1)
    pattern = re.compile(r"(\d+(?:-\d+)+)[~∼](\d+)")
    
    try:
        doc = fitz.open(pdf_path)
        full_md_text = ""
        
        for idx, page in enumerate(doc):
            page_text = page.get_text("text")
            
            # 본문에서 패턴 검색
            match = pattern.search(page_text)
            if match:
                current_doc_id = match.group(1) 
                page_num = match.group(2)       
            else:
                # 패턴을 못 찾으면 파일명에서 추출한 ID(fallback_id)와 물리적 페이지 번호 사용
                current_doc_id = fallback_id
                page_num = str(idx + 1)
                
            # 텍스트 정리 (공백 및 줄바꿈 정리)
            full_md_text += page_text.strip()
            
            # ★ 청커를 위한 메타데이터 태그 삽입
            tag = f'\n\n<page doc_id="{current_doc_id}" num="{page_num}"></page>\n\n'
            full_md_text += tag
            
        # 출력 폴더 확인 및 파일 저장
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}.md")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_md_text)
            
        print(f"✅ 파싱 완료 (문서ID: {fallback_id}): {filename}")
        return True
    except Exception as e:
        print(f"❌ 에러 ({filename}): {e}")
        return False

if __name__ == "__main__":
    # 스크립트 파일 위치 기준 절대 경로 생성
    base_path = os.getcwd()
    
    # 🎯 구조에 맞춘 정확한 경로 세팅
    INPUT_PDF_DIR = os.path.join(base_path, "data", "rules_regulations", "raw_pdfs")
    OUTPUT_MD_DIR = os.path.join(base_path, "data", "rules_regulations", "markdown_parsed")
    
    print(f"📂 입력 경로: {INPUT_PDF_DIR}")
    print(f"📂 출력 경로: {OUTPUT_MD_DIR}\n")

    if os.path.exists(INPUT_PDF_DIR):
        pdf_files = [f for f in os.listdir(INPUT_PDF_DIR) if f.lower().endswith('.pdf')]
        print(f"🔍 발견된 PDF: {len(pdf_files)}개")
        
        success_count = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(INPUT_PDF_DIR, pdf_file)
            if parse_pdf_to_md_with_smart_tags(pdf_path, OUTPUT_MD_DIR):
                success_count += 1
        
        print(f"\n🎉 작업 완료: {success_count}/{len(pdf_files)} 성공")
    else:
        print(f"❌ 오류: 경로를 찾을 수 없습니다. ({INPUT_PDF_DIR})")