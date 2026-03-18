import fitz
import pytesseract
from PIL import Image
import io
import os

# Tesseract 경로 설정 (설치 경로에 맞춰 확인 필요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_final_ocr(pdf_path, page_num=0):
    if not os.path.exists(pdf_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        return

    print(f"📄 대상: {os.path.basename(pdf_path)} ({page_num+1}페이지)")
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # 해상도를 3배로 높여서 인식률 극대화 (zoom=3.0)
        zoom = 3.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # 이미지 로드
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # OCR 실행 (psm 3: 표준 자동 레이아웃 분석 모드)
        print("🔍 OCR 엔진 가동 중...")
        ocr_text = pytesseract.image_to_string(img, lang='kor+eng', config='--psm 3')
        
        print("\n" + "="*50)
        print("✨ [1-1-1-1 학칙 OCR 결과] ✨")
        print("="*50)
        print(ocr_text.strip())
        print("="*50)
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    base_path = os.getcwd()
    
    # 🎯 1-1-1-1 학칙 PDF 경로
    target_pdf = os.path.join(base_path, "data", "rules_regulations", "raw_pdfs", "1-1-1-1. 학칙.pdf")
    
    test_final_ocr(target_pdf, page_num=0)