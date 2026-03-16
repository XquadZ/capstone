import os
import json
import fitz  # PyMuPDF
import olefile
from pathlib import Path
import easyocr
import warnings
import numpy as np
from PIL import Image
import io

# 라이브러리 경고 및 MuPDF 내부 에러 출력 억제
warnings.filterwarnings("ignore")
fitz.TOOLS.mupdf_display_errors(False) 

class FullTextExtractor:
    def __init__(self):
        print("🔍 OCR 모델 로드 중 (GPU 사용)...")
        # 한글, 영어 모델 로드
        self.reader = easyocr.Reader(['ko', 'en'], gpu=True)

    def extract_hwp(self, file_path):
        """HWP에서 텍스트 추출 (OLE 구조 분석)"""
        try:
            f = olefile.OleFileIO(file_path)
            dirs = f.listdir()
            if ["PrvText"] in dirs:
                text = f.openstream("PrvText").read().decode("utf-16le")
                return text
            return ""
        except Exception as e:
            return f"[HWP 에러: {str(e)}]"

    def extract_pdf(self, file_path):
        """PDF 텍스트 추출 + 폰트 깨짐 대비 OCR 백업 로직"""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                # 1. 일반적인 방식으로 텍스트 추출 시도
                page_text = page.get_text().strip()
                
                # 2. 추출된 텍스트가 너무 적거나(5자 미만) 에러가 의심될 경우 OCR 실행
                if len(page_text) < 5:
                    # 페이지를 고해상도 이미지로 렌더링 (300 DPI)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    
                    # EasyOCR로 이미지 읽기
                    ocr_result = self.reader.readtext(img_data, detail=0)
                    page_text = " ".join(ocr_result)
                    # print(f"   -> [PDF-OCR 가동] {file_path.name} {page_num+1}페이지")
                
                text += f"--- Page {page_num+1} ---\n{page_text}\n"
            doc.close()
            return text
        except Exception as e:
            return f"[PDF 에러: {str(e)}]"

    def extract_ocr(self, img_path):
        """이미지(JPG, PNG 등) 전용 OCR"""
        try:
            result = self.reader.readtext(str(img_path), detail=0)
            return " ".join(result)
        except Exception as e:
            return f"[OCR 에러: {str(e)}]"

    def process_all(self, base_raw_path, save_base_path):
        raw_path = Path(base_raw_path)
        save_path = Path(save_base_path)
        save_path.mkdir(parents=True, exist_ok=True)

        notice_folders = sorted([f for f in raw_path.iterdir() if f.is_dir()])
        total = len(notice_folders)
        
        print(f"🚀 총 {total}개의 공지사항 처리를 시작합니다.")

        for idx, folder in enumerate(notice_folders):
            notice_id = folder.name
            print(f"[{idx+1}/{total}] 처리 중: {notice_id}")
            
            integrated_parts = []

            # 1. info.json
            info_file = folder / "info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    integrated_parts.append(f"### TITLE: {data.get('title', '')}")
                    integrated_parts.append(f"### DATE: {data.get('date', '')}")
                    integrated_parts.append(f"### URL: {data.get('url', '')}")
                    integrated_parts.append(f"### CONTENT:\n{data.get('content', '')}")

            # 2. Images (OCR)
            img_dir = folder / "images"
            if img_dir.exists():
                img_texts = []
                for img_file in img_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img_texts.append(self.extract_ocr(img_file))
                if img_texts:
                    integrated_parts.append(f"### IMAGE_OCR:\n" + "\n".join(img_texts))

            # 3. Attachments (PDF, HWP)
            attach_dir = folder / "attachments"
            if attach_dir.exists():
                attach_texts = []
                for attach_file in attach_dir.glob("*"):
                    ext = attach_file.suffix.lower()
                    if ext == '.pdf':
                        attach_texts.append(f"<{attach_file.name}>\n" + self.extract_pdf(attach_file))
                    elif ext == '.hwp':
                        attach_texts.append(f"<{attach_file.name}>\n" + self.extract_hwp(attach_file))
                if attach_texts:
                    integrated_parts.append(f"### ATTACHMENT_TEXT:\n" + "\n".join(attach_texts))

            # 4. 결과 통합 및 저장
            final_text = "\n\n".join(integrated_parts)
            with open(save_path / f"{notice_id}.txt", "w", encoding="utf-8") as f:
                f.write(final_text)

if __name__ == "__main__":
    # 경로 설정
    RAW_PATH = "./data/raw"
    SAVE_PATH = "./data/processed/integrated_text"
    
    extractor = FullTextExtractor()
    extractor.process_all(RAW_PATH, SAVE_PATH)
    print("\n✅ 모든 데이터 추출 및 통합 완료!")