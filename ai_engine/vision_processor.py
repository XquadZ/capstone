import os
import json
import base64
import fitz  # PyMuPDF
from openai import OpenAI

# 1. 환경 변수에서 API 키 로드
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"   ❌ PDF 추출 에러: {e}")
    return text

def summarize_with_ai(raw_text, data_type="텍스트"):
    if not raw_text.strip(): return "내용 없음"
    prompt = f"""
    다음은 학교 공지사항의 {data_type} 원본입니다. 
    학생들에게 필요한 핵심 정보(모집 기간, 지원 자격, 주요 혜택, 신청 방법 등)만 추출해서 요약해주세요.
    
    [원본 데이터]
    {raw_text[:4000]}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def analyze_image_with_vision(image_path):
    base64_image = encode_image(image_path)
    prompt = """
    이 이미지는 대학교 모집 포스터입니다. 
    포스터 내의 텍스트를 읽고 다음 정보를 요약해주세요:
    1. 모집 명칭 및 목적
    2. 주요 일정 및 접수 방법
    3. 지원 자격 및 선발 혜택
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content

def process_single_notice(target_idx):
    """하나의 공지 데이터를 처리하는 단위 로직"""
    base_dir = os.path.join(os.getcwd(), "data", "raw", target_idx)
    info_path = os.path.join(base_dir, "info.json")
    
    if not os.path.exists(info_path): return

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    print(f"\n🚀 [{target_idx}] 분석 시작: {info.get('title')}")

    # 1. 본문 요약
    info["ai_summary_content"] = summarize_with_ai(info["content"], "본문 텍스트")

    # 2. 이미지 분석
    info["ai_image_summaries"] = []
    for img_rel_path in info.get("saved_images", []):
        img_name = os.path.basename(img_rel_path)
        full_img_path = os.path.join(base_dir, "images", img_name)
        if os.path.exists(full_img_path):
            summary = analyze_image_with_vision(full_img_path)
            info["ai_image_summaries"].append({"image": img_rel_path, "summary": summary})

    # 3. PDF 요약
    info["ai_attachment_summaries"] = []
    for attach_rel_path in info.get("saved_attachments", []):
        if attach_rel_path.lower().endswith(".pdf"):
            attach_name = os.path.basename(attach_rel_path)
            full_pdf_path = os.path.join(base_dir, "attachments", attach_name)
            if os.path.exists(full_pdf_path):
                raw_pdf_text = extract_text_from_pdf(full_pdf_path)
                pdf_summary = summarize_with_ai(raw_pdf_text, "첨부 PDF")
                info["ai_attachment_summaries"].append({"file": attach_rel_path, "summary": pdf_summary})

    # 4. 결과 저장
    processed_dir = os.path.join(os.getcwd(), "data", "processed", target_idx)
    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, "ai_extracted_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    print(f"✅ [{target_idx}] 분석 완료 및 저장됨")

def process_all_raw_data():
    """data/raw 폴더의 모든 폴더를 순회하며 처리"""
    raw_root = os.path.join(os.getcwd(), "data", "raw")
    if not os.path.exists(raw_root):
        print("❌ data/raw 폴더가 없습니다.")
        return

    # 폴더 목록 추출
    target_folders = [f for f in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, f))]
    
    print(f"📂 총 {len(target_folders)}개의 공지사항을 처리합니다: {target_folders}")
    
    for folder_name in target_folders:
        process_single_notice(folder_name)

if __name__ == "__main__":
    process_all_raw_data()