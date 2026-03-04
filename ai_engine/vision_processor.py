import os
import json
import base64
import fitz  # PyMuPDF
from openai import OpenAI

# 1. 환경 변수에서 API 키 로드
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    """이미지 파일을 Base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_pdf(pdf_path):
    """PDF에서 텍스트 추출"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"   ❌ PDF 추출 에러: {e}")
    return text

def summarize_with_ai(raw_text, data_type="텍스트"):
    """Text AI (GPT-4o-mini)를 이용한 요약"""
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
    """Multimodal AI (GPT-4o)를 이용한 이미지 분석"""
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

def process_notice_data(target_idx="95774"):
    """전체 데이터 처리 파이프라인"""
    # 경로 설정
    base_dir = os.path.join(os.getcwd(), "data", "raw", target_idx)
    info_path = os.path.join(base_dir, "info.json")
    
    if not os.path.exists(info_path):
        print(f"❌ 에러: {info_path} 파일이 없습니다.")
        return

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # 1. 본문 텍스트 요약
    print("🧠 [1] 본문 텍스트 요약 중...")
    info["ai_summary_content"] = summarize_with_ai(info["content"], "본문 텍스트")

    # 2. 이미지 데이터 분석 (경로 수정됨)
    info["ai_image_summaries"] = []
    print("👁️ [2] 포스터 이미지 분석 중...")
    for img_rel_path in info.get("saved_images", []):
        # 'images/img_1.jpg' -> 'img_1.jpg' 추출 후 경로 재조합
        img_name = os.path.basename(img_rel_path)
        full_img_path = os.path.join(base_dir, "images", img_name)
        
        if os.path.exists(full_img_path):
            print(f"   🔎 이미지 처리 중: {img_name}")
            summary = analyze_image_with_vision(full_img_path)
            info["ai_image_summaries"].append({
                "image": img_rel_path,
                "summary": summary
            })
            print(f"   ✅ {img_name} 분석 완료")
        else:
            print(f"   ⚠️ 이미지를 찾을 수 없음: {full_img_path}")

    # 3. PDF 데이터 요약
    info["ai_attachment_summaries"] = []
    print("📄 [3] 첨부파일(PDF) 요약 중...")
    for attach_rel_path in info.get("saved_attachments", []):
        if attach_rel_path.lower().endswith(".pdf"):
            # 파일명 추출 및 경로 조합
            attach_name = os.path.basename(attach_rel_path)
            full_pdf_path = os.path.join(base_dir, "attachments", attach_name)
            
            if os.path.exists(full_pdf_path):
                raw_pdf_text = extract_text_from_pdf(full_pdf_path)
                pdf_summary = summarize_with_ai(raw_pdf_text, "첨부 PDF")
                info["ai_attachment_summaries"].append({
                    "file": attach_rel_path,
                    "summary": pdf_summary
                })
                print(f"   ✅ {attach_name} 분석 완료")

    # 4. 최종 데이터 저장
    processed_dir = os.path.join(os.getcwd(), "data", "processed", target_idx)
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "ai_extracted_info.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
        
    print(f"\n✨ 분석 완료! 결과 확인: {output_path}")

if __name__ == "__main__":
    process_notice_data("95774")