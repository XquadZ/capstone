import os
import sys
import io
import json
import base64
import time
import random
import re
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 🛡️ 0. 설정 및 API 클라이언트
# ==========================================
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SAIFEX_API_KEY = os.environ.get("SAIFEX_API_KEY")
if not SAIFEX_API_KEY:
    raise ValueError("❌ 'SAIFEX_API_KEY'가 없습니다. 환경변수를 확인하세요.")

client = OpenAI(
    api_key=SAIFEX_API_KEY,
    base_url="https://ahoseo.saifex.ai/v1"
)
MODEL_NAME = "gpt-4o-mini"

TARGET_QA_COUNT = 300
OUTPUT_PATH = "evaluation/datasets/vision_ragas_testset_300.json"
SAVE_INTERVAL = 10  # 10문제마다 실시간 저장

# ==========================================
# 🛠️ 1. 유틸리티: 찐 학칙 PDF 목록 수집
# ==========================================
def get_all_pdf_paths():
    """오직 'data/rules_regulations/raw_pdfs' 폴더에서 찐 학생용 학칙 PDF만 수집합니다."""
    pdf_paths = []
    rules_dir = "data/rules_regulations/raw_pdfs"
    
    if os.path.exists(rules_dir):
        for f in os.listdir(rules_dir):
            if f.lower().endswith('.pdf'):
                # 💡 핵심 방어: '1-' 로 시작하거나 이름에 핵심 키워드가 있는 학생 규정만 쏙쏙!
                # '2-' 등으로 시작하는 교직원용 행정/인사/기록물 규정은 여기서 자동 탈락됩니다.
                if f.startswith('1-') or '학칙' in f or '장학' in f:
                    pdf_paths.append(os.path.join(rules_dir, f))
                    
    # 공지사항(data/raw) 폴더 뒤지는 로직은 완전히 삭제 완료!
    return pdf_paths

def pdf_page_to_base64(pdf_path, page_num):
    """PDF의 특정 페이지만 해상도 최적화하여 Base64로 변환 (토큰 폭탄 방지)"""
    try:
        images = convert_from_path(pdf_path, dpi=100, first_page=page_num, last_page=page_num)
        if not images: return None
            
        img = images[0]
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        return None

# ==========================================
# 🤖 2. Vision LLM 질문 생성기 (매운맛 프롬프트)
# ==========================================
def generate_qa_from_vision(pdf_path, page_num, base64_img, num_questions):
    filename = os.path.basename(pdf_path)
    
    system_prompt = (
        "당신은 제공된 학칙 및 규정 문서(이미지)를 분석하여 RAG 시스템 평가용 고품질 Q&A 데이터셋을 구축하는 AI입니다.\n"
        "문서에 포함된 표(Table), 수치, 예외 조건 등을 바탕으로 적절한 추론이 필요한 자연스러운 질문을 생성하세요."
    )
    
    user_prompt = f"""
첨부된 이미지는 '{filename}' 문서의 {page_num}페이지입니다.
이 이미지를 바탕으로, 팩트 기반의 정확한 추론이 요구되는 깔끔한 질문과 모범 답안 쌍을 정확히 {num_questions}개 생성하세요.

[제약사항]
1. 억지스러운 페르소나(예: 과장된 학생 말투)는 배제하고, 규정이나 표의 내용을 명확하게 묻는 문장으로 작성하세요.
2. 🚨 [치명적 감점 요소] "이 문서에 따르면", "이 표에 의하면", "자료를 보면", "제3조에 의하면" 같은 기계적인 출처 언급 표현은 절대, 네버, 무조건 빼고 질문하세요. 그냥 다이렉트로 물어보세요. (예: "기계공학과의 선택 전공 학점은 몇 학점이야?")
3. 난이도: 단순한 제목 읽기가 아니라, 이미지 내의 '표(Table) 교차 데이터'나 '복잡한 조건문'을 이해해야만 답할 수 있는 질문일수록 좋습니다.
4. 정답(Ground Truth): 이미지 내의 정보를 바탕으로 객관적이고 정확하게 팩트 위주로 작성하세요.

[출력 형식 (순수 JSON)]
{{
  "qa_pairs": [
    {{
      "question": "깔끔하고 다이렉트한 질문",
      "ground_truth": "이미지 기반의 정확한 팩트 모범 답안",
      "context_used": "{filename} {page_num}페이지"
    }}
  ]
}}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"}}
            ]
        }
    ]

    for attempt in range(3): # 재시도 로직
        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            content = res.choices[0].message.content.strip()
            parsed = json.loads(content)
            return parsed.get("qa_pairs", [])
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"\n❌ Vision API 에러: {e}")
                return []

# ==========================================
# 🚀 3. 메인 실행 루프 (중복 방지 & 이어하기)
# ==========================================
def main():
    pdf_paths = get_all_pdf_paths()
    if not pdf_paths:
        print("❌ 처리할 학생용 학칙 PDF 파일을 찾을 수 없습니다.")
        return
        
    print(f"📥 총 {len(pdf_paths)}개의 찐 학칙 PDF 파일을 로드했습니다.")

    # 각 PDF별로 "전체 페이지 수"와 "사용한 페이지 목록" 추적
    pdf_info = {}
    print("📊 PDF 페이지 정보 분석 중 (수십 초 정도 소요될 수 있습니다)...")
    for path in pdf_paths:
        try:
            info = pdfinfo_from_path(path)
            total_pages = info.get('Pages', 0)
            if total_pages > 0:
                pdf_info[path] = {
                    "total_pages": total_pages,
                    "used_pages": set()
                }
        except Exception:
            pass # 암호 걸린 PDF 등은 스킵
            
    valid_pdfs = list(pdf_info.keys())
    print(f"✅ 사용 가능한 정상 PDF {len(valid_pdfs)}개 확인 완료.")

    final_dataset = []
    start_index = 0

    # 💡 이어하기(Resume) 로직
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                final_dataset = json.load(f)
            start_index = len(final_dataset)
            print(f"🔄 [이어하기] 기존 {start_index}개의 데이터를 발견했습니다.")
            
            # 기존에 사용했던 페이지들 다시 세팅 (중복 방지)
            for item in final_dataset:
                context = item.get("context_used", "")
                for path in valid_pdfs:
                    filename = os.path.basename(path)
                    if filename in context:
                        # "파일명 3페이지" 형태에서 숫자 추출
                        match = re.search(r'(\d+)페이지', context)
                        if match:
                            page_num = int(match.group(1))
                            pdf_info[path]["used_pages"].add(page_num)
        except json.JSONDecodeError:
            print("⚠️ 기존 파일이 손상되어 처음부터 시작합니다.")
            final_dataset = []

    pbar = tqdm(total=TARGET_QA_COUNT, initial=start_index, desc="Vision RAG 킬러 질문 생성")

    # 💡 300개가 다 채워질 때까지 PDF들을 계속 순회
    while len(final_dataset) < TARGET_QA_COUNT:
        random.shuffle(valid_pdfs) # 순차적으로 돌되, PDF 순서는 섞어서 다양성 확보
        
        pages_found_in_this_loop = False
        
        for pdf_path in valid_pdfs:
            if len(final_dataset) >= TARGET_QA_COUNT:
                break
                
            info = pdf_info[pdf_path]
            total_p = info["total_pages"]
            used_p = info["used_pages"]
            
            # 남은 사용 가능한 페이지 찾기
            available_pages = [p for p in range(1, total_p + 1) if p not in used_p]
            
            if not available_pages:
                continue # 이 PDF의 모든 페이지를 다 썼다면 패스
                
            pages_found_in_this_loop = True
            
            # 랜덤하게 안 쓴 페이지 하나 선택
            target_page = random.choice(available_pages)
            used_p.add(target_page)
            
            # 💡 1개 또는 2개의 질문을 랜덤하게 요청
            num_questions_to_ask = random.choice([1, 2])
            
            b64_img = pdf_page_to_base64(pdf_path, target_page)
            if not b64_img: continue
            
            qa_pairs = generate_qa_from_vision(pdf_path, target_page, b64_img, num_questions_to_ask)
            
            if qa_pairs:
                for pair in qa_pairs:
                    if len(final_dataset) < TARGET_QA_COUNT:
                        final_dataset.append(pair)
                        pbar.update(1)
                
                # 💡 10개마다 실시간 저장 (안전장치)
                if len(final_dataset) % SAVE_INTERVAL == 0:
                    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
                    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
                        
            time.sleep(1.0) # API Limit 방지
            
        # 모든 PDF를 순회했는데도 쓸 페이지가 없다면 무한루프 탈출
        if not pages_found_in_this_loop:
            print("\n⚠️ 더 이상 추출할 수 있는 새로운 PDF 페이지가 없습니다. 여기서 종료합니다.")
            break

    pbar.close()

    # 최종 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 구축 완료! 총 {len(final_dataset)}개의 고품질 찐 학칙 추론형 QA가 저장되었습니다: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()