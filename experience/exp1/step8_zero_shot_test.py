import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# ⚙️ 1. 설정
# ==========================================
BASE_MODEL_ID = "google/gemma-2b-it"
LORA_ADAPTER_DIR = "experience/exp1/gemma_router_lora_v4" 

def main():
    print("🚀 [과적합 검증] 완전히 낯선 문장 패턴으로 V4 모델의 일반화 능력을 테스트합니다!")

    # ==========================================
    # 📝 2. 학습 때 단 한 번도 안 쓴 '완전 새로운' 테스트 쌍 5개
    # ==========================================
    test_cases = [
        # [Pair 1: 일정 확인]
        {"expected": "TEXT", "query": "2024학년도 2학기 수강신청 기간이 언제부터 언제까지인가요?"},
        {"expected": "VISION", "query": "홈페이지에 올라온 달력 이미지 보니까, 2학기 수강신청 며칠부터 시작인지 알 수 있어?"}, # '달력 이미지' 신규 단어

        # [Pair 2: 길 찾기]
        {"expected": "TEXT", "query": "예비군 훈련장은 학교에서 어떻게 가야 해? 버스 노선 좀 알려줘."},
        {"expected": "VISION", "query": "이 약도 보니까 헷갈리는데, 학교에서 예비군 훈련장 가는 길 좀 다시 설명해줄래?"}, # '약도' 신규 단어

        # [Pair 3: 데이터 확인]
        {"expected": "TEXT", "query": "작년도 우리 학과 취업률이 몇 퍼센트였나요?"},
        {"expected": "VISION", "query": "안내 책자에 있는 통계 그래프 상으로, 작년 우리 학과 취업률이 몇 프로로 나와?"}, # '통계 그래프' 신규 단어

        # [Pair 4: 절차 확인]
        {"expected": "TEXT", "query": "졸업 작품 프로젝트 제출 양식과 절차가 어떻게 되나요?"},
        {"expected": "VISION", "query": "첨부된 순서도를 따라가면, 졸업 작품 제출할 때 첫 번째로 해야 할 일이 뭐야?"}, # '순서도' 신규 단어

        # [Pair 5: 표/조건 확인]
        {"expected": "TEXT", "query": "교환학생 지원할 때 토익 점수 커트라인이 있나요?"},
        {"expected": "VISION", "query": "밑에 있는 자격 요건 테이블표 기준으로, 교환학생 지원 가능한 토익 커트라인이 몇 점이야?"} # '테이블표' 신규 단어
    ]

    # ==========================================
    # 🧠 3. 모델 및 토크나이저 로드
    # ==========================================
    print("🤖 V4 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model.eval()

    # ==========================================
    # 🎯 4. 제로샷(Zero-shot) 추론 실행
    # ==========================================
    print("\n" + "="*60)
    print("🎯 [낯선 문장 테스트 결과]")
    print("="*60)
    
    instruction = "주어진 질문에 답변하기 위해 텍스트 기반 검색(TEXT)과 시각적 문서 검색(VISION) 중 어느 것이 더 적합한지 판단하여 하나만 출력하세요."
    correct_count = 0

    for i, case in enumerate(test_cases, 1):
        query = case['query']
        expected = case['expected']

        # 학습 때와 동일한 프롬프트 포맷팅
        prompt = f"<bos><start_of_turn>user\n{instruction}\n질문: {query}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0)
            
        full_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
        
        prediction = "VISION" if "VISION" in full_response else "TEXT"
        
        is_correct = "✅" if prediction == expected else "❌"
        if prediction == expected:
            correct_count += 1

        print(f"[{i:02d}] 정답: {expected:<6} | 예측: {prediction:<6} | {is_correct}")
        print(f" 🗣️ 질문: {query}\n")

    print("="*60)
    print(f"🏆 최종 정답률: {correct_count} / {len(test_cases)} ({correct_count/len(test_cases)*100:.1f}%)")
    if correct_count == len(test_cases):
        print("🔥 미쳤습니다! 모델이 단순 암기를 넘어 '문맥'을 완벽히 이해했습니다! 일반화 대성공!")
    else:
        print("💡 틀린 문제가 있다면, 모델이 어떤 단어를 놓쳤는지 확인해보면 됩니다.")
    print("="*60)

if __name__ == "__main__":
    main()