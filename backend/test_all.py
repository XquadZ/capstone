import requests
import json

# 조장님 로컬 AI 서버 주소
url = "http://localhost:8000/ask"

# 학습 데이터의 '트리거 문구'를 완벽히 적용한 4가지 시나리오
test_cases = [
    {
        "name": "1. 공지사항 (TV-RAG 켜짐) - 확실한 VISION 트리거",
        # 학습 데이터 패턴: "아래 시각 자료에 따르면 ~ 알려줄 수 있어?"
        "payload": {
            "question": "아래 시각 자료에 따르면 2024년 11월 4일에 청양문화예술회관에서 열리는 행사에서 어떤 프로그램들이 진행될 예정인지 알려줄 수 있어?", 
            "domain": "notice", 
            "use_tv_rag": True
        }
    },
    {
        "name": "2. 공지사항 (텍스트 전용) - 일반 TEXT 트리거",
        # 학습 데이터 패턴: 시각적 단어 없는 일반 질문
        "payload": {
            "question": "2024-2학기 글쓰기 워크숍은 어떤 방식으로 신청하고, 장소는 어디서 확인해?", 
            "domain": "notice", 
            "use_tv_rag": False
        }
    },
    {
        "name": "3. 학칙규정 (TV-RAG 켜짐) - 확실한 VISION 트리거",
        # 학습 데이터 패턴: "문서에 포함된 도표를 바탕으로 ~"
        "payload": {
            "question": "문서에 포함된 도표를 바탕으로 학칙 제20조에 나와있는 이수 학점 설명해줘.", 
            "domain": "rules", 
            "use_tv_rag": True
        }
    },
    {
        "name": "4. 학칙규정 (텍스트 전용) - 일반 TEXT 트리거",
        # 일반적인 규정 질문
        "payload": {
            "question": "호서대학교 졸업 요건이 어떻게 돼?", 
            "domain": "rules", 
            "use_tv_rag": False
        }
    }
]

print("🚀 학습 데이터 기반 AI 서버 4종 통합 테스트를 시작합니다...\n")

for idx, case in enumerate(test_cases, 1):
    print("="*60)
    print(f"▶️ [Test {idx}] {case['name']}")
    print(f"❓ 질문: {case['payload']['question']}")
    
    try:
        response = requests.post(url, json=case['payload'])
        
        if response.status_code == 200:
            result = response.json()
            # 여기서 라우터가 VISION을 제대로 잡았는지 확인!!
            print(f"✅ 라우터 경로: {result.get('route')} 🎯")
            print(f"✅ 참조한 출처: {result.get('sources')}")
            print(f"⏱️ 소요 시간: {result.get('latency_sec')}초")
            print("-" * 60)
            print(f"🤖 [AI 최종 답변]\n{result.get('answer')}")
            print("-" * 60 + "\n")
        else:
            print(f"❌ 에러 발생: {response.status_code} - {response.text}\n")
            
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}\n")

print("🎉 모든 테스트가 완료되었습니다!")