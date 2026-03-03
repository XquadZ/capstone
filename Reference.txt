1. RAG의 바이블 (전체 흐름 파악용)
논문명: Retrieval-Augmented Generation for Large Language Models: A Survey

핵심 내용: RAG의 발전 과정을 Naive RAG, Advanced RAG, Modular RAG 세 단계로 아주 상세하게 분류하고 분석한 종합 리뷰 논문입니다. 검색, 생성, 증강 기술의 핵심 요소와 최신 평가 프레임워크까지 모두 다루고 있습니다.

추천 이유: 논문의 '관련 연구(Related Work)' 챕터를 쓰실 때 무조건 인용하게 될 1순위 논문입니다.

2. 구글 딥마인드의 혁신 (기술 고도화 참고용)
논문명: Corrective Retrieval Augmented Generation (CRAG)

핵심 내용: 구글 딥마인드에서 개발한 모델로, 검색된 문서가 질문과 진짜 관련이 있는지 평가하는 '가벼운 검색 평가기(lightweight retrieval evaluator)'를 도입했습니다. 검색 결과에 대해 스스로 피드백 루프를 돌려 생성된 콘텐츠를 검토하고 개선하는 방식을 제안했습니다.

추천 이유: "단순 검색(Baseline)의 한계를 극복하기 위해 어떻게 검증 단계를 거칠 것인가?"에 대한 완벽한 아이디어를 줍니다.

3. 실무/기업형 RAG (캡스톤 디자인 맞춤형)
논문명: Integrating Retrieval-Augmented Generation (RAG) and Knowledge Augmented Generation (KAG) Frameworks to Build Accurate Enterprise Question Answering Systems

출처: IEEE Conference (2025년 채택)

핵심 내용: 기업 환경(전력 회사 데이터 등)에서 정확한 QA 시스템을 만들기 위해 일반적인 RAG와 지식 그래프 기반의 KAG를 결합하는 5가지 전략을 제안했습니다.

추천 이유: 학교 데이터라는 '도메인 특화 지식'을 다루는 질문자님의 캡스톤 환경과 매우 유사하여 실제 아키텍처 설계에 큰 도움이 됩니다.

4. 평가 지표의 새로운 시각 (논문 실험 설계용)
논문명: Correctness is not Faithfulness in Retrieval Augmented Generation

출처: ACM ICTIR (2025년)

핵심 내용: RAG 시스템을 평가할 때 '정답을 맞혔는가(Correctness)'와 '주어진 문서에만 기반해서 대답했는가(Faithfulness)'는 전혀 다른 문제임을 지적하는 연구입니다.

추천 이유: 제가 아까 제안해 드린 "답변 신뢰도 및 할루시네이션 평가"를 논문 주제로 잡으신다면, 이 논문의 실험 방식을 그대로 벤치마킹하기 좋습니다.