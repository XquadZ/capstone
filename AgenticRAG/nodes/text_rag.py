import sys
import os
from typing import Dict

# 상위 폴더(capstone)를 시스템 경로에 추가하여 ai_engine 모듈을 찾을 수 있게 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Graph State 구조 임포트
from AgenticRAG.graph.state import AgentState

# 기존에 잘 만들어둔 검색 및 생성 함수 임포트
# (주의: 임포트하는 순간 BGE 모델 로드 및 Milvus 연결이 1회 진행됩니다)
from ai_engine.rag_pipeline_rules import retrieve_documents, generate_answer

def text_rag_node(state: AgentState) -> Dict:
    print("\n--- [NODE: Text RAG] 학칙 텍스트 검색 및 생성 중 ---")
    question = state["question"]
    
    # 1. 문서 검색 (기존 로직)
    chunks = retrieve_documents(question)
    
    # 검색 실패 시 예외 처리
    if not chunks:
        print("    -> 관련된 학칙을 찾을 수 없습니다.")
        return {
            "generation": "제공된 규정에서는 해당 내용을 찾을 수 없습니다.", 
            "context": "검색 결과 없음"
        }
        
    # 2. 문맥 정리 (Graph State 메모리에 저장하기 위한 용도)
    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += f"[문서 {i+1}] {chunk.get('source', '알수없음')} (p.{chunk.get('page_num', '?')})\n"
        
    # 3. 답변 생성 (기존 타자기 효과가 그대로 터미널에 출력됩니다)
    final_answer = generate_answer(question, chunks)
    
    # 4. 상태(State) 업데이트 반환
    return {
        "generation": final_answer,
        "context": context_text
    }