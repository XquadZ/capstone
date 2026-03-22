import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 모델 로드 (경로 확인!)
base_model_id = "google/gemma-2-2b-it"
adapter_path = "hoseo_router_gemma_2b_v2"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)

# 'TEXT'와 'VISION' 토큰 ID 추출
text_token_id = tokenizer.encode("TEXT", add_special_tokens=False)[0]
vision_token_id = tokenizer.encode("VISION", add_special_tokens=False)[0]

def debug_routing_logic(query):
    # 페르소나 포함 주입
    system_persona = "당신은 호서대학교 RAG 시스템의 [AI 라우터]입니다. 다른 부가 설명은 절대 하지 말고 오직 'TEXT' 또는 'VISION'으로만 대답하세요.\n\n"
    full_prompt = system_persona + query + "\n이 질문에 대해 텍스트 RAG와 비전 RAG 중 어떤 모드를 실행할까요?"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 마지막 토큰의 다음 단어 확률(Logits) 추출
        next_token_logits = outputs.logits[0, -1, :]
        
        # TEXT와 VISION의 확률값 비교
        text_score = next_token_logits[text_token_id].item()
        vision_score = next_token_logits[vision_token_id].item()
        
    print(f"💬 질문: {query}")
    print(f"📊 TEXT 점수: {text_score:.4f} | VISION 점수: {vision_score:.4f}")
    print(f"🚀 결과: {'✅ VISION' if vision_score > text_score else '❌ TEXT'}")

# 명백한 비전 질문 테스트
debug_routing_logic("호서대 아산캠퍼스 셔틀버스 시간표 이미지 좀 보여줘.")