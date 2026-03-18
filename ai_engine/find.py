import json
import os

def estimate_correction_cost(json_filepath):
    if not os.path.exists(json_filepath):
        print(f"❌ 파일을 찾을 수 없습니다: {json_filepath}")
        return

    with open(json_filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    total_chunks = len(chunks)
    total_characters = 0

    for chunk in chunks:
        total_characters += len(chunk.get("text", ""))

    # 한국어의 경우 보통 1글자가 1~1.5 토큰 정도로 계산됩니다. 
    # 보수적으로(비용을 넉넉하게 잡기 위해) 1글자 = 1.5 토큰으로 가정합니다.
    estimated_input_tokens = total_characters * 1.5
    
    # 프롬프트 토큰 추가 (청크당 약 50토큰: "다음 텍스트의 띄어쓰기와 줄바꿈만 교정해줘:")
    estimated_input_tokens += (total_chunks * 50)
    
    # 출력 토큰은 입력 텍스트와 거의 비슷하거나 줄바꿈이 사라져서 약간 줄어듭니다.
    estimated_output_tokens = total_characters * 1.5

    # 💸 GPT-4o-mini API 단가 (2024년 기준, 1M 토큰당 가격)
    # Input: $0.150 / 1M tokens
    # Output: $0.600 / 1M tokens
    input_cost_usd = (estimated_input_tokens / 1_000_000) * 0.150
    output_cost_usd = (estimated_output_tokens / 1_000_000) * 0.600
    total_cost_usd = input_cost_usd + output_cost_usd

    # 환율 적용 (약 1,350원)
    total_cost_krw = total_cost_usd * 1350

    print("=" * 40)
    print("📊 [LLM 띄어쓰기 교정 비용 예측]")
    print("=" * 40)
    print(f"총 청크(Chunk) 개수 : {total_chunks:,} 개")
    print(f"총 텍스트 글자 수   : {total_characters:,} 자")
    print("-" * 40)
    print(f"예상 Input 토큰     : 약 {int(estimated_input_tokens):,} Tokens")
    print(f"예상 Output 토큰    : 약 {int(estimated_output_tokens):,} Tokens")
    print("-" * 40)
    print(f"💵 예상 총 비용     : ${total_cost_usd:.4f} (약 {int(total_cost_krw):,} 원)")
    print("=" * 40)
    print("💡 기준 모델: GPT-4o-mini (가장 빠르고 저렴한 교정용 모델)")

if __name__ == "__main__":
    # 경로 설정 (현재 환경에 맞게 수정)
    base_path = os.getcwd()
    JSON_PATH = os.path.join(base_path, "data", "rules_regulations", "chunks", "all_rules_chunks.json")
    
    estimate_correction_cost(JSON_PATH)