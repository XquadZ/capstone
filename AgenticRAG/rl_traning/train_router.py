import os
import torch
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel

# ==========================================
# ⚙️ 1. 기본 설정 및 VRAM 최적화 로드
# ==========================================
max_seq_length = 1024 # 라우터는 긴 문맥이 필요 없으므로 메모리 절약을 위해 제한
dtype = None # Unsloth가 하드웨어(4090)에 맞춰 자동 설정 (보통 bfloat16)
load_in_4bit = True # 🔥 핵심: 4-bit 양자화로 로드하여 VRAM 사용량 극소화 (약 2GB 소모)

print("🧠 베이스 모델(Gemma-2-2B-it)을 4-bit 양자화 상태로 로드 중...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-2b-it",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ==========================================
# 🧩 2. LoRA 어댑터 설정 (기존 지식 보호 + 라우팅 능력 학습)
# ==========================================
print("🔧 LoRA 어댑터 장착 중...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank 크기 (높을수록 똑똑해지나 메모리 차지, 16이 적당)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth 최적화를 위해 0 사용
    bias = "none",
    use_gradient_checkpointing = "unsloth", # VRAM 폭발 방지
    random_state = 42,
    use_rslora = False,
)

# ==========================================
# 📝 3. DPO 데이터셋 로드 및 전처리 (Gemma 포맷팅)
# ==========================================
dataset_path = "dpo_dataset.jsonl"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ 데이터셋을 찾을 수 없습니다: {dataset_path}")

raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

def format_dpo_gemma(sample):
    """
    Gemma-2의 프롬프트 템플릿(<bos>, <start_of_turn> 등)에 맞게 데이터 포맷팅
    TRL DPOTrainer는 'prompt', 'chosen', 'rejected' 키를 요구합니다.
    """
    system_instruction = "당신은 호서대 학칙 AI의 지능형 라우터입니다. 사용자의 질문을 분석하여 텍스트 검색(TEXT)과 이미지/표 검색(VISION) 중 적절한 경로를 단답으로 출력하세요."
    
    # Gemma Prompt 형식: <bos><start_of_turn>user\n[내용]<end_of_turn>\n<start_of_turn>model\n
    prompt = f"<bos><start_of_turn>user\n{system_instruction}\n\n{sample['prompt']}<end_of_turn>\n<start_of_turn>model\n"
    
    # 정답과 오답에 종료 토큰 명시
    chosen = f"{sample['chosen']}<end_of_turn><eos>"
    rejected = f"{sample['rejected']}<end_of_turn><eos>"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

print(f"📊 {len(raw_dataset)}개의 데이터를 DPO 형식으로 변환 중...")
dpo_dataset = raw_dataset.map(format_dpo_gemma)

# ==========================================
# 🏋️‍♂️ 4. DPO 트레이너 설정 및 학습 시작
# ==========================================
training_args = DPOConfig(
    output_dir="router_gemma_checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200, # 600개 데이터 기준 약 1~2 에폭 분량 (과적합 방지)
    logging_steps=10,
    optim="adamw_8bit", # 8bit 옵티마이저로 VRAM 추가 절약
    warmup_ratio=0.1,
    bf16=True, # RTX 4090은 bfloat16을 완벽 지원
    beta=0.1, # DPO 패널티 파라미터 (0.1이 국룰)
    max_prompt_length=512,
    max_length=1024,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None, # Unsloth에서는 ref_model을 None으로 둬도 내부적으로 자동 처리함
    args=training_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)

print("\n🔥 DPO 파인튜닝 시작! (4090 기준 약 10~20분 소요 예정)\n" + "="*50)
trainer.train()

# ==========================================
# 💾 5. 학습된 라우터 모델(LoRA 가중치) 저장
# ==========================================
save_path = "hoseo_router_gemma_2b"
print(f"\n💾 학습 완료! 모델 가중치를 '{save_path}' 폴더에 저장합니다...")

# LoRA 어댑터만 저장 (매우 가볍고 빠름)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("🎉 모든 과정이 성공적으로 완료되었습니다!")