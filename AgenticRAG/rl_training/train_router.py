import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# ==========================================
# 1. 인증 및 기본 설정
# ==========================================
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

model_id = "google/gemma-2-2b-it"
save_path = "hoseo_router_gemma_2b"
dataset_path = "AgenticRAG/rl_traning/dpo_dataset.jsonl" 

print(f"🔬 논문용 연구 환경 세팅 중... (Model: {model_id})")

# ==========================================
# 2. 토크나이저 및 모델 로드
# ==========================================
print("📦 토크나이저 및 모델 로드 중 (16-bit fp16)...")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16, 
    token=hf_token
)

# 그래디언트 계산 누락 방지 (Gradients will be None 해결)
model.enable_input_require_grads()

# ==========================================
# 3. 표준 LoRA 설정
# ==========================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print("✅ 표준 LoRA 어댑터 장착 완료!")

# ==========================================
# 4. 데이터셋 로드 및 8:1:1 분할
# ==========================================
full_dataset = load_dataset("json", data_files=dataset_path, split="train")

train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

train_dataset = train_testvalid['train']
eval_dataset = test_valid['train']
test_dataset = test_valid['test']

print(f"📊 데이터 분할 완료 - Train: {len(train_dataset)}개 | Eval: {len(eval_dataset)}개 | Test: {len(test_dataset)}개")
test_dataset.to_json("test_dataset.jsonl")

# ==========================================
# 5. DPO 학습 설정 (🚀 윈도우 프리징 방지 추가)
# ==========================================
training_args = DPOConfig(
    output_dir="./temp_router_checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=10,
    logging_steps=1,                 # 매 스텝마다 실시간 로그 출력
    report_to="none",                # 외부 로깅 끔
    save_strategy="no",
    fp16=True,
    optim="adamw_torch",
    seed=42,
    remove_unused_columns=False,
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    
    # ✨ [핵심 추가] 윈도우 데이터로더 무한 로딩 방지!
    dataloader_num_workers=0,      # 멀티프로세싱 충돌 방지 (메인 프로세스에서만 처리)
    dataloader_pin_memory=False,   # pin_memory 데드락 방지
    gradient_checkpointing=False,  # 4090은 메모리가 넉넉하므로 속도를 위해 끔
)

# ==========================================
# 6. 트레이너 초기화 및 학습 시작
# ==========================================
dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print("\n🔥 논문 품질의 DPO 파인튜닝을 시작합니다! (실시간 로그 활성화)")
dpo_trainer.train()

# ==========================================
# 7. 학습된 모델(가중치) 저장
# ==========================================
print(f"\n💾 학습 완료! 모델 가중치를 '{save_path}' 폴더에 저장합니다...")
dpo_trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("🎉 연구용 라우터 모델 학습 및 저장이 완벽하게 종료되었습니다!")