import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer

# ==========================================
# 1. 기본 설정
# ==========================================
model_id = "google/gemma-2-2b-it"
save_path = "hoseo_router_gemma_2b"
dataset_path = "dpo_dataset.jsonl"

print(f"🔬 논문용 연구 환경 세팅 중... (Model: {model_id})")

# ==========================================
# 2. 토크나이저 및 모델 로드 (양자화 제거 -> 순수 bfloat16)
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit 양자화(BitsAndBytes)를 빼고, 4090의 네이티브 16-bit(bfloat16)로 로드합니다.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
)

# ==========================================
# 3. 표준 LoRA 설정 (논문 방법론 기재용)
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
print("✅ 표준 LoRA 어댑터 장착 완료! (양자화 없음)")

# ==========================================
# 4. 데이터셋 로드 및 Train/Eval 분리 (논문 디펜스용 핵심!!!)
# ==========================================
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 전체 데이터를 90%는 학습용, 10%는 검증용(Overfitting 확인)으로 나눕니다.
# 나중에 논문에 "Train/Val split ratio 9:1" 이라고 적으시면 됩니다.
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"📊 데이터 분할 완료 - Train: {len(train_dataset)}개, Eval: {len(eval_dataset)}개")

# ==========================================
# 5. DPO 학습 설정 (재현성 및 로깅 강화)
# ==========================================
training_args = TrainingArguments(
    output_dir="./temp_router_checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=3,              # max_steps 대신 Epoch(3회독) 사용 (논문 표준)
    evaluation_strategy="steps",     # 평가(Eval) 수행 방식 지정
    eval_steps=10,                   # 10 스텝마다 검증셋으로 Loss 측정
    logging_steps=10,                # 10 스텝마다 로그 기록
    save_strategy="no",
    bf16=True,                       # RTX 4090 가속
    optim="adamw_torch",             # 8bit 옵티마이저 대신 순정 AdamW 사용
    seed=42,                         # ✨ 재현성을 위한 시드 고정
    remove_unused_columns=False,
)

# 논문 서술 포인트: "We set the KL penalty coefficient (beta) to 0.1..."
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,       # 검증 데이터셋 추가
    tokenizer=tokenizer,
    max_length=512,
    max_prompt_length=256,
)

# ==========================================
# 6. 학습 시작 및 저장
# ==========================================
print("🔥 논문 품질의 DPO 파인튜닝을 시작합니다!")
dpo_trainer.train()

print(f"\n💾 학습 완료! 모델 가중치를 '{save_path}' 폴더에 저장합니다...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("🎉 연구용 모델 학습 및 저장이 완료되었습니다!")