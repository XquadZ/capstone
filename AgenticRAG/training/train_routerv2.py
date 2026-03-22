import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# 환경 설정
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
model_id = "google/gemma-2-2b-it"
save_path = "hoseo_router_gemma_2b_v2" # v2로 저장!
dataset_path = "AgenticRAG/rl_training/dpo_dataset_balanced_final.jsonl" # ✨ 밸런싱된 새 데이터!

print(f"🔬 논문용 라우터 V2 학습 세팅 중... (데이터 5:5 황금 밸런스)")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16, 
    token=hf_token
)
model.enable_input_require_grads()

# ✨ LoRA Rank 32로 상향 (더 복잡한 메타 인지 패턴 학습 가능)
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

full_dataset = load_dataset("json", data_files=dataset_path, split="train")
train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

# 나중에 평가할 테스트 데이터 저장
test_valid['test'].to_json("test_dataset_v2.jsonl")

training_args = DPOConfig(
    output_dir="./temp_router_checkpoints_v2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=4,              # ✨ 데이터가 2배로 늘었으니 4 에포크면 충분!
    eval_strategy="epoch",           # 에포크마다 평가
    save_strategy="epoch",           # 에포크마다 저장
    logging_steps=1,
    report_to="none",
    fp16=True,
    optim="adamw_torch",
    seed=42,
    beta=0.2,                        # ✨ 기존 지식을 덜 까먹도록 보수적으로 조정
    max_length=512,
    max_prompt_length=256,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    gradient_checkpointing=False,
)

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_testvalid['train'],
    eval_dataset=test_valid['train'],
    processing_class=tokenizer,
)

print("\n🔥 V2: 단호박 라우터 파인튜닝을 시작합니다!")
dpo_trainer.train()

print(f"\n💾 학습 완료! 최종 모델을 '{save_path}' 폴더에 저장합니다...")
dpo_trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("🎉 V2 모델 학습 완벽 종료!")