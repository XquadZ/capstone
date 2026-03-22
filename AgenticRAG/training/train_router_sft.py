import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. 환경 및 경로 설정
# ==========================================
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
model_id = "google/gemma-2-2b-it"
save_path = "hoseo_router_gemma_2b_sft" # SFT 버전 저장 경로
dataset_path = "AgenticRAG/training/sft_dataset.jsonl"

print("🔬 논문용 라우터 SFT(지도학습) 세팅 중...")

# ==========================================
# 2. 토크나이저 및 모델 로드
# ==========================================
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

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ==========================================
# 3. 데이터 로드 및 100% 안전한 템플릿 매핑
# ==========================================
full_dataset = load_dataset("json", data_files=dataset_path, split="train")

# ✨ SFTTrainer의 버그를 피하기 위해, 학습 전 미리 텍스트를 완성해둡니다.
def apply_template(row):
    # row["messages"]는 [{"role": "user", ...}, {"role": "model", ...}] 형태
    row["text"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return row

print("⚙️ 데이터셋에 챗 템플릿(포장지)을 미리 씌우는 중...")
full_dataset = full_dataset.map(apply_template, num_proc=1)

# 학습/검증/테스트 세트 분할
train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

# 나중에 평가할 테스트 데이터 저장 (SFT 버전)
test_valid['test'].to_json("test_dataset_sft.jsonl")

# ==========================================
# 4. SFT 학습 설정 및 실행
# ==========================================
training_args = SFTConfig(
    output_dir="./temp_sft_checkpoints",
    per_device_train_batch_size=1,   # ✨ [핵심] 2 -> 1로 낮춤 (VRAM 직격탄)
    gradient_accumulation_steps=8,  # ✨ [보완] 배치를 낮춘 만큼 쌓아서 계산 (학습 퀄리티 유지)
    learning_rate=2e-4, 
    lr_scheduler_type="cosine",
    num_train_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    fp16=True,                      # 이미 적용 중
    optim="adamw_torch",
    seed=42,
    max_seq_length=256,             # ✨ [절약] 512 -> 256 (라우터는 질문이 짧아서 256이면 충분!)
    dataset_text_field="text",
)

sft_trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_testvalid['train'],
    eval_dataset=test_valid['train'],
    # formatting_func는 맵핑을 했으므로 생략! (에러 원천 차단)
)

print("\n🔥 SFT: 딴생각 금지! 무지성 주입식 파인튜닝을 시작합니다!")
sft_trainer.train()

# ==========================================
# 5. 저장
# ==========================================
print(f"\n💾 학습 완료! 최종 모델을 '{save_path}' 폴더에 저장합니다...")
sft_trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("🎉 SFT 모델 학습 완벽 종료!")