import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ==========================================
# ⚙️ 1. 설정 (Path & Model)
# ==========================================
TRAIN_PATH = "evaluation/datasets/sft_splits/train.jsonl"
VAL_PATH = "evaluation/datasets/sft_splits/val.jsonl"

# 💡 대망의 명시적 의도(Intent) 기반 최종 V4 라우터 모델!
OUTPUT_DIR = "experience/exp1/gemma_router_lora_v4"          
MODEL_ID = "google/gemma-2b-it"                           

def main():
    print("🚀 [Step 1] Train / Valid 데이터 로드 중... (완벽한 5:5 황금 밸런스 V4)")
    
    train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")
    val_dataset = load_dataset("json", data_files=VAL_PATH, split="train")
    
    print(f"📥 Train 데이터 수: {len(train_dataset)}개")
    print(f"📥 Valid 데이터 수: {len(val_dataset)}개")

    # 2. Gemma 공식 프롬프트 포맷 적용
    def generate_prompt(example):
        text = f"<bos><start_of_turn>user\n{example['instruction']}\n{example['input']}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
        return {"text": text}
        
    train_dataset = train_dataset.map(generate_prompt)
    val_dataset = val_dataset.map(generate_prompt)
    print("✅ 데이터 프롬프트 포맷팅 완료!")

    print("\n🧠 [Step 2] Gemma-2B 모델 & 토크나이저 로드 (BF16 모드)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("\n💉 [Step 3] LoRA (가중치 미세조정) 설정 적용...")
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() 

    print("\n🏃‍♂️ [Step 4] 본격적인 [최종 V4] SFT 학습 시작! (VRAM 다이어트 모드)")
    
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        
        # 💡 [VRAM 다이어트 1: 배치 사이즈 반토막]
        # 한 번에 GPU에 올리는 데이터(배치)를 4 -> 2로 줄입니다. 
        # 대신 누적(accumulation)을 4 -> 8로 늘려서 전체 학습 효과(실질 배치 16)는 똑같이 유지합니다!
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        
        # 💡 [VRAM 다이어트 2: 그래디언트 체크포인팅] (가장 중요★)
        # 메모리에 다 올려두지 않고 필요할 때마다 다시 계산하는 기술입니다. 
        # 학습 속도는 살짝(10~20%) 느려지지만 VRAM 사용량을 기적처럼 반토막 냅니다.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, # 최신 PyTorch 안정화 옵션
        
        # 💡 [VRAM 다이어트 3: 8비트 옵티마이저]
        # 옵티마이저가 잡아먹는 메모리를 획기적으로 줄여줍니다. 
        # (만약 bitsandbytes 모듈 없다고 에러 나면 원래 쓰던 "adamw_torch"로 돌려주세요!)
        optim="adamw_torch", 

        save_steps=100,
        logging_steps=10,
        eval_strategy="steps",       
        eval_steps=50,               
        learning_rate=5e-5,          
        max_steps=300,               
        warmup_ratio=0.1,            
        lr_scheduler_type="cosine",  
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        report_to="none",
        max_seq_length=512,          
        dataset_text_field="text",   
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,    
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,  
    )

    # 🚀 모델 굽기 시작!
    trainer.train()

    print("\n💾 [Step 5] 학습 완료! V4 LoRA 어댑터 저장 중...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("="*60)
    print(f"🎉 성공! 의도 기반 라우팅을 마스터한 V4 모델이 '{OUTPUT_DIR}'에 저장되었습니다.")
    print("="*60)

if __name__ == "__main__":
    main()