import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class HoseoLoader:
    def __init__(self, raw_path="data/raw", processed_path="data/processed"):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.text_save_path = os.path.join(processed_path, "text")
        self.image_save_path = os.path.join(processed_path, "image")
        
        # 폴더 자동 생성
        os.makedirs(self.text_save_path, exist_ok=True)
        os.makedirs(self.image_save_path, exist_ok=True)

    def load_raw_data(self):
        """raw 폴더의 모든 info.json 데이터를 수집합니다"""
        data_list = []
        if not os.path.exists(self.raw_path): return []
        folders = [f for f in os.listdir(self.raw_path) if os.path.isdir(os.path.join(self.raw_path, f))]
        for n_id in folders:
            json_file = os.path.join(self.raw_path, n_id, "info.json")
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        data_list.append(json.load(f))
                    except: continue
        return data_list

    def run_text_summary(self):
        """[Mode 1] sLM 요약 (4090 GPU 가속 강제 설정)"""
        print("\n🚀 [Text Mode] 모델 로딩 중... (4090 GPU 가속)")
        
        # 다운로드 완료된 모델 ID
        model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B" 
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # 4090 핵심 설정
            device_map={"": 0},         # GPU 0번에 강제 할당
            low_cpu_mem_usage=True
        )
        
        # pipeline에서 device=0을 명시해야 CPU로 빠지지 않습니다
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            device=0 
        )

        raw_data = self.load_raw_data()
        print(f"📊 총 {len(raw_data)}건의 공지사항 처리 시작...")

        for item in tqdm(raw_data, desc="요약 진행 중"):
            n_id = item['notice_id']
            save_file = os.path.join(self.text_save_path, f"{n_id}.json")
            
            if os.path.exists(save_file): continue

            messages = [
                {"role": "system", "content": "너는 호서대 공지 요약 전문가야. 핵심 정보를 [대상, 일시, 장소, 신청방법] 위주로 아주 간결하게 요약해줘."},
                {"role": "user", "content": f"제목: {item['title']}\n내용: {item['content'][:1500]}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 4090에서는 이 과정이 매우 빠르게 진행됩니다
            outputs = pipe(
                prompt, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.2, 
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            summary_text = outputs[0]["generated_text"][len(prompt):]

            with open(save_file, "w", encoding="utf-8") as f:
                json.dump({"notice_id": n_id, "summary": summary_text}, f, ensure_ascii=False, indent=4)

    def run_vision_embedding(self):
        """[Mode 2] 이미지 벡터화 (ColPali)"""
        print("\n🖼️ [Vision Mode] 준비 중...")
        # 텍스트 요약이 끝난 후 이어서 구현할 부분입니다

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "vision"], default="text")
    args = parser.parse_args()

    loader = HoseoLoader()
    if args.mode == "text":
        loader.run_text_summary()
    elif args.mode == "vision":
        loader.run_vision_embedding()