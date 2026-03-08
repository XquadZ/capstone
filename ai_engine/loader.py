import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ColPali 전용 라이브러리 임포트
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
except ImportError:
    print("❌ 'colpali-engine'이 설치되지 않았습니다. 'pip install colpali-engine'을 실행하세요.")

class HoseoLoader:
    def __init__(self, raw_path="data/raw", processed_path="data/processed"):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.text_save_path = os.path.join(processed_path, "text")
        self.image_save_path = os.path.join(processed_path, "image")
        
        # 출력 폴더 자동 생성
        os.makedirs(self.text_save_path, exist_ok=True)
        os.makedirs(self.image_save_path, exist_ok=True)

    def load_raw_data(self):
        """raw 폴더의 모든 info.json 데이터를 수집합니다."""
        data_list = []
        if not os.path.exists(self.raw_path): return []
        folders = [f for f in os.listdir(self.raw_path) if os.path.isdir(os.path.join(self.raw_path, f))]
        for n_id in folders:
            json_file = os.path.join(self.raw_path, n_id, "info.json")
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as f:
                    try: data_list.append(json.load(f))
                    except: continue
        return data_list

    def run_text_summary(self):
        """[Mode 1] sLM 요약 (RTX 4090 GPU 가속)"""
        print("\n🚀 [Text Mode] Llama-3 한국어 모델 로딩 중...")
        model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B" 
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        raw_data = self.load_raw_data()
        print(f"📊 총 {len(raw_data)}건의 텍스트 요약 시작...")

        for item in tqdm(raw_data, desc="요약 진행 중"):
            n_id = item['notice_id']
            save_file = os.path.join(self.text_save_path, f"{n_id}.json")
            if os.path.exists(save_file): continue

            messages = [
                {"role": "system", "content": "너는 호서대 공지 요약 전문가야. [대상, 일시, 장소, 신청방법] 위주로 아주 간결하게 요약해줘."},
                {"role": "user", "content": f"제목: {item['title']}\n내용: {item['content'][:1500]}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
            
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump({"notice_id": n_id, "summary": outputs[0]["generated_text"][len(prompt):].strip()}, f, ensure_ascii=False, indent=4)

    def run_vision_embedding(self):
        """[Mode 2] 이미지 벡터화 (ColPali v1.2 최적화)"""
        print("\n🖼️ [Vision Mode] ColPali 모델 로딩 중... (GPU 가속)")
        
        model_name = "vidore/colpali-v1.2" 
        
        # 모델과 프로세서 로드 (ValueError 방지를 위해 직접 지정)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        ).eval()
        
        processor = ColPaliProcessor.from_pretrained(model_name)

        folders = [f for f in os.listdir(self.raw_path) if os.path.isdir(os.path.join(self.raw_path, f))]
        print(f"📊 이미지 데이터 스캔 시작...")

        for n_id in tqdm(folders, desc="이미지 벡터화 중"):
            img_dir = os.path.join(self.raw_path, n_id, "images")
            if not os.path.exists(img_dir): continue
            
            image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in image_files:
                img_path = os.path.join(img_dir, img_name)
                # 저장 파일명: notice_id_이미지파일명.pt
                save_file = os.path.join(self.image_save_path, f"{n_id}_{img_name}.pt")
                
                if os.path.exists(save_file): continue

                try:
                    image = Image.open(img_path).convert("RGB")
                    
                    with torch.no_grad():
                        # ColPali 전용 프로세싱
                        batch_images = processor.process_images([image]).to("cuda")
                        # 4090 GPU를 이용한 멀티-벡터 임베딩 추출
                        embeddings = model(**batch_images)
                        
                    # 결과 벡터를 CPU로 옮겨 저장하여 VRAM 확보
                    torch.save(embeddings.cpu(), save_file)
                except Exception as e:
                    print(f"❌ {img_name} 처리 실패: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "vision"], default="text")
    args = parser.parse_args()

    loader = HoseoLoader()
    if args.mode == "text":
        loader.run_text_summary()
    elif args.mode == "vision":
        loader.run_vision_embedding()