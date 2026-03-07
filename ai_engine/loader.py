import os
import json
import torch
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from PIL import Image
from colpali_engine.models import ColQwen2_5 # ColPali 최신 엔진
from colpali_engine.utils.processing_utils import process_images
from colpali_engine.utils.torch_utils import get_torch_device

class HoseoLoader:
    def __init__(self, raw_path="data/raw", processed_path="data/processed"):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.text_save_path = os.path.join(processed_path, "text")
        self.image_save_path = os.path.join(processed_path, "image")
        
        os.makedirs(self.text_save_path, exist_ok=True)
        os.makedirs(self.image_save_path, exist_ok=True)
        self.device = get_torch_device()

    def load_raw_data(self):
        """Raw 데이터를 수집합니다"""
        data_list = []
        folders = [f for f in os.listdir(self.raw_path) if os.path.isdir(os.path.join(self.raw_path, f))]
        for n_id in folders:
            json_path = os.path.join(self.raw_path, n_id, "info.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data_list.append(json.load(f))
        return data_list

    def run_text_summary(self):
        """[Mode 1] sLM 요약 (vLLM 활용)"""
        print("\n🚀 [Text Mode] sLM(Llama-3.3-8B) 가동...")
        model_name = "Llama-3.3-8B-Korean-Instruct" 
        llm = LLM(model=model_name, gpu_memory_utilization=0.7) # 4090 VRAM 70% 점유
        sampling_params = SamplingParams(temperature=0.2, max_tokens=600)

        raw_data = self.load_raw_data()
        prompts, target_ids = [], []

        for item in raw_data:
            if os.path.exists(os.path.join(self.text_save_path, f"{item['notice_id']}.json")): continue
            prompt = f"<|im_start|>system\n공지사항 요약 전문가야. [대상, 일시, 장소, 방법] 위주로 요약해.<|im_end|>\n<|im_start|>user\n제목: {item['title']}\n본문: {item['content'][:1500]}\n요약해줘:<|im_end|>\n<|im_start|>assistant"
            prompts.append(prompt)
            target_ids.append(item['notice_id'])

        if prompts:
            outputs = llm.generate(prompts, sampling_params)
            for n_id, output in zip(target_ids, outputs):
                with open(os.path.join(self.text_save_path, f"{n_id}.json"), "w", encoding="utf-8") as f:
                    json.dump({"notice_id": n_id, "summary": output.outputs[0].text}, f, ensure_ascii=False, indent=4)
        print("✨ 텍스트 요약 완료.")

    def run_vision_embedding(self):
        """[Mode 2] 이미지 벡터화 (ColPali 활용)"""
        print("\n🖼️ [Vision Mode] ColPali(ColQwen2.5) 가동...")
        
        # 모델 로드 (비전 모드 실행 시 VRAM을 최대로 쓰기 위해 텍스트 모델은 종료되어야 함)
        model = ColQwen2_5.from_pretrained("vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16).to(self.device)
        processor = model.get_processor()

        folders = [f for f in os.listdir(self.raw_path) if os.path.isdir(os.path.join(self.raw_path, f))]
        
        for n_id in tqdm(folders, desc="이미지 벡터화 중"):
            img_dir = os.path.join(self.raw_path, n_id, "images")
            if not os.path.exists(img_dir): continue
            
            images = []
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
            
            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                images.append(Image.open(img_path).convert("RGB"))

            if images:
                # ColPali 임베딩 생성 (이미지 패치를 직접 벡터화)
                batch_images = process_images(processor, images).to(self.device)
                with torch.no_grad():
                    image_embeddings = model.get_image_embeddings(batch_images)
                
                # 벡터 데이터 저장 (추후 Elasticsearch로 전송될 원천 데이터)
                save_data = {
                    "notice_id": n_id,
                    "embeddings": image_embeddings.cpu().tolist() # 리스트로 변환하여 저장
                }
                with open(os.path.join(self.image_save_path, f"{n_id}_vision.json"), "w") as f:
                    json.dump(save_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "vision"], required=True)
    args = parser.parse_args()

    loader = HoseoLoader()
    if args.mode == "text":
        loader.run_text_summary()
    elif args.mode == "vision":
        loader.run_vision_embedding()