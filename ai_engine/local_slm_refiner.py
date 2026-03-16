import os
import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalSLMRefiner:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"🚀 로컬 sLM 모델 로드 중... ({model_id})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ 모델 로드 완료!\n")

        self.system_prompt = """너는 대학교 공지사항 데이터를 RAG 시스템에 맞게 최적화하는 데이터 정제 및 메타데이터 추출 전문가다.

입력된 텍스트는 본문, 이미지 OCR, 첨부파일 텍스트가 섞여 있어 중복과 노이즈가 많다.

[작업 지침]

1. 정보 무손실 통합
- 일시, 장소, 지원 자격, 전화번호, 이메일, 제출 서류 등은 반드시 유지
- 중복되는 내용은 하나의 문장이나 리스트로 통합
- 노이즈(확장자, 깨진 문자 등)는 제거

2. 메타데이터 추출
- year
- category
- target
- entity
- status

[출력 형식]

반드시 아래 JSON 형식으로만 출력

{
  "metadata": {
    "year": "",
    "category": "",
    "target": "",
    "entity": "",
    "status": ""
  },
  "refined_content": ""
}
"""

    def extract_json_from_response(self, text):
        """
        LLM 응답에서 JSON만 안전하게 추출
        """

        text = text.strip()

        # ```json 블록 제거
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)

        # JSON 영역 탐색
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            json_str = match.group()

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("⚠ JSON 파싱 실패")
                return None

        print("⚠ JSON을 찾을 수 없음")
        return None

    def refine(self, raw_text):

        prompt = f"""{self.system_prompt}

다음 텍스트를 정제하라.

{raw_text}
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                temperature=0.1,
                do_sample=False
            )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        result = self.extract_json_from_response(response)

        torch.cuda.empty_cache()

        return result


def process_directory(input_dir, output_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    refiner = LocalSLMRefiner()

    files = list(input_dir.glob("*.json"))

    print(f"총 {len(files)}개 파일 처리 시작\n")

    for i, file in enumerate(files):

        print(f"[{i+1}/{len(files)}] 처리 중: {file.name}")

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_text = json.dumps(data, ensure_ascii=False)

        result = refiner.refine(raw_text)

        if result:

            output_path = output_dir / file.name

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        torch.cuda.empty_cache()

    print("\n✅ 전체 처리 완료")


if __name__ == "__main__":

    INPUT_DIR = "./raw_notice_data"
    OUTPUT_DIR = "./refined_notice_data"

    process_directory(INPUT_DIR, OUTPUT_DIR)