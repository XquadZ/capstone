"""로컬 Qwen3-VL 공유 인스턴스 (TEXT / VISION 공통 생성기)."""

from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any, Dict, List, Optional

import torch

_DEFAULT_LOCAL_PATH = r"D:\models\Qwen3-VL-2B-Instruct"
_DEFAULT_HUB_ID = "Qwen/Qwen3-VL-2B-Instruct"


def _default_model_id() -> str:
    env = (os.getenv("LOCAL_VLM_MODEL_ID") or "").strip()
    if env:
        return env
    if os.path.isdir(_DEFAULT_LOCAL_PATH):
        return _DEFAULT_LOCAL_PATH
    return _DEFAULT_HUB_ID


DEFAULT_MAX_NEW_TOKENS = int(os.getenv("LOCAL_VLM_MAX_NEW_TOKENS", "1500"))

_lock = threading.Lock()
_model = None
_processor = None
_model_id: Optional[str] = None


def _decode_data_url_to_pil(data_url: str):
    from PIL import Image

    raw = data_url
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    img_bytes = base64.b64decode(raw)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def openai_image_parts_to_pil(image_contents: List[dict]) -> List[Any]:
    """OpenAI vision content 파트([{type:image_url,...}]) → PIL 리스트."""
    images = []
    for part in image_contents or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "image_url":
            continue
        url = (part.get("image_url") or {}).get("url") or ""
        if not url:
            continue
        try:
            if url.startswith("data:"):
                images.append(_decode_data_url_to_pil(url))
            elif os.path.isfile(url):
                from PIL import Image

                images.append(Image.open(url).convert("RGB"))
        except Exception as e:
            print(f"[LocalVLM] image decode failed: {e}")
    return images


def get_shared_local_vlm(model_id: Optional[str] = None):
    """프로세스당 1회 로드. (model, processor) 반환."""
    global _model, _processor, _model_id
    mid = (model_id or _default_model_id()).strip()
    with _lock:
        if _model is not None and _processor is not None and _model_id == mid:
            return _model, _processor

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"[LocalVLM] loading: {mid}")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            mid,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        _model.eval()
        _model_id = mid
        print(f"[LocalVLM] loaded: {mid}")
        return _model, _processor


def warm_local_vlm(model_id: Optional[str] = None) -> None:
    """서버 부팅 시 VRAM 사전 적재 + 더미 1회 생성."""
    model, processor = get_shared_local_vlm(model_id)
    _ = generate_with_local_vlm(
        system_prompt="당신은 간단한 테스트 응답기입니다.",
        user_text="ping",
        images=None,
        max_new_tokens=8,
        model=model,
        processor=processor,
    )


@torch.inference_mode()
def generate_with_local_vlm(
    *,
    system_prompt: str = "",
    user_text: str = "",
    images: Optional[List[Any]] = None,
    max_new_tokens: Optional[int] = None,
    model=None,
    processor=None,
) -> str:
    """텍스트(+선택 이미지)로 로컬 VLM 답변 생성."""
    if model is None or processor is None:
        model, processor = get_shared_local_vlm()

    content: List[Dict[str, Any]] = []
    for img in images or []:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text or ""})

    messages: List[Dict[str, Any]] = []
    if (system_prompt or "").strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}],
            }
        )
    messages.append({"role": "user", "content": content})

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # device_map=auto 시 입력은 모델 첫 디바이스로
    try:
        first_param = next(model.parameters())
        inputs = {k: v.to(first_param.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    except StopIteration:
        pass

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens or DEFAULT_MAX_NEW_TOKENS),
        "do_sample": False,
    }
    generated = model.generate(**inputs, **gen_kwargs)
    # prompt 길이만큼 잘라 응답만 디코드
    in_len = inputs["input_ids"].shape[-1]
    trimmed = generated[:, in_len:]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (text[0] if text else "").strip()


def generate_text(
    system_prompt: str,
    user_text: str,
    max_new_tokens: int = 1500,
) -> str:
    return generate_with_local_vlm(
        system_prompt=system_prompt,
        user_text=user_text,
        images=None,
        max_new_tokens=max_new_tokens,
    )


def generate_vision(
    system_prompt: str,
    user_text: str,
    image_contents: Optional[List[dict]] = None,
    max_new_tokens: int = 2500,
) -> str:
    images = openai_image_parts_to_pil(image_contents or [])
    return generate_with_local_vlm(
        system_prompt=system_prompt,
        user_text=user_text,
        images=images,
        max_new_tokens=max_new_tokens,
    )
