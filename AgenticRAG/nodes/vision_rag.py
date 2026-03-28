import os
import base64
import io
import time
from typing import List, Optional, Tuple

from openai import OpenAI
from pdf2image import convert_from_path
from AgenticRAG.graph.state import AgentState

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from ai_engine.rag_pipeline_notice import get_shared_notice_pipeline
except ImportError:
    print("⚠️ [Vision] ai_engine.rag_pipeline_notice 를 불러오지 못했습니다.")
    get_shared_notice_pipeline = None  # type: ignore


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

POPPLER_PATH = None

# TPM 절감: VLM 요청당 이미지 상한(기본 10) + OpenAI vision low-detail
MAX_VLM_IMAGES = int(os.environ.get("VLM_MAX_IMAGES", "10"))
VLM_IMAGE_DETAIL = os.environ.get("VLM_IMAGE_DETAIL", "low").strip().lower()
if VLM_IMAGE_DETAIL not in ("low", "high", "auto"):
    VLM_IMAGE_DETAIL = "low"

# PDF 래스터 해상도 (낮을수록 바이트↓; detail=low일 때도 업로드 크기에 영향)
PDF_RASTER_DPI = int(os.environ.get("VLM_PDF_DPI", "96"))

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
_PDF_EXT = ".pdf"


def _entity_to_dict(ent):
    if isinstance(ent, dict):
        return ent
    if hasattr(ent, "to_dict"):
        return ent.to_dict()
    return dict(ent)


def _notice_attachments_dir(doc_id: str) -> str:
    return os.path.join("data", "raw", str(doc_id), "attachments")


def _notice_images_dir(doc_id: str) -> str:
    return os.path.join("data", "raw", str(doc_id), "images")


def _safe_listdir(path: str) -> List[str]:
    try:
        if not path or not os.path.isdir(path):
            return []
        return sorted(os.listdir(path))
    except OSError as e:
        print(f"⚠️ [Vision] 디렉터리 목록 실패 (건너뜀): {path} | {e}")
        return []


def _collect_notice_media_paths(doc_id: str) -> List[str]:
    """
    크롤러 구조: data/raw/<notice_id>/attachments, data/raw/<notice_id>/images
    파일만 포함하며, 학칙(raw_pdfs) 경로는 사용하지 않습니다.
    """
    if not doc_id or doc_id == "unknown":
        return []

    paths: List[str] = []
    seen = set()

    for folder_fn in (_notice_attachments_dir, _notice_images_dir):
        base = folder_fn(doc_id)
        for name in _safe_listdir(base):
            low = name.lower()
            if not (low.endswith(_PDF_EXT) or low.endswith(_IMAGE_EXTS)):
                continue
            full = os.path.normpath(os.path.join(base, name))
            try:
                if not os.path.isfile(full):
                    continue
            except OSError:
                continue
            if full not in seen:
                seen.add(full)
                paths.append(full)

    return paths


def resolve_notice_attachment_file(doc_id: str, filename_hint: Optional[str]) -> Optional[str]:
    """메타데이터의 파일명이 있으면 attachments 내 실제 파일과 매칭합니다."""
    if not doc_id or doc_id == "unknown":
        return None
    if not filename_hint or not str(filename_hint).strip():
        return None

    fn = os.path.basename(str(filename_hint).strip())
    if not fn:
        return None

    cand = os.path.normpath(os.path.join(_notice_attachments_dir(doc_id), fn))
    try:
        if os.path.isfile(cand):
            return cand
    except OSError:
        pass
    return None


def _chunk_texts_from_hits(search_hits: list) -> List[str]:
    """RAGAS·벤치마크용: Milvus 검색 히트에서 본문 청크만 추출."""
    texts: List[str] = []
    for h in search_hits:
        try:
            ent = _entity_to_dict(h["entity"])
        except Exception:
            continue
        t = (ent.get("chunk_text") or "").strip()
        if t:
            texts.append(t)
    return texts


def _hits_to_text_context(search_hits: list) -> str:
    """첨부가 없을 때 VLM(텍스트 전용)에 넘길 검색 본문을 만듭니다."""
    parts = []
    for h in search_hits:
        try:
            ent = _entity_to_dict(h["entity"])
        except Exception:
            continue
        pid = ent.get("parent_id", "")
        cat = ent.get("category", "") or "일반"
        year = ent.get("year", "")
        body = (ent.get("chunk_text") or "").strip()
        if not body:
            continue
        header = f"[공지-{cat}] parent_id={pid}" + (f" ({year}년)" if year else "")
        parts.append(f"{header}\n{body}")

    if not parts:
        return "(검색된 공지 본문 없음)"
    return "\n\n---\n\n".join(parts)


def _pil_image_to_vlm_part(path: str) -> Optional[dict]:
    if not _HAS_PIL:
        print("⚠️ [Vision] PIL 없음 — 이미지 첨부 스킵")
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": VLM_IMAGE_DETAIL},
        }
    except Exception as e:
        print(f"⚠️ [Vision] 이미지 로드 실패 (건너뜀): {path} | {e}")
        return None


def _pdf_page_to_vlm_parts(
    pdf_path: str, page_nums: List[int], log_name: str
) -> Tuple[List[dict], List[str]]:
    parts: List[dict] = []
    logs: List[str] = []
    if not pdf_path or not pdf_path.lower().endswith(_PDF_EXT):
        return parts, logs

    try:
        if not os.path.isfile(pdf_path):
            return parts, logs
    except OSError:
        return parts, logs

    for p_num in page_nums:
        if len(parts) >= MAX_VLM_IMAGES:
            break
        if p_num < 1:
            continue
        try:
            pages = convert_from_path(
                pdf_path,
                dpi=PDF_RASTER_DPI,
                first_page=p_num,
                last_page=p_num,
                poppler_path=POPPLER_PATH,
            )
        except Exception as e:
            print(f"⚠️ [Vision] PDF 페이지 변환 실패 (건너뜀): {pdf_path} p.{p_num} | {e}")
            continue

        for page in pages:
            if len(parts) >= MAX_VLM_IMAGES:
                break
            try:
                buf = io.BytesIO()
                page.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": VLM_IMAGE_DETAIL},
                    }
                )
                logs.append(f"{log_name} | {os.path.basename(pdf_path)} (p.{p_num})")
            except Exception as e:
                print(f"⚠️ [Vision] PDF 페이지 JPEG 인코딩 실패: {e}")

    return parts, logs


def _build_image_contents_from_hits(search_hits: list) -> Tuple[List[dict], List[str]]:
    """검색 히트에 대응하는 공지 폴더의 PDF/이미지에서 VLM용 이미지 파트를 수집합니다."""
    image_contents: List[dict] = []
    processed_log: List[str] = []

    ordered_doc_ids: List[str] = []
    seen_ids = set()
    for h in search_hits:
        ent = _entity_to_dict(h["entity"])
        doc_id = str(ent.get("parent_id", "unknown"))
        if doc_id not in seen_ids and doc_id != "unknown":
            seen_ids.add(doc_id)
            ordered_doc_ids.append(doc_id)

    media_by_doc: dict = {}
    for doc_id in ordered_doc_ids:
        paths_set = set()
        hint_paths = []

        for h in search_hits:
            ent = _entity_to_dict(h["entity"])
            if str(ent.get("parent_id", "")) != doc_id:
                continue
            for key in ("source_path", "file_path", "attachment", "filename", "attach_name"):
                raw = ent.get(key)
                if raw:
                    resolved = resolve_notice_attachment_file(doc_id, raw)
                    if resolved:
                        hint_paths.append(resolved)

        for p in hint_paths:
            paths_set.add(p)
        for p in _collect_notice_media_paths(doc_id):
            paths_set.add(p)

        media_by_doc[doc_id] = sorted(paths_set, key=lambda x: (x.lower().endswith(_PDF_EXT), x))

    # PDF당 기본 페이지 수: 상한이 작으면 1페이지만, 10장 이하이면 1~2페이지, 그 이상 허용 시 3페이지까지
    if MAX_VLM_IMAGES <= 4:
        page_num_default = [1]
    elif MAX_VLM_IMAGES <= 10:
        page_num_default = [1, 2]
    else:
        page_num_default = [1, 2, 3]

    for doc_id in ordered_doc_ids:
        if len(image_contents) >= MAX_VLM_IMAGES:
            break
        cat = "일반"
        for h in search_hits:
            ent = _entity_to_dict(h["entity"])
            if str(ent.get("parent_id", "")) == doc_id:
                cat = ent.get("category", "") or cat
                break
        label = f"[공지-{cat}] {doc_id}"

        for full_path in media_by_doc.get(doc_id, []):
            if len(image_contents) >= MAX_VLM_IMAGES:
                break
            try:
                if not os.path.isfile(full_path):
                    continue
            except OSError:
                continue

            low = full_path.lower()
            if low.endswith(_PDF_EXT):
                page_override = None
                for h in search_hits:
                    ent = _entity_to_dict(h["entity"])
                    if str(ent.get("parent_id", "")) != doc_id:
                        continue
                    pn = int(ent.get("page_num", 0) or 0)
                    if pn > 0:
                        page_override = pn
                        break
                if page_override:
                    pages_to_try = sorted(
                        {page_override, max(1, page_override - 1), page_override + 1}
                    )
                else:
                    pages_to_try = page_num_default

                new_parts, new_logs = _pdf_page_to_vlm_parts(full_path, pages_to_try, label)
                image_contents.extend(new_parts)
                processed_log.extend(new_logs)
            elif low.endswith(_IMAGE_EXTS):
                part = _pil_image_to_vlm_part(full_path)
                if part:
                    image_contents.append(part)
                    processed_log.append(f"{label} | {os.path.basename(full_path)} (image)")

    return image_contents[:MAX_VLM_IMAGES], processed_log


def _call_vlm(
    question: str,
    image_contents: List[dict],
    text_context: str,
    attachment_status: str,
) -> str:
    if image_contents:
        intro = (
            "당신은 호서대학교 공지·첨부 문서(표, 이미지, 안내문) 분석 전문가입니다. "
            "아래 이미지와 [검색된 공지 본문]을 함께 참고하세요. 이미지와 본문이 다를 경우 본문을 보조 근거로 사용하세요.\n\n"
            "### 지시:\n"
            "1. 표·수치·기한·절차는 가능하면 이미지에 보이는 그대로 인용하세요.\n"
            "2. 확실하지 않은 내용은 추측하지 마세요.\n"
            "3. 답변 끝에 '📚 [분석 근거]' 섹션을 두고, 사용한 파일·페이지(또는 본문 출처)를 나열하세요.\n\n"
            f"[첨부 상태] {attachment_status}\n\n"
            f"[검색된 공지 본문]\n{text_context}\n\n"
            f"사용자 질문: {question}"
        )
        user_content: List[dict] = [{"type": "text", "text": intro}, *image_contents]
    else:
        intro = (
            "당신은 호서대학교 공지 안내 전문가입니다. "
            "첨부 PDF/이미지는 없거나 열 수 없어, 아래 [검색된 공지 본문]만 근거로 답하세요.\n\n"
            "### 지시:\n"
            "1. 본문에 근거해 답하세요. 없는 내용은 지어내지 마세요.\n"
            "2. 답변 끝에 '📚 [분석 근거]'에 인용한 공지 출처(parent_id·분류)를 요약해 적으세요.\n\n"
            f"[첨부 상태] {attachment_status}\n\n"
            f"[검색된 공지 본문]\n{text_context}\n\n"
            f"사용자 질문: {question}"
        )
        user_content = [{"type": "text", "text": intro}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_content}],
        max_tokens=2500,
        temperature=0.0,
    )
    return response.choices[0].message.content


def vision_rag_node(state: AgentState) -> dict:
    question = state["question"]
    start_time = time.time()
    print(f"\n--- [NODE: Vision RAG] 공지·첨부 시각 분석 (gpt-4o-mini, OpenAI) ---")

    if get_shared_notice_pipeline is None:
        return {
            "generation": "공지사항 검색 모듈을 불러오지 못했습니다.",
            "context": [],
            "retrieved_chunk_texts": [],
        }

    search_hits: list = []
    try:
        pipe = get_shared_notice_pipeline()
        search_hits = pipe.search_and_rerank(question, retrieve_k=50, final_k=25)
    except Exception as e:
        print(f"❌ [Vision] 공지 검색 실패: {e}")
        search_hits = []

    retrieved_chunk_texts = _chunk_texts_from_hits(search_hits)

    if not search_hits:
        return {
            "generation": "관련 공지를 찾지 못했습니다.",
            "context": [],
            "retrieved_chunk_texts": [],
        }

    text_context = _hits_to_text_context(search_hits)

    image_contents: List[dict] = []
    processed_pages_log: List[str] = []

    try:
        image_contents, processed_pages_log = _build_image_contents_from_hits(search_hits)
    except Exception as e:
        print(f"⚠️ [Vision] 첨부 미디어 수집 중 예외 (텍스트만 진행): {e}")
        image_contents = []
        processed_pages_log = []

    if image_contents:
        attachment_status = f"첨부 미디어 {len(image_contents)}건 로드됨"
        print(
            f"🚀 [VLM] 이미지 {len(image_contents)}장 (상한 {MAX_VLM_IMAGES}, detail={VLM_IMAGE_DETAIL}) + 본문 → OpenAI"
        )
    else:
        attachment_status = "첨부파일 없음 (또는 열 수 없음) — 검색된 공지 본문만 사용"
        print("ℹ️ [Vision] 첨부 PDF/이미지 없음 → 검색 텍스트만으로 VLM에 질의합니다.")

    try:
        generation = _call_vlm(question, image_contents, text_context, attachment_status)
        trace_title = (
            "**TV-RAG Traceability (Notice Vision):**\n"
            if image_contents
            else "**TV-RAG (텍스트 전용 폴백):**\n"
        )
        source_footer = f"\n\n📍 {trace_title}"
        if processed_pages_log:
            source_footer += "\n".join(f"- {log}" for log in processed_pages_log)
        else:
            source_footer += "- (이미지 로그 없음 — 본문 기반 응답)"
        generation += source_footer

        elapsed = time.time() - start_time
        print(f"✅ [Vision] 완료 (소요 {elapsed:.1f}초)")

    except Exception as e:
        print(f"❌ [VLM API Error] {e}")
        generation = (
            "문서 분석 중 오류가 발생했습니다. "
            f"(검색 본문 일부: {text_context[:500]}...)" if len(text_context) > 500 else f"(검색 본문: {text_context})"
        )

    ctx_out = processed_pages_log if processed_pages_log else [attachment_status]
    return {
        "generation": generation,
        "context": ctx_out,
        "retrieved_chunk_texts": retrieved_chunk_texts,
    }
