"""
PDF 문서 로더 및 텍스트 분할 모듈

system_arch.md 설계에 따라:
- PyMuPDFLoader로 data/ 폴더의 PDF 로드
- RecursiveCharacterTextSplitter로 800자 청크, 100자 오버랩
- RTX 4090 환경 대량 문서 처리 최적화 (병렬 I/O, 메모리 효율)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 기본 경로 (프로젝트 루트 기준)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# RTX 4090 환경: I/O 병렬화를 위한 워커 수 (CPU 코어 활용, 과도한 동시 I/O 방지)
MAX_WORKERS = 8

logger = logging.getLogger(__name__)


def _get_pdf_paths(data_dir: Path | None = None) -> list[Path]:
    """data/ 폴더 내 모든 PDF 파일 경로를 재귀적으로 수집"""
    base = data_dir or DATA_DIR
    if not base.exists():
        logger.warning("데이터 폴더가 존재하지 않습니다: %s", base)
        return []
    return sorted(base.rglob("*.pdf"))


def _load_single_pdf(pdf_path: Path) -> list[Document]:
    """단일 PDF 파일 로드 (페이지 단위)"""
    loader = PyMuPDFLoader(str(pdf_path), extract_images=False)
    return loader.load()


def load_pdfs(
    data_dir: Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> list[Document]:
    """
    data/ 폴더의 모든 PDF를 병렬로 로드.

    RTX 4090 환경에서 대량 문서 처리 시:
    - ThreadPoolExecutor로 I/O 병렬화 (디스크 읽기 병렬)
    - extract_images=False로 메모리 절약
    """
    paths = _get_pdf_paths(data_dir)
    if not paths:
        return []

    all_docs: list[Document] = []
    workers = min(max_workers, len(paths))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(_load_single_pdf, p): p for p in paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                docs = future.result()
                all_docs.extend(docs)
                logger.debug("로드 완료: %s (%d 페이지)", path.name, len(docs))
            except Exception as e:
                logger.exception("PDF 로드 실패: %s - %s", path, e)

    return all_docs


def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """설계도 규격의 RecursiveCharacterTextSplitter 반환"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    문서 리스트를 청크로 분할.

    - chunk_size: 800자
    - chunk_overlap: 100자 (문맥 유지)
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)


def load_and_split(
    data_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_workers: int = MAX_WORKERS,
) -> list[Document]:
    """
    data/ 폴더의 PDF를 로드하고 청크로 분할하는 원스텝 함수.

    RTX 4090 대량 문서 처리 최적화:
    1. 병렬 PDF 로드 (I/O)
    2. RecursiveCharacterTextSplitter로 800자/100자 오버랩 분할
    """
    docs = load_pdfs(data_dir, max_workers)
    if not docs:
        return []
    return split_documents(docs, chunk_size, chunk_overlap)


def load_and_split_stream(
    data_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_workers: int = MAX_WORKERS,
) -> Iterator[Document]:
    """
    메모리 효율 스트리밍: PDF별로 로드 → 분할 → yield.

    대용량 문서 처리 시 전체를 메모리에 올리지 않고 청크 단위로 스트리밍.
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    paths = _get_pdf_paths(data_dir)
    workers = min(max_workers, len(paths))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(_load_single_pdf, p): p for p in paths}
        for future in as_completed(future_to_path):
            try:
                docs = future.result()
                for chunk in splitter.split_documents(docs):
                    yield chunk
            except Exception as e:
                logger.exception("PDF 처리 실패: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = load_and_split()
    print(f"총 {len(chunks)}개 청크 생성")
    if chunks:
        print("첫 청크 미리보기:", chunks[0].page_content[:200], "...")
