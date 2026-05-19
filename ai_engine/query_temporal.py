"""공지 RAG 검색용 질의 연도 탐지 및 보강.

질의에 연도 표기가 없으면 검색 쿼리 끝에 기본 연도(2026년)를 붙입니다.
월·학기·날짜 등은 검사하지 않습니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Pattern, Tuple

DEFAULT_YEAR_SUFFIX = "2026년"
DEFAULT_MILVUS_YEAR = "2026"

# 연·년·연도를 나타내는 표기만 탐지
_YEAR_MARKER_PATTERNS: Tuple[str, ...] = (
    r"20\d{2}\s*학년도",
    r"20\d{2}\s*년",
    r"(?<![\d])20\d{2}(?![\d년월일/\-.])",
    r"(?<![\d])(?:1\d|2[0-9])\s*년(?!\s*생)",
    r"연도",
)

_COMPILED: Tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in _YEAR_MARKER_PATTERNS
)

_YEAR_EXTRACT_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"(20\d{2})\s*학년도"),
    re.compile(r"(20\d{2})\s*년"),
    re.compile(r"(?<![\d])(20\d{2})(?![\d년월일/\-.])"),
    re.compile(r"(?<![\d])((?:1\d|2[0-9]))\s*년(?!\s*생)"),
)


@dataclass(frozen=True)
class NoticeSearchQueryPrep:
    original: str
    search_query: str
    has_explicit_year: bool
    filter_year: Optional[str] = None


def has_explicit_year_expression(text: str) -> bool:
    """질의에 연도(2026년, 26년, 2026, 2026학년도, 연도 등)가 있는지 판별."""
    if not (text or "").strip():
        return False
    return any(p.search(text) for p in _COMPILED)


def extract_calendar_years(text: str) -> List[str]:
    """질의에서 4자리 연도 후보를 추출 (중복 제거, 등장 순)."""
    if not text:
        return []
    seen: set[str] = set()
    years: List[str] = []
    for pat in _YEAR_EXTRACT_PATTERNS:
        for m in pat.finditer(text):
            y = m.group(1)
            if len(y) == 2:
                y = f"20{y}"
            if y not in seen:
                seen.add(y)
                years.append(y)
    return years


def prepare_notice_search_query(
    query: str,
    *,
    default_suffix: str = DEFAULT_YEAR_SUFFIX,
    default_year: str = DEFAULT_MILVUS_YEAR,
) -> NoticeSearchQueryPrep:
    """검색용 쿼리 준비: 연도 미명시 시 default_suffix(기본 2026년)를 붙임."""
    original = (query or "").strip()
    if not original:
        return NoticeSearchQueryPrep(
            original="",
            search_query="",
            has_explicit_year=False,
            filter_year=default_year,
        )

    if has_explicit_year_expression(original):
        years = extract_calendar_years(original)
        return NoticeSearchQueryPrep(
            original=original,
            search_query=original,
            has_explicit_year=True,
            filter_year=years[-1] if years else None,
        )

    suffix = (default_suffix or "").strip()
    if suffix and suffix in original:
        search_query = original
    elif suffix:
        search_query = f"{original} {suffix}"
    else:
        search_query = original

    return NoticeSearchQueryPrep(
        original=original,
        search_query=search_query,
        has_explicit_year=False,
        filter_year=default_year,
    )
