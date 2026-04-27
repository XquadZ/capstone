import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import List, Set

from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드 (환경변수에 OPENAI_API_KEY가 있어야 함)
load_dotenv()


class GPTRefiner:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ 환경변수 'OPENAI_API_KEY'가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)
        self.fallback_client = None
        self.using_fallback = False
        saifex_api_key = os.getenv("SAIFEX_API_KEY")
        if saifex_api_key:
            self.fallback_client = OpenAI(
                api_key=saifex_api_key,
                base_url="https://ahoseo.saifex.ai/v1",
            )
        self.model = "gpt-4o-mini"
        self.allowed_categories = self._load_allowed_categories()
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        category_lines = "\n".join(f"- {c}" for c in self.allowed_categories)
        return f"""당신은 대학 공지사항 전문 정제기입니다.

제공된 [TITLE], [DATE], [URL], [CONTENT], [ATTACHMENT_TEXT]를 분석하여 핵심 정보를 보존하며 정제하십시오.

[반드시 준수할 작업 원칙]
1. 데이터 무결성: TITLE, DATE, URL은 수정 없이 그대로 유지하십시오.
2. 정보 극대화: CONTENT와 ATTACHMENT_TEXT에 포함된 상세 일정, 지원 자격, 제출 서류, 연락처 등 모든 구체적 정보를 마크다운 리스트나 표를 활용하여 상세히 정리하십시오. (절대 요약하여 생략하지 마십시오.)
3. 노이즈 제거: 의미 없는 서식 기호나 빈 칸 양식은 삭제하십시오.
4. 출력 형식: 오직 순수한 JSON 객체 하나만 출력하십시오.
5. category는 반드시 아래 허용 목록 중 정확히 1개만 선택하십시오.

[허용 category 목록]
{category_lines}

[JSON 스키마]
{{
  "title": "원본 TITLE",
  "date": "원본 DATE",
  "url": "원본 URL",
  "metadata": {{
    "year": "YYYY",
    "category": "허용 목록 중 1개",
    "major_category": "대분류",
    "target": "대상",
    "entity": "주관 부서명"
  }},
  "refined_content": "상세 정제 내용 (마크다운 포맷)"
}}"""

    @staticmethod
    def _load_allowed_categories() -> List[str]:
        """기존 정제 산출물에서 카테고리 폐쇄집합 로드 (없으면 기본값)."""
        base = Path(__file__).resolve().parents[1]
        text_dir = base / "data" / "processed" / "text"
        categories: Set[str] = set()
        if text_dir.exists():
            for file in text_dir.glob("*.json"):
                try:
                    data = json.loads(file.read_text(encoding="utf-8"))
                    cat = str(((data.get("metadata") or {}).get("category") or "")).strip()
                    if cat:
                        categories.add(cat)
                except Exception:
                    continue
        if categories:
            return sorted(categories)
        return ["공지사항", "학사", "장학", "취업", "창업", "비교과 프로그램", "행사", "안내", "모집", "기타"]

    @staticmethod
    def _infer_major_category(subcategory: str) -> str:
        cat = (subcategory or "").strip().lower()
        rules = [
            ("학사/수업", ["학사", "수강", "수업", "시험", "졸업", "학위", "교육과정", "집중이수"]),
            ("장학/등록금/근로", ["장학", "등록금", "근로"]),
            ("취업/진로/채용", ["취업", "진로", "채용", "인턴", "면접", "직무"]),
            ("창업/산학", ["창업", "산학", "캡스톤", "프로젝트", "해커톤"]),
            ("국제/교류/연수", ["교환학생", "유학생", "국제", "연수", "해외", "어학"]),
            ("비교과/특강/행사", ["특강", "세미나", "워크숍", "행사", "공모전", "경진대회", "설명회", "멘토", "학습"]),
            ("모집/선발/신청", ["모집", "선발", "신청", "참여자"]),
            ("생활/복지", ["기숙사", "생활관", "학생식당", "학생증", "상담", "심리", "복지", "셔틀", "통학", "주차"]),
            ("안전/보안/시설", ["안전", "예비군", "훈련", "보안", "교통", "시설", "시스템 중단", "서비스 중단"]),
            ("연구/사업/행정", ["연구", "지원사업", "정부포상", "대학혁신", "사업", "조사", "행정"]),
        ]
        for major, keys in rules:
            if any(k in cat for k in keys):
                return major
        return "기타/일반"

    def _normalize_category(self, raw_category: str) -> str:
        cat = (raw_category or "").strip()
        if not self.allowed_categories:
            return cat or "공지사항"
        if cat in self.allowed_categories:
            return cat
        match = difflib.get_close_matches(cat, self.allowed_categories, n=1, cutoff=0.5)
        if match:
            return match[0]
        return "공지사항" if "공지" in cat else self.allowed_categories[0]

    def _postprocess_result(self, result: dict) -> dict:
        if not isinstance(result, dict):
            return result
        meta = result.setdefault("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
            result["metadata"] = meta
        normalized = self._normalize_category(str(meta.get("category", "")))
        meta["category"] = normalized
        meta["major_category"] = self._infer_major_category(normalized)
        return result

    @staticmethod
    def _is_openai_quota_error(error) -> bool:
        msg = str(error).lower()
        keywords = [
            "insufficient_quota",
            "quota",
            "rate limit",
            "429",
            "billing",
            "exceeded your current quota",
        ]
        return any(k in msg for k in keywords)

    def refine(self, raw_text, max_retries=3):
        """실패 시 재시도 로직 포함"""
        # GPT-4o-mini의 넉넉한 컨텍스트 활용 (약 15,000자까지 수용)
        truncated_text = raw_text[:15000]

        for attempt in range(max_retries):
            try:
                active_client = self.fallback_client if self.using_fallback else self.client
                response = active_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"다음 텍스트를 JSON으로 정제하십시오:\n\n{truncated_text}"},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                parsed = json.loads(response.choices[0].message.content)
                return self._postprocess_result(parsed)
            except Exception as e:
                if (not self.using_fallback) and self.fallback_client and self._is_openai_quota_error(e):
                    print("\n🚨 OpenAI 비용/한도 이슈 감지 → SAIFEX로 전환하여 재시도합니다.")
                    self.using_fallback = True
                    continue
                wait_time = (attempt + 1) * 2
                print(f"\n⚠️ {attempt+1}회차 시도 실패 ({e}). {wait_time}초 후 재시도...")
                time.sleep(wait_time)
        return None


def process_directory(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    refiner = GPTRefiner()
    files = sorted(list(input_path.glob("*.txt")))  # 정렬해서 순차적 처리
    failed_files = []

    print("🌟 [돈의 맛] GPT-4o-mini 정제 프로세스 시작")
    print(f"📂 총 대상 파일: {len(files)}개\n")

    for i, file in enumerate(files):
        # 1. 이어하기 로직: 이미 파일이 있으면 건너뜀
        target_file = output_path / f"{file.stem}.json"
        if target_file.exists():
            # 파일이 존재하지만 내용이 비어있는지 체크 (선택사항)
            if target_file.stat().st_size > 10:
                continue

        print(f"[{i+1}/{len(files)}] {file.name} 처리 중...", end=" ", flush=True)

        try:
            raw_text = file.read_text(encoding="utf-8")
            result = refiner.refine(raw_text)

            if result:
                with open(target_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print("✅ 성공")
            else:
                failed_files.append(file.name)
                print("❌ 최종 실패")

        except Exception as e:
            failed_files.append(file.name)
            print(f"❌ 치명적 에러: {e}")

    # 최종 리포트 출력
    print("\n" + "=" * 60)
    print("🚀 모든 데이터 정제 프로세스 종료")
    print(f"📊 총 처리: {len(files)}건")
    print(f"✅ 성공: {len(files) - len(failed_files)}건")
    print(f"❌ 실패: {len(failed_files)}건")

    if failed_files:
        print("\n📂 수동 확인이 필요한 실패 파일 리스트:")
        for idx, f_name in enumerate(failed_files):
            print(f"   {idx+1}. {f_name}")
    print("=" * 60)


if __name__ == "__main__":
    # 경로 설정 (본인의 환경에 맞게 수정)
    INPUT_DIR = "./data/processed/integrated_text"
    OUTPUT_DIR = "./data/processed/text"

    start_time = time.time()
    process_directory(INPUT_DIR, OUTPUT_DIR)
    end_time = time.time()

    print(f"\n⏱️ 총 소요 시간: {int(end_time - start_time) // 60}분 {int(end_time - start_time) % 60}초")