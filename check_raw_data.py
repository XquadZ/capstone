import os
import json
from pathlib import Path

def analyze_capstone_raw_data(base_path):
    raw_dir = Path(base_path) / "data" / "raw"
    
    if not raw_dir.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {raw_dir}")
        return

    report = {
        "total_notices": 0,
        "missing_info_json": [],
        "with_attachments": 0,
        "with_images": 0,
        "total_attachment_files": 0,
        "total_image_files": 0
    }

    print(f"🚀 RAW 데이터 스캔 시작: {raw_dir}\n" + "="*50)

    # 폴더 리스트 가져오기 (notice_id 기반)
    notice_folders = [f for f in raw_dir.iterdir() if f.is_dir()]
    report["total_notices"] = len(notice_folders)

    for folder in notice_folders:
        notice_id = folder.name
        info_json = folder / "info.json"
        attachment_dir = folder / "attachments"
        image_dir = folder / "images"

        # 1. info.json 확인
        if not info_json.exists():
            report["missing_info_json"].append(notice_id)
            continue

        # 2. 첨부파일 확인
        if attachment_dir.exists():
            files = list(attachment_dir.glob("*"))
            if files:
                report["with_attachments"] += 1
                report["total_attachment_files"] += len(files)

        # 3. 이미지 확인
        if image_dir.exists():
            imgs = list(image_dir.glob("*"))
            if imgs:
                report["with_images"] += 1
                report["total_image_files"] += len(imgs)

    # 결과 요약 출력
    print(f"📊 스캔 결과 요약")
    print(f"- 전체 공지사항 수: {report['total_notices']}개")
    print(f"- info.json 누락: {len(report['missing_info_json'])}개")
    print(f"- 첨부파일 보유 공지: {report['with_attachments']}개 (총 {report['total_attachment_files']}개 파일)")
    print(f"- 이미지 보유 공지: {report['with_images']}개 (총 {report['total_image_files']}개 파일)")
    
    if report["missing_info_json"]:
        print(f"⚠️ 경고: info.json 누락된 ID: {report['missing_info_json'][:10]}...")

    # 샘플 데이터 구조 확인 (첫 번째 폴더 대상)
    if notice_folders:
        sample_id = notice_folders[0].name
        sample_json_path = notice_folders[0] / "info.json"
        print("\n📝 데이터 샘플 확인 (ID: " + sample_id + ")")
        print("-" * 30)
        with open(sample_json_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
            print(f"Title: {sample_data.get('title', 'N/A')}")
            content_preview = sample_data.get('content', '')[:100].replace('\n', ' ')
            print(f"Content Preview: {content_preview}...")
            print(f"Attachments: {sample_data.get('attachments', [])}")

if __name__ == "__main__":
    # 최상위 capstone 폴더 경로 (현재 실행 위치 기준)
    current_path = os.getcwd()
    analyze_capstone_raw_data(current_path)