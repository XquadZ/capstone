import os

def find_chunking_logic(search_path="."):
    target_keywords = ["data/chunks", "processed/chunks", "chunks"]
    found_files = []

    print(f"🔎 '{search_path}' 경로에서 청킹 로직 파일을 찾는 중...")

    # 프로젝트 내의 모든 디렉토리와 파일을 순회
    for root, dirs, files in os.walk(search_path):
        # 가상환경이나 캐시 폴더는 제외 (속도 향상)
        if 'venv' in root or '__pycache__' in root or '.git' in root:
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            # 키워드가 라인에 포함되어 있는지 확인
                            if any(keyword in line for keyword in target_keywords):
                                found_files.append({
                                    "file": file_path,
                                    "line_no": i + 1,
                                    "content": line.strip()
                                })
                except Exception as e:
                    # 인코딩 문제 등으로 읽을 수 없는 파일은 건너뜀
                    pass

    if found_files:
        print(f"\n✅ 로직이 의심되는 파일을 {len(found_files)}개 찾았습니다:\n")
        current_file = ""
        for item in found_files:
            if current_file != item['file']:
                print(f"📂 파일 경로: {item['file']}")
                current_file = item['file']
            print(f"   📍 [Line {item['line_no']}]: {item['content']}")
    else:
        print("\n❌ 해당 로직이 포함된 파일을 찾지 못했습니다. 키워드를 변경해 보세요.")

if __name__ == "__main__":
    # 캡스톤 프로젝트 최상위 경로에서 실행하거나 경로를 지정하세요.
    find_chunking_logic(".")