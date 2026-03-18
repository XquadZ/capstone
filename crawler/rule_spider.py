import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def main():
    # 1. 저장할 디렉토리 설정 (제안해 드린 구조 반영)
    # 현재 스크립트 위치(crawler) 기준으로 상위 폴더의 data/rules_regulations/raw_pdfs 로 이동
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/rules_regulations/raw_pdfs'))
    os.makedirs(base_dir, exist_ok=True)

    # 2. 타겟 URL 및 도메인 설정 (캡처본 브라우저 주소창 참조)
    target_url = "https://www.hoseo.ac.kr/Home/Contents.mbz?action=MAPP_1708220030"
    domain = "https://www.hoseo.ac.kr"
    
    # 봇 차단 방지를 위한 User-Agent 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    print(f"🚀 호서대 학칙 페이지 접속 중: {target_url}")
    response = requests.get(target_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # 3. 다운로드 버튼 찾기 (href에 '/File/Download.do'가 포함된 모든 a 태그)
    download_links = soup.find_all('a', href=lambda x: x and '/File/Download.do' in x)

    print(f"🔍 총 {len(download_links)}개의 PDF 링크를 찾았습니다. 다운로드를 시작합니다!\n")

    success_count = 0
    for idx, link in enumerate(download_links, 1):
        href = link['href']
        full_url = urljoin(domain, href)

        # 4. 파일명 추출 마법 (DOM 트리 역추적)
        try:
            # <a> 태그를 감싸고 있는 <dd class="doc-box-btn"> 찾기
            parent_dd = link.find_parent('dd', class_='doc-box-btn')
            # 그 바로 앞에 있는 형제 <dd> 태그 찾기 (여기에 "1-1-1-1. 학칙" 텍스트가 있음)
            title_dd = parent_dd.find_previous_sibling('dd')
            raw_title = title_dd.get_text(strip=True)
            
            # 파일명에 쓸 수 없는 특수문자 제거 (\, /, :, *, ?, ", <, >, |)
            safe_title = "".join(c for c in raw_title if c not in r'\/:*?"<>|')
            filename = f"{safe_title}.pdf"
        except Exception:
            # 만약 DOM 구조가 예외적인 곳이 있다면 안전하게 fallback 이름 사용
            filename = f"hoseo_rule_{idx:03d}.pdf"

        file_path = os.path.join(base_dir, filename)

        # 5. 파일 다운로드 및 저장
        print(f"[{idx:03d}/{len(download_links):03d}] 📥 다운로드 중: {filename}")
        try:
            pdf_resp = requests.get(full_url, headers=headers, stream=True)
            pdf_resp.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in pdf_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            success_count += 1
        except Exception as e:
            print(f"  ❌ 다운로드 실패 ({filename}): {e}")

    print("\n" + "="*60)
    print(f"🎉 크롤링 완료! (성공: {success_count}/{len(download_links)}개)")
    print(f"💾 저장 위치: {base_dir}")
    print("="*60)

if __name__ == "__main__":
    main()