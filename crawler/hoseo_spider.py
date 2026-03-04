import os
import json
import time
import requests
import re
import traceback
import urllib3
from datetime import datetime
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file(url, save_path, cookies):
    """물리적 파일 다운로드 및 로컬 저장"""
    try:
        session = requests.Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'])
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = session.get(url, headers=headers, stream=True, verify=False, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"   ❌ 다운로드 실패: {e}")
        return False

def crawl_notices_by_date(target_date="2026-03-04"):
    """날짜 기반 공지 수집 및 상세 페이지 로딩 이슈 해결 버전"""
    short_date = "-".join(target_date.split("-")[1:])
    
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    # 상세 페이지 로딩 상황을 모니터링하기 위해 브라우저 창을 띄웁니다.
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    wait = WebDriverWait(driver, 25) # 대기 시간을 25초로 설정
    
    base_url = "https://www.hoseo.ac.kr"
    list_url = "https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_1708240139"
    
    try:
        print(f"📅 목표 날짜({target_date}) 공지 수집 시작...")
        driver.get(list_url)
        
        # 1. 목록 로딩 대기 및 타겟 인덱스 확보
        wait.until(EC.presence_of_element_located((By.ID, "example1")))
        rows = driver.find_elements(By.CSS_SELECTOR, "#example1 tbody tr")
        
        target_indices = []
        for idx, row in enumerate(rows):
            try:
                date_cols = row.find_elements(By.CSS_SELECTOR, "td.pc_view")
                if date_cols and (date_cols[-1].text.strip() in [target_date, short_date]):
                    target_indices.append(idx)
            except: continue

        if not target_indices:
            print(f"ℹ️ {target_date} 날짜의 공지를 찾을 수 없습니다.")
            return

        print(f"📢 총 {len(target_indices)}개의 공지 발견. 상세 수집 루프 진입...")

        # 2. 순차 수집 실행
        for idx in target_indices:
            try:
                # 목록 복귀 후 요소 재탐색 (StaleElement 방지)
                wait.until(EC.presence_of_element_located((By.ID, "example1")))
                rows = driver.find_elements(By.CSS_SELECTOR, "#example1 tbody tr")
                link_el = rows[idx].find_element(By.CSS_SELECTOR, "td.board-list-title a")
                title = link_el.text.strip()
                
                print(f"📂 [{idx+1}/{len(target_indices)}] 수집 중: {title}")
                
                # 자바스크립트 클릭 실행
                driver.execute_script("arguments[0].click();", link_el)
                
                # [핵심] 상세 페이지 본문 영역을 여러 셀렉터로 탐색
                content_selectors = [".board-view-content", "#board_item_list", ".board_view_content", "dd.content"]
                content_area = None
                
                for selector in content_selectors:
                    try:
                        # 요소가 화면에 나타날 때까지 대기
                        content_area = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
                        if content_area and content_area.text.strip():
                            break
                    except: continue

                if not content_area:
                    print(f"   ⚠️ 상세 페이지 본문 요소를 찾지 못했습니다. 목록으로 복귀합니다.")
                    driver.get(list_url)
                    continue

                # 고유 ID 및 경로 설정
                idx_match = re.search(r'schIdx=(\d+)', driver.current_url)
                notice_id = idx_match.group(1) if idx_match else str(int(time.time()))
                
                base_dir = os.path.join(os.getcwd(), "data", "raw", notice_id)
                img_dir = os.path.join(base_dir, "images")
                attach_dir = os.path.join(base_dir, "attachments")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(attach_dir, exist_ok=True)

                info_data = {
                    "notice_id": notice_id,
                    "target_date": target_date,
                    "title": title,
                    "content": content_area.text.strip(),
                    "saved_images": [],
                    "saved_attachments": []
                }

                cookies = driver.get_cookies()

                # 이미지 다운로드 로직
                imgs = content_area.find_elements(By.TAG_NAME, "img")
                for i, img in enumerate(imgs):
                    src = img.get_attribute("src")
                    if src and any(k in src for k in ["ThumbnailPrint", "upload", "MAPP"]):
                        filename = f"img_{i+1}.jpg"
                        if download_file(urljoin(base_url, src), os.path.join(img_dir, filename), cookies):
                            info_data["saved_images"].append(f"images/{filename}")

                # 첨부파일 다운로드 로직
                files = driver.find_elements(By.CSS_SELECTOR, "a[href*='FileDownload']")
                for f in files:
                    f_name = f.text.strip()
                    if f_name:
                        clean_name = re.sub(r'[\\/:*?"<>|]', '_', f_name)
                        if download_file(urljoin(base_url, f.get_attribute("href")), os.path.join(attach_dir, clean_name), cookies):
                            info_data["saved_attachments"].append(f"attachments/{clean_name}")

                # 결과 저장
                with open(os.path.join(base_dir, "info.json"), "w", encoding="utf-8") as f:
                    json.dump(info_data, f, ensure_ascii=False, indent=4)
                
                print(f"   ✅ {notice_id} 수집 성공")

            except Exception as e:
                print(f"   ❌ 오류 발생: {e}")
            
            # 안전한 루프를 위해 목록 페이지로 명시적 이동
            driver.get(list_url)
            time.sleep(2)

    except Exception:
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    crawl_notices_by_date("2026-03-04")