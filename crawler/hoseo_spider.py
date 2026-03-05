import os
import json
import time
import requests
import re
import urllib3
import traceback
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# SSL 경고 무시 (연구실 보안 환경 대응)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file_with_session(url, save_path, driver):
    """브라우저의 쿠키와 헤더를 완벽히 복사하여 연구실 방화벽 우회"""
    try:
        session = requests.Session()
        # 1. 브라우저 쿠키 이식
        for cookie in driver.get_cookies():
            session.cookies.set(cookie['name'], cookie['value'])
        
        # 2. 연구실 방화벽 통과를 위한 필수 헤더 설정
        headers = {
            'User-Agent': driver.execute_script("return navigator.userAgent;"),
            'Referer': driver.current_url,
            'Origin': 'https://www.hoseo.ac.kr',
            'Accept': 'application/octet-stream, */*'
        }
        
        response = session.get(url, headers=headers, stream=True, verify=False, timeout=30)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"   ❌ 다운로드 실패: {e}")
        return False

def crawl_notices_by_date(target_date="2026-03-04"):
    short_date = "-".join(target_date.split("-")[1:])
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    wait = WebDriverWait(driver, 20)
    
    base_url = "https://www.hoseo.ac.kr"
    list_url = "https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_1708240139"
    
    try:
        print(f"📅 목표 날짜({target_date}) 수집 시작...")
        driver.get(list_url)
        
        # 1. 목록 로딩 및 대상 인덱스 탐색
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
            print(f"ℹ️ {target_date} 날짜의 공지가 없습니다.")
            return

        # 2. 상세 수집 루프
        for idx in target_indices:
            try:
                # 목록 복귀 후 재탐색
                wait.until(EC.presence_of_element_located((By.ID, "example1")))
                rows = driver.find_elements(By.CSS_SELECTOR, "#example1 tbody tr")
                link_el = rows[idx].find_element(By.CSS_SELECTOR, "td.board-list-title a")
                title = link_el.text.strip()
                
                print(f"📂 수집 중: {title}")
                driver.execute_script("arguments[0].click();", link_el)
                
                # 본문 대기 (여러 셀렉터 대응)
                content_selectors = [".board-view-content", "#board_item_list", ".fileBox"]
                content_area = None
                for selector in content_selectors:
                    try:
                        content_area = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
                        if content_area: break
                    except: continue

                # 고유 ID 및 폴더 생성
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
                    "content": content_area.text.strip() if content_area else "",
                    "saved_images": [],
                    "saved_attachments": []
                }

                # [HTML 구조 반영] 첨부파일 다운로드 로직
                # 사진 속 'div.fileBox' 내부의 'li > a'를 정확히 타격합니다.
                try:
                    file_links = driver.find_elements(By.CSS_SELECTOR, ".fileBox .fileList li a[href*='Download.do']")
                    for f in file_links:
                        # 텍스트에서 파일명만 추출 (줄바꿈 제거)
                        f_name = f.text.split('\n')[0].strip()
                        if f_name:
                            clean_name = re.sub(r'[\\/:*?"<>|]', '_', f_name)
                            file_url = urljoin(base_url, f.get_attribute("href"))
                            
                            print(f"   📥 첨부파일 시도: {clean_name}")
                            if download_file_with_session(file_url, os.path.join(attach_dir, clean_name), driver):
                                info_data["saved_attachments"].append(f"attachments/{clean_name}")
                                print(f"   ✅ 다운로드 성공")
                except Exception as fe:
                    print(f"   ⚠️ 첨부파일 영역 탐색 실패: {fe}")

                # 이미지 다운로드 (기존 방식 유지)
                imgs = driver.find_elements(By.CSS_SELECTOR, ".board-view-content img, #board_item_list img")
                for i, img in enumerate(imgs):
                    src = img.get_attribute("src")
                    if src and any(k in src for k in ["ThumbnailPrint", "upload", "MAPP"]):
                        filename = f"img_{i+1}.jpg"
                        if download_file_with_session(urljoin(base_url, src), os.path.join(img_dir, filename), driver):
                            info_data["saved_images"].append(f"images/{filename}")

                # 결과 저장
                with open(os.path.join(base_dir, "info.json"), "w", encoding="utf-8") as f:
                    json.dump(info_data, f, ensure_ascii=False, indent=4)
                
                print(f"   ✅ {notice_id} 저장 완료")

            except Exception as e:
                print(f"   ❌ 공지 처리 중 오류: {e}")
            
            driver.get(list_url)
            time.sleep(2)

    except Exception:
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    crawl_notices_by_date("2026-03-04")