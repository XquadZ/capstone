import os
import json
import time
import requests
import re
import traceback
import urllib3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file(url, save_path, cookies):
    try:
        session = requests.Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'])
        response = session.get(url, stream=True, verify=False, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"   ❌ 다운로드 에러: {e}")
        return False

def crawl_specific_notice(target_idx="95774"):
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    url = f"https://www.hoseo.ac.kr/Home/BBSView.mbz?action=MAPP_1708240139&schIdx={target_idx}"
    
    base_dir = f"data/raw/{target_idx}"
    os.makedirs(f"{base_dir}/images", exist_ok=True)
    os.makedirs(f"{base_dir}/attachments", exist_ok=True)
    
    try:
        print(f"🚀 호서대 공지사항(ID: {target_idx}) 접속 중...")
        driver.get(url)
        wait = WebDriverWait(driver, 15)
        time.sleep(3) # 안정적인 로딩 대기

        # 1. 제목 추출 (여러 후보 선택자 사용)
        title = "제목 없음"
        title_selectors = [
            "div.board-view-title h4",
            "h4.ui-type-content-title",
            "div.fix-layout h4"
        ]
        for selector in title_selectors:
            try:
                title_el = driver.find_element(By.CSS_SELECTOR, selector)
                title = title_el.text.strip()
                if title: break
            except: continue
        print(f"✅ 제목: {title}")

        # 2. 본문 텍스트 추출 (ID가 없으므로 넓은 범위에서 텍스트 수집)
        print("📝 본문 내용 수집 중...")
        content_text = ""
        # 호서대 게시판 본문은 보통 <dd> 또는 특정 div 안에 있음
        content_selectors = ["div#board_item_list", "div.board-view-content", "dl.board-view-info + dd", "div.ui-type-content"]
        for selector in content_selectors:
            try:
                target = driver.find_element(By.CSS_SELECTOR, selector)
                content_text = target.text.strip()
                if content_text: break
            except: continue

        info_data = {
            "notice_id": target_idx,
            "title": title,
            "content": content_text,
            "saved_images": [],
            "saved_attachments": []
        }

        cookies = driver.get_cookies()

        # 3. 이미지 수집 (본문 영역 내 이미지)
        print("🖼️ 이미지 분석 중...")
        imgs = driver.find_elements(By.TAG_NAME, "img")
        for i, img in enumerate(imgs):
            src = img.get_attribute("src")
            # 학교 내부 경로를 포함한 실제 데이터 이미지만 필터링
            if src and ("MAPP" in src or "Common" in src or "upload" in src):
                path = f"{base_dir}/images/img_{i+1}.jpg"
                if download_file(src, path, cookies):
                    info_data["saved_images"].append(path)
                    print(f"   📸 저장 완료: img_{i+1}.jpg")

        # 4. 첨부파일 수집
        print("📎 첨부파일 분석 중...")
        files = driver.find_elements(By.CSS_SELECTOR, "div.fileBox a, div.fileList a")
        for f in files:
            name = f.text.strip()
            link = f.get_attribute("href")
            if link and "javascript" not in link.lower():
                clean_name = re.sub(r'[\\/:*?"<>|]', '_', name)
                path = f"{base_dir}/attachments/{clean_name}"
                if download_file(link, path, cookies):
                    info_data["saved_attachments"].append(path)
                    print(f"   📎 저장 완료: {clean_name}")

        # 5. 결과 저장
        with open(f"{base_dir}/info.json", "w", encoding="utf-8") as f:
            json.dump(info_data, f, ensure_ascii=False, indent=4)
        
        print(f"✨ 수집 성공! 모든 데이터가 {base_dir} 폴더에 저장되었습니다.")

    except Exception:
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    crawl_specific_notice("95774")