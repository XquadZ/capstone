import os
import json
import time
import requests
import re
import traceback
import urllib3
from urllib.parse import urljoin # 상대 경로 변환을 위해 추가
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
        # 학교 서버는 User-Agent가 없으면 거부할 수 있으므로 추가
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/145.0.0.0'}
        response = session.get(url, headers=headers, stream=True, verify=False, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"   ❌ 다운로드 실패: {e}")
        return False

def crawl_specific_notice(target_idx="95774"):
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    # 기본 베이스 URL (상대 경로 변환용)
    base_url = "https://www.hoseo.ac.kr"
    direct_url = f"{base_url}/Home/BBSView.mbz?action=MAPP_1708240139&schIdx={target_idx}"
    
    # 로컬 저장 경로 설정
    work_dir = os.getcwd()
    base_dir = os.path.join(work_dir, "data", "raw", target_idx)
    img_dir = os.path.join(base_dir, "images")
    attach_dir = os.path.join(base_dir, "attachments")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(attach_dir, exist_ok=True)
    
    try:
        print(f"🚀 호서대 공지사항(ID: {target_idx}) 접속 중...")
        driver.get(direct_url)
        wait = WebDriverWait(driver, 20)

        # 제목 추출
        try:
            title_el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.board-view-title h4, h4")))
            title = title_el.text.strip()
        except:
            title = "제목 없음"
        print(f"✅ 제목 확인: {title}")

        # 본문 영역 확보
        content_area = driver.find_element(By.ID, "board_item_list")
        
        info_data = {
            "notice_id": target_idx,
            "title": title,
            "content": content_area.text.strip(),
            "saved_images": [],
            "saved_attachments": []
        }

        cookies = driver.get_cookies()

        # 🖼️ [해결] 본문 내 이미지 저장 로직 강화
        print("🖼️ 본문 이미지 수집 및 저장 시작...")
        # dd 내부의 모든 img 태그를 찾습니다.
        imgs = content_area.find_elements(By.TAG_NAME, "img")
        
        for i, img in enumerate(imgs):
            raw_src = img.get_attribute("src")
            if not raw_src: continue

            # [핵심] 상대 경로(/ThumbnailPrint...)를 절대 경로로 변환
            full_src = urljoin(base_url, raw_src)
            
            # 본문 이미지(ThumbnailPrint 포함) 혹은 업로드된 이미지만 수집
            if "ThumbnailPrint" in full_src or "upload" in full_src or "MAPP" in full_src:
                # 너무 작은 아이콘 제외 (가로 100px 이상만)
                if img.size['width'] > 100:
                    filename = f"img_{i+1}.jpg"
                    save_path = os.path.join(img_dir, filename)
                    
                    if download_file(full_src, save_path, cookies):
                        info_data["saved_images"].append(f"images/{filename}")
                        print(f"   📸 이미지 저장 완료: {filename}")

        # 📎 첨부파일 저장
        print("📎 첨부파일 수집 및 저장 시작...")
        files = driver.find_elements(By.CSS_SELECTOR, ".fileList a, .fileBox a")
        for f in files:
            f_name = f.text.strip()
            f_url = f.get_attribute("href")
            if f_url and "javascript" not in f_url.lower():
                clean_name = re.sub(r'[\\/:*?"<>|]', '_', f_name)
                save_path = os.path.join(attach_dir, clean_name)
                
                # 첨부파일 URL도 절대 경로 확인
                full_f_url = urljoin(base_url, f_url)
                if download_file(full_f_url, save_path, cookies):
                    info_data["saved_attachments"].append(f"attachments/{clean_name}")
                    print(f"   📎 첨부파일 저장 완료: {clean_name}")

        # 메타데이터 저장
        with open(os.path.join(base_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info_data, f, ensure_ascii=False, indent=4)
        
        print(f"✨ 모든 데이터가 '{base_dir}' 폴더에 성공적으로 저장되었습니다.")

    except Exception:
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    crawl_specific_notice("95774")