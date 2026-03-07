import os
import json
import time
import requests
import re
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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HoseoRealCrawler:
    def __init__(self):
        self.base_url = "https://www.hoseo.ac.kr"
        self.list_url_template = "https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_1708240139&schCategorycode=CTG_17082400011&pageIndex={}"
        self.view_url_template = "https://www.hoseo.ac.kr/Home/BBSView.mbz?action=MAPP_1708240139&schCategorycode=CTG_17082400011&schIdx={}"
        
        self.save_root = os.path.join(os.getcwd(), "data", "raw")
        
        # 🎯 목표 기한을 2024년 1월 1일로 연장
        self.target_limit = datetime(2024, 1, 1)
        
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1280,1024")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        
        print("🌐 브라우저를 실행 중입니다...")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)

    def download_file(self, url, save_path):
        try:
            session = requests.Session()
            for cookie in self.driver.get_cookies():
                session.cookies.set(cookie['name'], cookie['value'])
            headers = {'User-Agent': self.driver.execute_script("return navigator.userAgent;"), 'Referer': self.driver.current_url}
            response = session.get(url, headers=headers, stream=True, verify=False, timeout=30)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                return True
            return False
        except: return False

    def crawl_details(self, notice_id, title, date_str):
        try:
            target_url = self.view_url_template.format(notice_id)
            self.driver.get(target_url)
            
            # 본문 대기 (스크린샷 기반 ID 사용)
            self.wait.until(EC.presence_of_element_located((By.ID, "board_item_list")))
            
            base_path = os.path.join(self.save_root, str(notice_id))
            os.makedirs(os.path.join(base_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(base_path, "attachments"), exist_ok=True)

            content_el = self.driver.find_element(By.ID, "board_item_list")
            
            info = {
                "notice_id": str(notice_id),
                "title": title,
                "date": date_str,
                "url": target_url,
                "content": content_el.text.strip(),
                "attachments": [],
                "images": []
            }

            # 1. 첨부파일 수집
            for f in self.driver.find_elements(By.CSS_SELECTOR, "a[href*='Download.do']"):
                try:
                    f_name = f.text.split('\n')[0].strip()
                    if not f_name: continue
                    clean_name = re.sub(r'[\\/:*?"<>|]', '_', f_name)
                    if self.download_file(urljoin(self.base_url, f.get_attribute("href")), os.path.join(base_path, "attachments", clean_name)):
                        info["attachments"].append(clean_name)
                except: continue

            # 2. 이미지 수집 (Thumbnail 경로 포함 전수 수집)
            imgs = content_el.find_elements(By.TAG_NAME, "img")
            for i, img in enumerate(imgs):
                try:
                    src = img.get_attribute("src")
                    if src and not src.startswith("data:"):
                        img_url = urljoin(self.base_url, src)
                        img_name = f"img_{i}.jpg"
                        if self.download_file(img_url, os.path.join(base_path, "images", img_name)):
                            info["images"].append(img_name)
                except: continue

            with open(os.path.join(base_path, "info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=4)

            return True
        except Exception:
            return False

    def run(self):
        print(f"\n🚀 호서대 정밀 크롤링 시작 (~{self.target_limit.strftime('%Y-%m-%d')})")
        page = 1
        try:
            while True:
                self.driver.get(self.list_url_template.format(page))
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
                
                rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                targets_to_crawl = []
                
                for row in rows:
                    try:
                        num_text = row.find_element(By.CSS_SELECTOR, "td[data-header='번호']").text.strip()
                        date_text = row.find_element(By.CSS_SELECTOR, "td[data-header='등록일자']").text.strip()
                        link_el = row.find_element(By.CSS_SELECTOR, "td.board-list-title a")
                        
                        is_pinned = not num_text.isdigit()
                        if len(date_text) <= 5: date_text = f"{datetime.now().year}-{date_text}"
                        row_date = datetime.strptime(date_text, "%Y-%m-%d")

                        # [중요] 타겟 날짜 도달 시 종료
                        if not is_pinned and row_date < self.target_limit:
                            print(f"\n🏁 타겟 날짜({date_text}) 도달! 수집을 최종 완료합니다.")
                            return

                        href_val = link_el.get_attribute("href") or ""
                        real_sch_idx = None
                        match = re.search(r"fn_viewData\('(\d+)'\)", href_val)
                        if match: real_sch_idx = match.group(1)

                        if real_sch_idx:
                            targets_to_crawl.append({"id": real_sch_idx, "title": link_el.text.strip(), "date": date_text})
                    except: continue

                for target in targets_to_crawl:
                    # 중복 스킵 로직: 이미 폴더가 있으면 상세 페이지 방문 안 함
                    if os.path.exists(os.path.join(self.save_root, target["id"])):
                        continue

                    if self.crawl_details(target["id"], target["title"], target["date"]):
                        print(f"✅ ID: {target['id']} | 수집 완료: [{target['date']}] {target['title'][:20]}...")
                    else:
                        print(f"❌ ID: {target['id']} | 수집 실패")
                    time.sleep(0.4)
                
                page += 1
                time.sleep(1)
        finally:
            self.driver.quit()

if __name__ == "__main__":
    HoseoRealCrawler().run()