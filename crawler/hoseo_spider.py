from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def crawl_recent_notices():
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1280,1024")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    url = "https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_1708240139"
    
    try:
        print(f"🚀 호서대 공지사항 접속... (기준: 2026년 3월 이후 전체)")
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "example1")))

        # 1. 목록에서 2026년 3월 이후 게시글 추출
        rows = driver.find_elements(By.CSS_SELECTOR, "table#example1 tbody tr")
        target_notices = []

        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                # 등록일자 칸 추출 (보통 마지막에서 두 번째 혹은 마지막 칸)
                date_text = cells[-1].text.strip() 
                
                # 날짜 비교: 2026-03-01 이후 데이터만 수집
                # 문자열 비교 방식으로 '2026-03-01' 보다 큰 날짜는 모두 포함
                if date_text >= "2026-03-01":
                    title_el = row.find_element(By.CSS_SELECTOR, "td.board-list-title a")
                    title_text = title_el.text.strip()
                    js_code = title_el.get_attribute("href")
                    
                    target_notices.append({
                        "title": title_text,
                        "date": date_text,
                        "js": js_code
                    })
            except: continue

        print(f"📊 검사 대상 선정 완료: 총 {len(target_notices)}건 발견")

        # 2. 각 공지사항 상세 분석 루프
        for i, notice in enumerate(target_notices):
            print(f"\n" + "="*50)
            print(f"🔍 [{i+1}/{len(target_notices)}] 분석 중: {notice['title']}")
            print(f"📅 등록일: {notice['date']}")
            
            # 상세 페이지 진입
            driver.execute_script(notice['js'])
            time.sleep(3) # 페이지 로딩 대기

            # --- 데이터 추출 및 형식 파악 ---
            # 본문 텍스트
            try:
                content_area = driver.find_element(By.CSS_SELECTOR, "div#board_item_list dd")
                text_content = content_area.text.strip()
            except: text_content = ""

            # 이미지 파일
            imgs = driver.find_elements(By.CSS_SELECTOR, "div#board_item_list img")
            img_list = [img.get_attribute("src") for img in imgs if img.get_attribute("src")]

            # 첨 be파일 (PDF/HWP/ZIP 등)
            files = driver.find_elements(By.CSS_SELECTOR, "div.fileBox div.fileList ul li a")
            attachments = []
            for f in files:
                f_name = f.text.strip()
                f_url = f.get_attribute("href")
                if f_url and "javascript" not in f_url:
                    attachments.append(f_name)

            # --- 결과 요약 출력 ---
            print(f"📝 텍스트 데이터: {'있음' if len(text_content) > 0 else '없음'} ({len(text_content)}자)")
            print(f"🖼️ 이미지 데이터: {len(img_list)}개 발견")
            print(f"📎 첨부파일 데이터: {len(attachments)}개 ({', '.join(attachments) if attachments else '없음'})")
            
            # 형식 진단
            if len(text_content) < 10 and len(img_list) > 0:
                print("🚩 진단: [이미지 전용 공지] -> 이미지 OCR/Vision 분석 필요")
            elif len(attachments) > 0 and ".pdf" in str(attachments).lower():
                print("🚩 진단: [파일 중심 공지] -> PDF 텍스트 추출기 연동 필요")
            else:
                print("🚩 진단: [일반 텍스트 공지]")

            # 목록으로 돌아가기
            driver.back()
            time.sleep(2)
            wait.until(EC.presence_of_element_located((By.ID, "example1")))

    except Exception as e:
        print(f"🚨 오류 발생: {e}")
    finally:
        driver.quit()
        print("\n🏁 모든 분석 작업이 종료되었습니다.")

if __name__ == "__main__":
    crawl_recent_notices()