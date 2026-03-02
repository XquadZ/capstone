from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def crawl_hoseo_details():
    chrome_options = Options()
    # 확인을 위해 브라우저를 띄웁니다.
    chrome_options.add_argument("--window-size=1280,1024")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    url = "https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_1708240139"
    
    try:
        print(f"🚀 호서대 게시판 접속 및 정밀 파싱 시작...")
        driver.get(url)
        
        # 표가 로딩될 때까지 명시적 대기 (최대 10초)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "example1")))

        # 1. 공지사항 행(tr)들 가져오기
        rows = driver.find_elements(By.CSS_SELECTOR, "table#example1 tbody tr")
        
        print(f"\n✅ 발견된 공지사항 목록:")
        notice_list = []

        for row in rows:
            try:
                # 제목 요소 찾기
                title_element = row.find_element(By.CSS_SELECTOR, "td.board-list-title a")
                title_text = title_element.text.strip()
                # 자바스크립트 함수 내용 (예: javascript:fn_viewData('95707');)
                js_code = title_element.get_attribute("href")
                
                if title_text:
                    print(f"- {title_text}")
                    notice_list.append((title_element, title_text, js_code))
            except:
                continue

        # 2. 테스트: 첫 번째 공지사항 클릭해서 들어가기
        if notice_list:
            print(f"\n🖱️ 첫 번째 게시글 클릭 시도: [{notice_list[0][1]}]")
            # element.click() 대신 안정적인 자바스크립트 실행 방식 사용
            driver.execute_script(notice_list[0][2])
            
            # 상세 페이지 로딩 대기
            time.sleep(5)
            print("✅ 상세 페이지 진입 성공! 이제 여기서 본문과 이미지를 추출하면 됩니다.")
            
            # 현재 상세 페이지의 제목이나 본문 일부 출력 테스트
            print(f"📄 현재 페이지 URL: {driver.current_url}")

    except Exception as e:
        print(f"🚨 에러 발생: {e}")
    finally:
        input("\n상세 페이지 화면을 확인하신 후 엔터를 누르세요...")
        driver.quit()

if __name__ == "__main__":
    crawl_hoseo_details()