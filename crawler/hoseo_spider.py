from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def debug_hoseo_crawl():
    chrome_options = Options()
    # 1. 눈으로 확인하기 위해 Headless 모드를 잠시 끕니다. (중요!)
    # 서버 모니터에서 브라우저가 직접 뜨는지 확인하세요.
    # chrome_options.add_argument("--headless") 
    
    chrome_options.add_argument("--window-size=1280,1024")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    url = "https://www.hoseo.ac.kr/Home/BBSList.mbz?action=MAPP_1708240139"
    
    try:
        print(f"🚀 브라우저를 실행합니다. 화면을 지켜봐 주세요...")
        driver.get(url)
        
        # 2. 충분한 대기 시간 부여
        time.sleep(7) 

        # 3. [핵심] iframe이 있는지 확인하고 안으로 들어가기
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        if iframes:
            print(f"ℹ️ {len(iframes)}개의 프레임을 발견했습니다. 내부로 진입을 시도합니다.")
            driver.switch_to.frame(0) # 첫 번째 프레임으로 전환

        # 4. 공지사항 제목들이 담긴 태그 찾기 (좀 더 넓은 범위로 탐색)
        # 텍스트가 5글자 이상 들어있는 모든 <a> 태그를 찾아봅니다.
        print("🔍 공지사항 제목 데이터를 검색 중...")
        possible_titles = driver.find_elements(By.XPATH, "//a[string-length(text()) > 5]")
        
        print(f"\n--- [발견된 텍스트 목록] ---")
        found = False
        for i, item in enumerate(possible_titles):
            title_text = item.text.strip()
            if "공지" in title_text or "학사" in title_text or "장학" in title_text: # 키워드 필터링
                print(f"[{i+1}] {title_text}")
                found = True
        
        if not found:
            print("❗ 필터링된 제목이 없습니다. 전체 텍스트 상위 5개를 출력합니다:")
            for i, item in enumerate(possible_titles[:5]):
                print(f"샘플 {i+1}: {item.text.strip()}")
        print("--------------------------\n")

    except Exception as e:
        print(f"🚨 에러 발생: {e}")
    finally:
        input("계속하려면 엔터를 누르세요... (브라우저 확인용)") # 확인 후 닫기 위해 대기
        driver.quit()

if __name__ == "__main__":
    debug_hoseo_crawl()