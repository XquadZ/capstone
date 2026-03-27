import json
import os

# 확인하고 싶은 파일 경로
FILE_PATH = "evaluation/datasets/final_intent_balanced_dataset.json"

def check_route_counts():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 총 데이터 개수: {len(data)}개")
    
    # route 개수 카운트
    text_count = sum(1 for item in data if item.get("route") == "TEXT")
    vision_count = sum(1 for item in data if item.get("route") == "VISION")
    other_count = len(data) - (text_count + vision_count)

    print("-" * 30)
    print(f"✅ TEXT   : {text_count}개")
    print(f"✅ VISION : {vision_count}개")
    if other_count > 0:
        print(f"⚠️ 기타(라벨없음): {other_count}개")
    print("-" * 30)
    
    if text_count == vision_count:
        print("🔥 완벽한 5:5 황금 밸런스입니다! ㄲㄲ!")
    else:
        print("💡 비율이 조금 차이가 나네요. 확인이 필요할 수도 있습니다.")

if __name__ == "__main__":
    check_route_counts()