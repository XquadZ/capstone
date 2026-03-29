import os
import json
import pandas as pd

# ==========================================
# ⚙️ 1. 경로 설정
# ==========================================
RESULTS_DIR = "evaluation/results"
STEP9_PROGRESS_PATH = os.path.join(RESULTS_DIR, "step9_progress.json")
RAGAS_SUMMARY_PATH = os.path.join(RESULTS_DIR, "RAGAS_Final_Summary.csv")
THESIS_TABLE_PATH = os.path.join(RESULTS_DIR, "Thesis_Final_Table.csv")

def main():
    print("\n" + "="*80)
    print("🎓 [STEP 11] 논문용 최종 데이터 테이블 병합 및 생성 시작")
    print("="*80)

    # ---------------------------------------------------------
    # 📊 [표 1] 라우터(Router) 성능 요약 (Step 8 & OOD 결과)
    # ---------------------------------------------------------
    # 형님이 터미널에서 뽑으신 결과를 하드코딩하여 예쁜 표로 만듭니다.
    router_performance = {
        "Metric": ["Accuracy (In-Domain)", "F1-Score (In-Domain)", "Accuracy (OOD - Zero Shot)"],
        "Score": ["98.0%", "0.98", "78.6%"],
        "Description": [
            "호서대 공지사항 도메인 (147개 테스트셋)",
            "TEXT/VISION 클래스 불균형 없는 안정적 성능",
            "8개 타 산업 도메인 낯선 질의 (42개 테스트셋) 일반화 성능"
        ]
    }
    df_router = pd.DataFrame(router_performance)

    print("\n[표 1] V4 라우터 분류 성능 및 일반화 검증 결과")
    print("-" * 80)
    print(df_router.to_string(index=False))
    print("-" * 80)

    # ---------------------------------------------------------
    # 📊 [표 2] RAG 파이프라인 End-to-End 성능 비교 (Step 9 & 10)
    # ---------------------------------------------------------
    if not os.path.exists(STEP9_PROGRESS_PATH) or not os.path.exists(RAGAS_SUMMARY_PATH):
        print("⚠️ 에러: Step 9(json) 또는 Step 10(csv) 결과 파일이 없습니다.")
        return

    # 1. Step 9 데이터 로드 (지연시간, 라우팅 정확도)
    with open(STEP9_PROGRESS_PATH, 'r', encoding='utf-8') as f:
        step9_data = json.load(f)
    
    latency_map = {
        "Always_Text": step9_data.get("avg_text_latency", 0.0),
        "Always_Vision": step9_data.get("avg_vision_latency", 0.0),
        "TV_RAG": step9_data.get("avg_tv_latency", 0.0)
    }
    
    # TV-RAG의 라우팅 정확도
    route_acc = step9_data.get("route_accuracy", 0.0) * 100

    # 2. Step 10 데이터 로드 (RAGAS)
    df_ragas = pd.read_csv(RAGAS_SUMMARY_PATH)

    # 3. 데이터 병합 (Latency 추가)
    df_ragas['Latency(s)'] = df_ragas['Pipeline'].map(latency_map)
    
    # 4. 논문용 컬럼명 변경 및 재배치
    rename_dict = {
        'Pipeline': 'Model Pipeline',
        'faithfulness': 'Faithfulness ↑',
        'answer_correctness': 'Correctness ↑',
        'answer_relevancy': 'Relevancy ↑',
        'context_precision': 'Ctx Precision ↑',
        'context_recall': 'Ctx Recall ↑'
    }
    df_ragas.rename(columns=rename_dict, inplace=True)

    # 라우팅 정확도 컬럼 추가 (TV-RAG에만 표기)
    df_ragas.insert(1, 'Routing Acc', ["-" if p != "TV_RAG" else f"{route_acc:.1f}%" for p in df_ragas['Model Pipeline']])

    # 소수점 3자리로 예쁘게 포맷팅
    for col in df_ragas.columns:
        if '↑' in col:  # RAGAS 지표들
            df_ragas[col] = df_ragas[col].apply(lambda x: f"{float(x):.3f}" if pd.notnull(x) else "-")
        elif col == 'Latency(s)': # 지연시간은 2자리
            df_ragas[col] = df_ragas[col].apply(lambda x: f"{float(x):.2f}")

    print("\n[표 2] 멀티모달 RAG 파이프라인별 End-to-End 성능 비교")
    print("-" * 100)
    print(df_ragas.to_string(index=False))
    print("-" * 100)

    # CSV로 최종 저장 (엑셀에서 복붙하기 좋게)
    df_ragas.to_csv(THESIS_TABLE_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 엑셀용 파일이 성공적으로 저장되었습니다: {THESIS_TABLE_PATH}")
    print("이제 이 두 개의 표를 논문 '실험 결과(Results)' 섹션에 복사+붙여넣기 하시면 됩니다!")
    print("="*80)

if __name__ == "__main__":
    main()