import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # 한글 폰트 깨짐 방지 세팅 (윈도우용)
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False 

    file_path = "evaluation/results/ragas_evaluation_report.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # 데이터 불러오기
    df = pd.read_csv(file_path)

    # 1. 잔고 부족으로 채점 안 된 빈칸(NaN) 데이터 깔끔하게 제거
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness']
    df_clean = df.dropna(subset=metrics)
    
    print(f"📊 총 {len(df)}문제 중 정상 채점된 {len(df_clean)}문제를 바탕으로 그래프를 그립니다.")

    # 2. 지표별 평균값 계산
    mean_scores = df_clean[metrics].mean()

    # 3. 예쁜 막대그래프 그리기 (Seaborn 활용)
    plt.figure(figsize=(10, 6))
    
    # x축: 예쁜 이름으로 변경
    labels = ['Faithfulness\n(환각 방어)', 'Answer Relevancy\n(질문 관련성)', 
              'Context Precision\n(검색 정밀도)', 'Answer Correctness\n(정답 일치율)']
    
    ax = sns.barplot(x=labels, y=mean_scores.values, palette='viridis')

    # 그래프 꾸미기
    plt.ylim(0, 1.05) # 점수는 0~1 사이
    plt.title('호서대 RAG 시스템 평가 결과 (RAGAS 벤치마크)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score (0 ~ 1)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 막대 위에 정확한 점수 텍스트 표시
    for i, v in enumerate(mean_scores.values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 이미지 저장
    save_path = "evaluation/results/evaluation_plot.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) # 고화질 논문용 저장
    print(f"✅ 그래프가 저장되었습니다: {save_path}")
    
    # 화면에 띄우기
    plt.show()

if __name__ == "__main__":
    main()