import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 경로 설정
results_dir = "evaluation/results"
rev_text_csv = os.path.join(results_dir, "ragas_reverse_report_text.csv")
rev_gt_csv = os.path.join(results_dir, "ragas_reverse_report_orig_gt.csv")

# 2. 데이터 로드
df_rev_text = pd.read_csv(rev_text_csv)
df_rev_gt = pd.read_csv(rev_gt_csv)

# 3. 평균 점수 계산
metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness']
rev_text_scores = df_rev_text[metrics].mean()
rev_gt_scores = df_rev_gt[metrics].mean()

# 시각화 데이터 구성
plot_data = pd.DataFrame({
    'Metric': metrics * 2,
    'Score': list(rev_text_scores) + list(rev_gt_scores),
    'Condition': ['Text RAG vs Vision GT'] * 4 + ['Original GT vs Vision GT'] * 4
})

# 4. 그래프 생성
plt.figure(figsize=(13, 7))
sns.set_theme(style="whitegrid")

# 역방향 평가인 만큼 조금 더 강렬한 색상(빨간색 계열)을 사용해 차이를 강조합니다.
ax = sns.barplot(x='Metric', y='Score', hue='Condition', data=plot_data, palette="OrRd")

plt.title('Reverse Evaluation: How much info did Text-only methods miss?', fontsize=16, pad=20)
plt.ylim(0, 1.0)
plt.ylabel('Score (Relative to Vision Gold Standard)', fontsize=12)
plt.xlabel('Ragas Metrics', fontsize=12)

# 막대 위에 점수 표시
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=11, fontweight='bold')

plt.tight_layout()

# 5. 결과 저장
output_plot = os.path.join(results_dir, "reverse_evaluation_plot.png")
plt.savefig(output_plot, dpi=300)
print(f"✅ 역평가 그래프 저장 완료: {output_plot}")
plt.show()