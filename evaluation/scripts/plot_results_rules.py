import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 경로 설정
results_dir = "evaluation/results"
text_csv = os.path.join(results_dir, "ragas_report_text.csv")
pdf_csv = os.path.join(results_dir, "ragas_report_pdf.csv")

# 2. 데이터 로드
df_text = pd.read_csv(text_csv)
df_pdf = pd.read_csv(pdf_csv)

# 3. 평균 점수 계산 (RAGAS 4대 핵심 지표)
metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness']
text_scores = df_text[metrics].mean()
pdf_scores = df_pdf[metrics].mean()

# 시각화를 위한 데이터프레임 재구성
plot_data = pd.DataFrame({
    'Metric': metrics * 2,
    'Score': list(text_scores) + list(pdf_scores),
    'Condition': ['Text RAG (Baseline)'] * 4 + ['Multimodal RAG (Ours)'] * 4
})

# 4. 그래프 생성
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")
palette = sns.color_palette("muted")

ax = sns.barplot(x='Metric', y='Score', hue='Condition', data=plot_data, palette=palette)

# 그래프 디테일 설정
plt.title('RAG Evaluation Results: Text vs. Multimodal (Vision)', fontsize=16, pad=20)
plt.ylim(0, 1.0)
plt.ylabel('Average Score (0.0 - 1.0)', fontsize=12)
plt.xlabel('Ragas Metrics', fontsize=12)
plt.legend(title='Experiment Condition', loc='upper left')

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
output_plot = os.path.join(results_dir, "evaluation_comparison_plot.png")
plt.savefig(output_plot, dpi=300)
print(f"✅ 그래프가 저장되었습니다: {output_plot}")
plt.show()