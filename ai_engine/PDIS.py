import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# 기존 파이프라인 임포트 (경로 자동 최적화)
try:
    from rag_pipeline import HoseoRAGPipeline
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rag_pipeline import HoseoRAGPipeline

class PDISResearchEngine(HoseoRAGPipeline):
    def __init__(self, collection_name="hoseo_notices"):
        # 부모 클래스의 모델 및 DB 연결 상속
        super().__init__(collection_name)
        print("\n" + "=".center(75, "="))
        print("📊 [PDIS Research Engine] 호서대 RAG 성능 분석 시스템 가동".center(63))
        print("=".center(75, "=") + "\n")

    def get_refined_data(self, query_text, top_k):
        """
        [핵심] Milvus에서 dense_vector를 강제로 추출하고 결측치를 정제함
        """
        # 1. 쿼리 임베딩
        q_emb = self.embed_model.encode([query_text], return_dense=True)
        q_vec = q_emb['dense_vecs'][0].astype(np.float32)

        # 2. Milvus 직접 검색 (output_fields에 dense_vector 강제 포함)
        # Baseline과의 공정성을 위해 search_params는 동일하게 설정
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[q_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_text", "dense_vector"], 
            consistency_level="Strong"
        )

        valid_vecs = []
        valid_entities = []

        if results and len(results[0]) > 0:
            for hit in results[0]:
                vec = hit.entity.get('dense_vector')
                # NaN, Inf, None이 없는 깨끗한 데이터만 선별
                if vec is not None and np.isfinite(np.sum(vec)):
                    valid_vecs.append(vec)
                    valid_entities.append(hit.entity)
        
        return q_vec, np.array(valid_vecs, dtype=np.float32), valid_entities

    def run_pdis_core(self, q_vec, c_vecs, c_entities, dims=[128, 512, 1024], Ks=[100, 50, 10]):
        """
        PDIS의 순수 알고리즘 연산 시간만을 정밀 측정
        """
        t_start = time.perf_counter()
        
        curr_vecs = c_vecs
        curr_entities = c_entities

        # 점진적 차원 확장 필터링 루프
        for i in range(1, len(dims)):
            target_dim = dims[i]
            target_k = min(Ks[i], len(curr_entities))

            # 차원 슬라이싱 및 쿼리 정렬
            red_q = q_vec[:target_dim].reshape(1, -1)
            red_c = curr_vecs[:, :target_dim]

            # 4090 CPU 병렬 KNN 연산 (이 부분이 논문의 핵심 속도 포인트)
            nn = NearestNeighbors(n_neighbors=target_k, metric='euclidean', n_jobs=-1)
            nn.fit(red_c)
            _, indices = nn.kneighbors(red_q)
            
            # 후보군 압축
            target_idx = indices[0]
            curr_vecs = curr_vecs[target_idx]
            curr_entities = [curr_entities[j] for j in target_idx]

        t_end = time.perf_counter()
        return curr_entities, (t_end - t_start)

def run_experiment():
    engine = PDISResearchEngine()
    
    # [설정] 논문용 테스트 쿼리셋
    test_queries = [
        "기숙사 신청 기간과 제출 서류 알려줘",
        "성적 경고 시 장학금 제한 기준",
        "호서대학교 셔틀버스 정류장 위치",
        "학생 식단표 확인하는 법",
        "도서관 이용 시간 및 대출 규정"
    ]

    # [1] 시스템 예열 (Warm-up)
    # 첫 쿼리 지연(5초대)을 방지하기 위해 더미 쿼리로 모델 및 DB 캐시 활성화
    print("🔥 시스템 예열 중 (GPU/DB Cache Warm-up)...")
    _ = engine.get_refined_data("예열용 쿼리", 50)
    time.sleep(1)

    results = []
    print(f"🚀 실험 시작: 총 {len(test_queries)}개 쿼리 분석")

    for i, q in enumerate(test_queries):
        # A. Baseline 측정 (기본 전체 차원 방식)
        t_b_start = time.perf_counter()
        _ = engine.search_and_rerank(q, retrieve_k=50, final_k=10)
        t_baseline = time.perf_counter() - t_b_start

        # B. PDIS 측정 (데이터 로드 + 알고리즘 연산)
        t_p_load_start = time.perf_counter()
        q_vec, c_vecs, c_ents = engine.get_refined_data(q, 100) # Ks[0]=100
        t_p_load = time.perf_counter() - t_p_load_start
        
        if len(c_vecs) > 0:
            res_pdis, t_p_pure = engine.run_pdis_core(q_vec, c_vecs, c_ents)
            t_pdis_total = t_p_load + t_p_pure
            
            speed_up = t_baseline / t_pdis_total
            
            results.append({
                "ID": i+1, "Query": q[:15],
                "Baseline_Latency": t_baseline,
                "PDIS_Total_Latency": t_pdis_total,
                "PDIS_Pure_Algo": t_p_pure,
                "Speed_Up": speed_up
            })
            print(f"✅ [{i+1}] Speed-up: {speed_up:.2f}x | PDIS: {t_pdis_total:.4f}s")

    # [결과 분석 및 저장]
    if results:
        df = pd.DataFrame(results)
        df.to_csv("PDIS_Research_Final_Report.csv", index=False, encoding='utf-8-sig')
        
        print("\n" + "=".center(50, "="))
        print(f"📊 최종 평균 속도 향상: {df['Speed_Up'].mean():.2f}배")
        print(f"📊 알고리즘 순수 연산 평균: {df['PDIS_Pure_Algo'].mean():.6f}s")
        print("=".center(50, "="))

        # 시각화 그래프 생성
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['ID'], y=df['Baseline_Latency'], name='Baseline (Full)'))
        fig.add_trace(go.Bar(x=df['ID'], y=df['PDIS_Total_Latency'], name='PDIS (Proposed)'))
        fig.update_layout(title="PDIS Performance Analysis", yaxis_title="Time (sec)", barmode='group')
        fig.write_html("PDIS_Analysis_Graph.html")

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        import traceback
        print(f"🔥 치명적 에러 발생:\n{traceback.format_exc()}")