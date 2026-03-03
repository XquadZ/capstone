# 🛠️ Infrastructure Setup Guide (RTX 4090 Local)

본 문서는 호서대학교 스마트 캠퍼스 도우미 챗봇의 **로컬 서버 구축 및 라이브 시연(Demo) 환경 세팅 가이드**입니다.
클라우드 서버(AWS 등)를 사용하지 않고, 연구실 또는 개인의 NVIDIA RTX 4090 GPU 자원을 100% 활용하여 실시간 API 서빙과 데이터베이스를 구동하는 것을 목표로 합니다.

## 1. System Requirements (권장 하드웨어 및 OS)

* **GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM 필수)
* **CPU:** Intel Core i9 또는 AMD Ryzen 9 이상 (멀티스레드 크롤링 및 백그라운드 작업용)
* **RAM:** 64GB 이상 (대규모 Vector DB 캐싱 및 로컬 텍스트 전처리용)
* **Storage:** 1TB NVMe SSD 이상 (속도가 느린 HDD 사용 시 DB 읽기/쓰기 병목 발생)
* **OS:** Ubuntu 22.04 LTS (Windows WSL2 환경에서도 구동 가능하나 Linux Native 권장)

---

## 2. VRAM 예산 관리 (Critical: OOM 방지 전략)

RTX 4090의 24GB VRAM 한계를 초과(Out of Memory)하지 않기 위해, 서비스 실행 전 모델별 메모리 할당량을 엄격하게 통제해야 합니다.

| 구성 요소 | 사용 모델 | VRAM 점유율 | 최적화 필수 사항 |
| :--- | :--- | :--- | :--- |
| **LLM (생성)** | `EEVE-Korean-10.8B` | **약 8.0 ~ 10.0 GB** | **AWQ 또는 GGUF 양자화(4-bit/8-bit) 버전 사용 필수.** (원본 FP16 사용 시 20GB 이상 점유하여 서버 다운) |
| **Embedding (검색)** | `BAAI/bge-m3` | **약 4.0 GB** | Batch Size 조절로 메모리 스파이크 방지. |
| **Reranker (재정렬)** | `ko-reranker` | **약 3.0 GB** | Top-K를 10개 내외로 제한하여 연산량 통제. |
| **KV Cache & OS** | `vLLM` 버퍼 및 시스템 | **약 5.0 ~ 7.0 GB** | vLLM 실행 시 `--gpu-memory-utilization 0.4` 수준으로 제한. |
| **Total** | | **Max ~23.5 GB** | 아슬아슬하므로 라이브 데모 중 크롤러 실행 절대 금지. |

---

## 3. Environment Setup (설치 및 실행 가이드)

### 단계 1: 시스템 드라이버 및 Docker 준비
GPU 가속 및 하이브리드 데이터베이스 구동을 위한 기본 환경을 세팅합니다.
```bash
# 1. NVIDIA Driver 및 CUDA 12.1 이상 설치 확인
nvidia-smi 

# 2. Docker 및 NVIDIA Container Toolkit 설치 (Ubuntu 기준)
sudo apt update
sudo apt install docker.io
# (NVIDIA Container Toolkit 설치는 공식 문서 참조)