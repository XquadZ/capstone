# 🛠️ Infrastructure Setup Guide

이 문서는 RTX 4090 기반의 로컬 서버 환경 구축을 위한 가이드입니다.

## 1. Hardware Requirements
- **GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM)
- **CPU:** Intel i9 / AMD Ryzen 9 이상 (멀티스레드 크롤링 및 전처리용)
- **RAM:** 64GB 이상 (Vector DB 및 모델 로드 최적화)
- **Storage:** NVMe SSD 1TB 이상 (데이터 수집 및 DB 저장용)

## 2. Software Stack
- **OS:** Ubuntu 22.04 LTS (권장)
- **NVIDIA Driver:** 535+ (CUDA 12.1 지원 필수)
- **Container:** Docker & NVIDIA Container Toolkit (GPU 가속 컨테이너화)
- **AI Framework:** vLLM (LLM 서빙), PyTorch 2.1+

## 3. Environment Setup (Quick Start)
1. **NVIDIA Driver 및 Docker 설치**
2. **Python 가상환경 설정:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt