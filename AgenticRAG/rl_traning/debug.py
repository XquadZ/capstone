import torch

print("="*30)
cuda_ok = torch.cuda.is_available()
print(f"CUDA Available: {cuda_ok}")

if cuda_ok:
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
else:
    print("❌ GPU가 인식되지 않습니다. PyTorch가 CPU 버전으로 설치되었을 가능성이 높습니다.")
print("="*30)