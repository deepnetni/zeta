import torch

# 查看 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 查看 CUDA 版本
print(f"CUDA version: {torch.version.cuda}")

# 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 查看当前使用的设备
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
