import torch

print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))
print("PyTorch version:", torch.__version__)
print("Device being used:", torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"))
