import torch
import torch.nn as nn

print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Create a simple tensor and move it to GPU
x = torch.randn(1000, 1000)
print("\nTensor device before moving to CUDA:", x.device)

# Try-except to catch any CUDA errors
try:
    x = x.cuda()
    print("Tensor device after moving to CUDA:", x.device)

    # Perform a simple operation to verify GPU computation
    y = torch.matmul(x, x)
    print("\nMatrix multiplication completed successfully on:", y.device)

except Exception as e:
    print("\nError when trying to use CUDA:", str(e))

# Check CUDA memory usage
if torch.cuda.is_available():
    print("\nCUDA Memory Summary:")
    print(torch.cuda.memory_summary())
