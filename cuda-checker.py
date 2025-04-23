
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available")
