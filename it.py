import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version PyTorch was built with:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
