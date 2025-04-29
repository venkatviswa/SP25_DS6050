# device_config.py
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # For Apple Silicon
    else:
        return "cpu"
