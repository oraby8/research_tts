import torch

class GPUAllocator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUAllocator, cls).__new__(cls)
            cls._instance.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            cls._instance.current_idx = 0
        return cls._instance
    
    def get_device(self):
        if self.num_gpus == 0:
            return "cpu"
        device = f"cuda:{self.current_idx % self.num_gpus}"
        self.current_idx += 1
        return device

def get_next_device():
    return GPUAllocator().get_device()
