import torch
from enum import Enum
import torch_config as config


class DeviceType(Enum):
    CPU = 1,
    GPU = 2


class GPUContext:
    cpu_name = "cpu"
    using_gpu = False
    device = torch.device(cpu_name)

    def __init__(self, gpu_name: str = "cuda"):
        self.gpu_name = gpu_name
        if torch.cuda.is_available():
            self.gpu_available = True
            print("(GPU is available)")
        else:
            self.gpu_available = False
            print("(GPU is not available)")
        self.set_default_device(DeviceType.CPU)

    def set_default_device(self, device_type: DeviceType):
        if not self.gpu_available:
            return
        if device_type == DeviceType.GPU:
            self.device = torch.device(self.gpu_name)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
            print(f"(Using Device: {self.gpu_name})")
            self.using_gpu = True
            return
        # (device_type == DeviceType.CPU)
        self.device = torch.device(self.cpu_name)
        torch.set_default_tensor_type(torch.FloatTensor)  # type: ignore
        print(f"(Using Device: {self.cpu_name})")
        self.using_gpu = False

    def __enter__(self):
        self.set_default_device(DeviceType.GPU)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.set_default_device(DeviceType.CPU)


device_context = GPUContext(config.CUDA_NAME)
