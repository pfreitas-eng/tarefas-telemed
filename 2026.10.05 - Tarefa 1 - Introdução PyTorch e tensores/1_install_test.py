import torch

# This script prints information about the installed PyTorch version and GPU availability.


def print_pytorch_info():
    version = torch.__version__
    print("PyTorch version:", version)

    cuda_version = torch.version.cuda
    print("CUDA version:", cuda_version)

    gpu_available = torch.cuda.is_available()
    print("GPU available:", gpu_available)

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")

        gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_total_memory_gb = gpu_total_memory / (1024 ** 3)
        print(
            f"GPU Total Memory: {gpu_total_memory} bytes ({gpu_total_memory_gb:.2f} GB)")


if __name__ == "__main__":
    print_pytorch_info()
