import os
import torch
from torch.utils.cpp_extension import CUDA_HOME, load

VOCAB_SIZE = 50000
K_VALUE = 2048
NUM_ITERATIONS = 1000
NUM_WARMUPS = 100
CUDA_FILE_PATH="/home/kyle/inference/moe-inference-ops/csrc/kernels"

if not torch.cuda.is_available() or CUDA_HOME is None:
    print("CUDA not available. Cannot run benchmarks.")
    exit()

cuda_file = f"{CUDA_FILE_PATH}/topk.cu"
if not os.path.exists(cuda_file):
    print(f"Error: {cuda_file} not found.")
    exit()

print(f"Compiling CUDA extension...")
topk_ext = load(
    name="topk_ext",
    sources=[cuda_file],
    extra_cuda_cflags=[
        "-O3", 
        f"-arch=sm_{torch.cuda.get_device_properties(0).major}0", 
        "-std=c++17",
        # Pass these to the host compiler to stop it from using the 
        # problematic built-ins in Ubuntu 24.04's stdlib.h
        "-Xcompiler", "-U_FORTIFY_SOURCE",
        "-Xcompiler", "-D_FORTIFY_SOURCE=0"
    ],
    verbose=True,
)

# def benchmark():
#     logits = torch.randn(VOCAB_SIZE, device="cuda", dtype=torch.float32)

#     for _ in range(NUM_WARMUPS):
#         _ = topk_ext.forward(logits, K_VALUE)
#         _ = torch.topk(logits, K_VALUE)
#     torch.cuda.synchronize()

#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)

#     start_event.record()
#     for _ in range(NUM_ITERATIONS):
#         custom_idx = topk_ext.forward(logits, K_VALUE)
#     end_event.record()

#     torch.cuda.synchronize()
#     custom_ms = start_event.elapsed_time(end_event) / NUM_ITERATIONS