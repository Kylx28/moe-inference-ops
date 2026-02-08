import os
import torch
from torch.utils.cpp_extension import CUDA_HOME, load

VOCAB_SIZE = 50000
K_VALUE = 64
NUM_ITERATIONS = 1000
NUM_WARMUPS = 100
CUDA_FILE_PATH="../csrc/kernels"

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
        "-Xcompiler", "-U_FORTIFY_SOURCE",
        "-Xcompiler", "-D_FORTIFY_SOURCE=0"
    ],
    verbose=True,
)

def benchmark():
    # Random Input Tokens
    logits = torch.randn(VOCAB_SIZE, device="cuda", dtype=torch.float32)

    # Warmup CUDA
    for _ in range(NUM_WARMUPS):
        _ = topk_ext.forward(logits, K_VALUE)
        _ = torch.topk(logits, K_VALUE)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(NUM_ITERATIONS):
        custom_idx = topk_ext.forward(logits, K_VALUE)
    end_event.record()

    torch.cuda.synchronize()
    custom_ms = start_event.elapsed_time(end_event) / NUM_ITERATIONS

    start_event.record()
    for _ in range(NUM_ITERATIONS):
        torch_vals, torch_idx = torch.topk(logits, K_VALUE)
    end_event.record()

    torch.cuda.synchronize()
    torch_ms = start_event.elapsed_time(end_event) / NUM_ITERATIONS

    print("\n" + "="*40)
    print(f"{'Method':<20} | {'Avg Latency (ms)':<15}")
    print("-" * 40)
    print(f"{'PyTorch Top-K':<20} | {torch_ms:>15.6f}")
    print(f"{'Custom Histogram':<20} | {custom_ms:>15.6f}")
    print("="*40)

    custom_vals = logits[custom_idx.long()].sort(descending=True)[0]
    torch_vals_sorted = torch_vals.sort(descending=True)[0]

    euclidean_dist = torch.dist(custom_vals, torch_vals_sorted, p=2).item()

    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(custom_vals, torch_vals_sorted).item()
    
    print("\n" + "="*40)
    print(f"{'Metric':<20} | {'Value':<10}")
    print("=" * 40)
    print(f"{'Euclidean Dist':<20} | {euclidean_dist:>10.6f}")
    print(f"{'Cosine Similarity':<20} | {similarity:>10.6f}")
    print("=" * 40)


if __name__ == "__main__":
    benchmark()