import torch
import os
from torch.utils.cpp_extension import load

N_ROWS = 4096  # Batch Size * Seq Length
N_EXPERTS = 32 # Must be <= 32 for warp
NUM_ITERATIONS = 1000
CUDA_FILE_PATH="../csrc/kernels"

# Compilation
print("Compiling CUDA extension...")
softmax_ext = load(
    name="softmax_ext",
    sources=[f"{CUDA_FILE_PATH}/softmax.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xcompiler", "-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=0"],
    verbose=True
)

def run_benchmark():
    input_tensor = torch.randn(N_ROWS, N_EXPERTS, device="cuda", dtype=torch.float32)
    
    with torch.no_grad():
        custom_out = softmax_ext.forward(input_tensor)
        torch_out = torch.softmax(input_tensor, dim=-1)
        
        euclidean_dist = torch.dist(custom_out, torch_out, p=2).item()
        cos_sim = torch.nn.functional.cosine_similarity(custom_out.flatten(), torch_out.flatten(), dim=0).item()
        
        print(f"\nAccuracy Verification:")
        print(f"Euclidean Distance: {euclidean_dist:.6e}")
        print(f"Cosine Similarity:  {cos_sim:.6f}")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Custom Kernel
    for _ in range(50): 
        softmax_ext.forward(input_tensor)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(NUM_ITERATIONS):
        _ = softmax_ext.forward(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    custom_time = start_event.elapsed_time(end_event) / NUM_ITERATIONS

    # PyTorch Baseline
    for _ in range(50):
        torch.softmax(input_tensor, dim=-1)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(NUM_ITERATIONS):
        _ = torch.softmax(input_tensor, dim=-1)
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / NUM_ITERATIONS

    print(f"\nPerformance Benchmark ({NUM_ITERATIONS} iterations):")
    print(f"{'Method':<20} | {'Avg Latency (ms)':<15}")
    print("-" * 40)
    print(f"{'PyTorch Softmax':<20} | {torch_time:>15.4f}")
    print(f"{'Custom Warp Softmax':<20} | {custom_time:>15.4f}")

if __name__ == "__main__":
    run_benchmark()