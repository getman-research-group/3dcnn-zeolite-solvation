import torch
import time
import argparse
import os
import socket

def benchmark(device, dtype, size=1024, repeats=10):
    """
    Perform repeated matrix multiplication and report performance.
    """
    try:
        # Generate two random matrices with the specified data type
        x = torch.randn(size, size, device=device, dtype=dtype)
        y = torch.randn(size, size, device=device, dtype=dtype)
        
        # Warm-up run to compile kernels, etc.
        z = x @ y
        
        # Synchronize to ensure warm-up is complete before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            start_time = time.time()
            z = x @ y
            # Synchronize after the operation to get accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000) # milliseconds
            
        mean_ms = sum(times) / repeats
        std_ms = (sum((t - mean_ms) ** 2 for t in times) / repeats) ** 0.5
        
        print(f"  - Device: {str(device.type).upper()}, Dtype: {str(dtype).split('.')[-1]}, Size: {size}x{size}")
        print(f"    Avg Time: {mean_ms:.2f} ± {std_ms:.2f} ms (over {repeats} runs)")

    except Exception as e:
        print(f"  - Could not run benchmark for {str(dtype)} on {device.type.upper()}: {e}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Benchmark Script for HPC")
    parser.add_argument('--size', type=int, default=2048, help='Matrix dimension')
    parser.add_argument('--repeats', type=int, default=20, help='Number of repetitions')
    args = parser.parse_args()

    print("="*60)
    print("           PyTorch Environment & Hardware Benchmark")
    print("="*60)
    
    # --- System and Environment Information ---
    print("\n--- System Info ---")
    print(f"SLURM Job ID:      {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Hostname:          {socket.gethostname()}")
    print(f"PyTorch Version:   {torch.__version__}")

    # --- GPU Information ---
    print("\n--- GPU Info ---")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Available:    {cuda_ok}")
    if cuda_ok:
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count:         {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}:           {props.name}")
            print(f"    Total Memory:  {props.total_memory / 1024**3:.2f} GB")
            print(f"    CUDA Version:  {torch.version.cuda}")
    else:
        print("  No GPU found. Benchmarks will run on CPU only.")

    # --- Benchmarks ---
    print("\n--- Benchmark Results ---")
    
    # GPU Benchmarks (if available)
    if cuda_ok:
        device_gpu = torch.device('cuda')
        print(f"\nRunning on GPU: {torch.cuda.get_device_name(0)}")
        benchmark(device_gpu, torch.float32, size=args.size, repeats=args.repeats)
        benchmark(device_gpu, torch.float16, size=args.size, repeats=args.repeats)
    
    # CPU Benchmark for comparison
    device_cpu = torch.device('cpu')
    print(f"\nRunning on CPU")
    benchmark(device_cpu, torch.float32, size=args.size, repeats=args.repeats)
    
    print("\n" + "="*60)
    print("Benchmark finished.")
    print("="*60)

if __name__ == "__main__":
    main()