#!/usr/bin/env python3
"""
Check the PyTorch installation and available compute devices.

This script is a lightweight environment test for users who want to run the
3D CNN workflow in this repository. It reports the Python and PyTorch versions,
detects CPU, NVIDIA CUDA, and Apple MPS devices, verifies a small forward and
backward tensor operation, and optionally benchmarks matrix multiplication.

The test does not read the zeolite dataset or train a model. It can be run on a
local computer or through ``test_pytorch.sh`` on a SLURM cluster.

Examples
--------
Run the default test on the CPU and every available accelerator::

    python test_pytorch.py

Test a specific CUDA device with a larger benchmark::

    python test_pytorch.py --device cuda --size 4096 --repeats 20

Check the installation without running the performance benchmark::

    python test_pytorch.py --skip-benchmark
"""

import argparse
import os
import platform
import socket
import sys
import time

try:
    import torch
except ImportError:
    print(
        "Error: PyTorch is not installed in the active Python environment.\n"
        "Install PyTorch by following https://pytorch.org/get-started/locally/",
        file=sys.stderr,
    )
    sys.exit(1)


def synchronize_device(device):
    """Wait for queued accelerator operations to finish before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_smoke_test(device):
    """
    Verify tensor creation, matrix multiplication, and automatic differentiation.

    Returns True when the operations complete and produce finite values. Any
    device or PyTorch error is reported and results in False.
    """
    try:
        x = torch.randn(32, 32, device=device, dtype=torch.float32, requires_grad=True)
        loss = (x @ x.T).square().mean()
        loss.backward()
        synchronize_device(device)

        passed = torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
        status = "PASSED" if passed else "FAILED (non-finite values)"
        print(f"  {str(device):<8} forward/backward test: {status}")
        return passed
    except Exception as error:
        print(f"  {str(device):<8} forward/backward test: FAILED ({error})")
        return False


def benchmark(device, dtype, size=1024, repeats=10):
    """
    Benchmark repeated square-matrix multiplication on one device.

    Returns True when the benchmark completes successfully and False when the
    requested data type or operation is unsupported.
    """
    try:
        x = torch.randn(size, size, device=device, dtype=dtype)
        y = torch.randn(size, size, device=device, dtype=dtype)

        # Warm up the selected backend before recording elapsed time.
        for _ in range(3):
            _ = x @ y
        synchronize_device(device)

        elapsed_times = []
        for _ in range(repeats):
            start_time = time.perf_counter()
            _ = x @ y
            synchronize_device(device)
            elapsed_times.append((time.perf_counter() - start_time) * 1000)

        mean_ms = sum(elapsed_times) / repeats
        std_ms = (sum((elapsed - mean_ms) ** 2 for elapsed in elapsed_times) / repeats) ** 0.5
        dtype_name = str(dtype).split(".")[-1]

        print(f"  Device: {str(device):<8} Dtype: {dtype_name:<7} Matrix: {size} x {size}")
        print(f"    Average time: {mean_ms:.2f} +/- {std_ms:.2f} ms ({repeats} runs)")
        return True
    except Exception as error:
        print(f"  Device: {str(device):<8} Dtype: {str(dtype).split('.')[-1]:<7} FAILED ({error})")
        return False


def get_available_devices():
    """Return the CPU and all accelerators available to the current PyTorch build."""
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.extend(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count()))

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        devices.append(torch.device("mps"))

    return devices


def select_devices(requested_device):
    """Select requested devices and raise a clear error when one is unavailable."""
    available_devices = get_available_devices()

    if requested_device == "auto":
        return available_devices

    if requested_device == "cpu":
        return [torch.device("cpu")]

    matching_devices = [device for device in available_devices if device.type == requested_device]
    if not matching_devices:
        raise RuntimeError(
            f"Requested device '{requested_device}' is not available in this environment."
        )

    return matching_devices


def print_environment_info():
    """Print software, host, and accelerator information useful for diagnostics."""
    print("=" * 72)
    print("PyTorch Environment Test")
    print("=" * 72)
    print(f"Python executable:  {sys.executable}")
    print(f"Python version:     {platform.python_version()}")
    print(f"PyTorch version:    {torch.__version__}")
    print(f"Operating system:   {platform.platform()}")
    print(f"Hostname:           {socket.gethostname()}")
    print(f"Conda environment:  {os.environ.get('CONDA_DEFAULT_ENV', 'N/A')}")
    print(f"SLURM job ID:       {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"CPU threads:        {torch.get_num_threads()}")
    print(f"CUDA build:         {torch.version.cuda or 'None'}")
    print(f"CUDA available:     {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(f"  cuda:{index}:          {properties.name}")
            print(f"    Total memory:    {properties.total_memory / 1024**3:.2f} GB")

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = mps_backend is not None and mps_backend.is_built()
    mps_available = mps_backend is not None and mps_backend.is_available()
    print(f"MPS built:          {mps_built}")
    print(f"MPS available:      {mps_available}")


def parse_arguments():
    """Parse command-line options for device selection and benchmarking."""
    parser = argparse.ArgumentParser(
        description="Test the PyTorch installation used by the 3D CNN workflow."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to test. 'auto' tests the CPU and all available accelerators (default: auto).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Dimension of each square matrix used for benchmarking (default: 1024).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timed matrix multiplications (default: 10).",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Run environment and forward/backward checks without timing matrix multiplication.",
    )
    args = parser.parse_args()

    if args.size < 1:
        parser.error("--size must be a positive integer")
    if args.repeats < 1:
        parser.error("--repeats must be a positive integer")

    return args


def main():
    """Run the environment report, smoke tests, and optional benchmarks."""
    args = parse_arguments()
    print_environment_info()

    try:
        devices = select_devices(args.device)
    except RuntimeError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1

    print("\nForward and backward operation tests")
    print("-" * 72)
    all_tests_passed = True
    for device in devices:
        all_tests_passed &= run_smoke_test(device)

    if not args.skip_benchmark:
        print("\nMatrix multiplication benchmarks")
        print("-" * 72)
        for device in devices:
            all_tests_passed &= benchmark(
                device, torch.float32, size=args.size, repeats=args.repeats
            )
            if device.type in ("cuda", "mps"):
                all_tests_passed &= benchmark(
                    device, torch.float16, size=args.size, repeats=args.repeats
                )

    print("\n" + "=" * 72)
    if all_tests_passed:
        print("PyTorch environment test PASSED.")
        return 0

    print("PyTorch environment test FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
