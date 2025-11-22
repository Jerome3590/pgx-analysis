#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check system hardware and GPU capabilities"""

import psutil
import platform
import subprocess
import sys

print("="*80)
print("SYSTEM HARDWARE ANALYSIS")
print("="*80)

# CPU Info
print(f"\nCPU Information:")
print(f"  Physical Cores: {psutil.cpu_count(logical=False)}")
print(f"  Logical Cores (with HT): {psutil.cpu_count(logical=True)}")
print(f"  CPU Frequency: {psutil.cpu_freq().max:.0f} MHz" if psutil.cpu_freq() else "  CPU Frequency: N/A")
print(f"  Architecture: {platform.machine()}")
print(f"  Processor: {platform.processor()}")

# Memory Info
mem = psutil.virtual_memory()
print(f"\nMemory Information:")
print(f"  Total RAM: {mem.total / (1024**3):.1f} GB")
print(f"  Available RAM: {mem.available / (1024**3):.1f} GB")
print(f"  Used RAM: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")

# GPU Check
print(f"\nGPU Detection:")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"  NVIDIA GPU detected!")
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}")
    else:
        print(f"  No NVIDIA GPU detected")
except:
    print(f"  nvidia-smi not available (no NVIDIA drivers or non-NVIDIA GPU)")

# Check GPU libraries
print(f"\nGPU-Enabled Libraries:")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
except:
    print(f"  PyTorch: Not installed")

try:
    import cudf
    print(f"  cuDF (RAPIDS): {cudf.__version__}")
except:
    print(f"  cuDF (RAPIDS): Not installed")

# Check FP-Growth libraries
print(f"\nFP-Growth Libraries:")
try:
    import mlxtend
    print(f"  mlxtend: {mlxtend.__version__} (CPU-only)")
except:
    print(f"  mlxtend: Not installed")

# Recommendations
print(f"\nConfiguration Recommendations:")
physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)
available_mem_gb = mem.available / (1024**3)

print(f"  Optimal MAX_WORKERS: {min(physical_cores - 1, 8)} (leaving 1 core for system)")
print(f"  Safe MAX_WORKERS: {max(physical_cores // 2, 2)} (conservative)")
print(f"  Available Memory: {available_mem_gb:.1f} GB")

if available_mem_gb < 8:
    print(f"  WARNING: Low memory - reduce MAX_WORKERS to 2-4")
elif available_mem_gb > 16:
    print(f"  High memory - can use MAX_WORKERS = {min(logical_cores - 2, 12)}")

print(f"\nGPU Acceleration Status:")
print(f"  mlxtend FP-Growth: CPU-only (no GPU support)")
print(f"  Best strategy: Maximize CPU parallelization")
print(f"  Current approach uses ProcessPoolExecutor (optimal for CPU)")

print("="*80)

