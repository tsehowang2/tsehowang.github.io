# # Tutorial 23: CUDA kernels and profiling
# 
# **Prerequisite:** lecture 23. **Compute:** Colab GPU; CUDA kernel also requires nvcc.
# 
# Predict the result before each code cell. Run cells in order from a fresh kernel. Tutorials 01–16 run without coursekit. Tutorials 17–32 use the extracted full course pack linked on the tutorial page. Reference outputs are labeled below; keep your own actual results.

import random
from pathlib import Path
import sys, json, time, random, hashlib, math
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / 'llm_course', Path.cwd() / 'llm_course_v2', Path('/content/llm_course'), Path('/content/llm_course_v2')]
COURSE = next((p for p in candidates if (p / 'coursekit/core.py').exists()), None)
if COURSE is None:
    raise RuntimeError('Extract the complete course pack; run from its root or notebooks directory.')
if str(COURSE) not in sys.path:
    sys.path.insert(0, str(COURSE))
RUNS = COURSE / 'runs'
RUNS.mkdir(exist_ok=True)
from coursekit.core import *
random.seed(7)
print('Course root:', COURSE)


# Inspect the vector-add kernel before compiling: each thread calculates one index and must check its bound. Predict bounds behavior for n=1, 257, and 1048576. Then compile only if nvcc exists and profile with explicit synchronization.

import sys
import shutil, subprocess
if shutil.which('nvidia-smi'):
    subprocess.run(['nvidia-smi'], check=True)
if shutil.which('nvcc'):
    target = RUNS / ('vector_add.exe' if sys.platform == 'win32' else 'vector_add')
    subprocess.check_call(['nvcc', '-O2', str(COURSE / 'cuda/vector_add.cu'), '-o', str(target)])
    for n in [1, 257, 1048576]:
        subprocess.check_call([str(target), str(n)])
else:
    print('CUDA kernel NOT executed: nvcc unavailable. Schedule a compatible runtime.')


import importlib.util
if importlib.util.find_spec('torch') is None:
    print('PyTorch profiling skipped: install the pinned course environment first.')
else:
    import torch
    from torch.profiler import profile, ProfilerActivity
    print('torch', torch.__version__, 'CUDA runtime', torch.version.cuda)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
        x = torch.randn(1024, 1024, device='cuda')
        y = torch.randn_like(x)
        for _ in range(10):
            z = x @ y
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, stop = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        times = []
        for _ in range(20):
            start.record()
            z = x @ y
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop))
        assert torch.allclose(z.cpu(), x.cpu() @ y.cpu(), atol=0.003, rtol=0.003)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            z = x @ y
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(RUNS / 'week23_trace.json'))
        print(prof.key_averages().table(sort_by='self_device_time_total', row_limit=8))
        print('Median kernel interval ms', sorted(times)[len(times) // 2], 'peak allocated bytes', torch.cuda.max_memory_allocated())
    else:
        print('GPU profiling NOT executed. Memory arithmetic is not a hardware measurement.')


# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# CUDA kernel NOT executed: nvcc unavailable. Schedule a compatible runtime.
# PyTorch profiling skipped: install the pinned course environment first.
# ```
