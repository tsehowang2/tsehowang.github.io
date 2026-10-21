# # Tutorial 22: CPU GPU memory and model resource accounting
# 
# **Prerequisite:** lecture 22. **Compute:** CPU; no downloads.
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


# ### Pause before the next step
# Explain the factor of two in the KV formula and what doubling the context length changes.

# Hardware planning starts with units. Weight memory, optimizer state, activations and the KV cache are different allocations. These are arithmetic estimates under stated assumptions, not GPU measurements.

parameters = 300000000
for bytes_per_parameter in [2, 4, 16]:
    size = parameters * bytes_per_parameter
    print(bytes_per_parameter, 'bytes/parameter:', size / 2 ** 30, 'GiB')
layers, batch, tokens, H_kv, d_head, bytes_per_element = (24, 1, 2048, 8, 64, 2)
cache = 2 * layers * batch * tokens * H_kv * d_head * bytes_per_element
assert cache / 2 ** 20 == 96
print('Hypothetical KV cache:', cache / 2 ** 20, 'MiB')


# ### Quantized weight storage and a bandwidth ceiling
# The four-bit estimate omits scales, metadata, unquantized tensors, KV cache, activations, and runtime buffers. The bandwidth estimate assumes one full weight read per generated token at batch one and perfect effective bandwidth; measured throughput can be lower and batching changes the model.

parameters = 1000000000
bits_per_weight = 4
packed_bytes = parameters * bits_per_weight / 8
print('Packed weights only, GiB:', packed_bytes / 2 ** 30)
bandwidth_bytes_per_second = 100000000000.0
fp16_bytes = parameters * 2
fp16_ceiling = bandwidth_bytes_per_second / fp16_bytes
ceiling = bandwidth_bytes_per_second / packed_bytes
assert fp16_ceiling == 50 and ceiling == 200
print('FP16 ceiling:', fp16_ceiling, 'tokens/s')
print('Idealized batch-one bandwidth ceiling, tokens/s:', ceiling)


# ### Independent experiment
# Inventory the actual CPU/GPU and free memory. Build a small-model budget with explicit precision assumptions and a safety margin. Distinguish measured allocations from estimates; explain what gradient accumulation can and cannot reduce.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# 2 bytes/parameter: 0.5587935447692871 GiB
# 4 bytes/parameter: 1.1175870895385742 GiB
# 16 bytes/parameter: 4.470348358154297 GiB
# Hypothetical KV cache: 96.0 MiB
# Packed weights only, GiB: 0.46566128730773926
# FP16 ceiling: 50.0 tokens/s
# Idealized batch-one bandwidth ceiling, tokens/s: 200.0
# ```
