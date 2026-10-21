# # Tutorial 25: LoRA and QLoRA from a small matrix
# 
# **Prerequisite:** lecture 25. **Compute:** CPU; standard library and NumPy.
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
# Why is low rank not parameter efficient for every tiny matrix? Check the three-by-four rank-two example.

# LoRA adds a low-rank update to a frozen matrix. This tiny example makes the factors and scaling visible. It is not a demonstration that low rank always preserves model quality.

import numpy as np
rng = np.random.default_rng(7)
din, dout, rank = (4, 3, 2)
W = rng.normal(size=(dout, din))
A = rng.normal(size=(rank, din))
B = np.zeros((dout, rank))
x = rng.normal(size=din)
alpha = 4
scale = alpha / rank
assert scale == 2
assert np.allclose(W @ x, (W + scale * B @ A) @ x)
B = rng.normal(size=(dout, rank))
assert np.allclose((W + scale * B @ A) @ x, W @ x + scale * B @ (A @ x))
print('Dense parameters', W.size, 'adapter parameters', A.size + B.size, 'delta rank', np.linalg.matrix_rank(B @ A))
for r in [1, 2, 4, 16]:
    print('2048x4096 rank', r, 'adapter count', r * (2048 + 4096))
print('alpha/r scaling:',alpha,'/',rank,'=',scale)


# ### Check the first LoRA gradient path
# For update BA and gradient G with respect to that update, grad_B=G Aᵀ and grad_A=Bᵀ G. Random A with zero B allows B to learn first. This calculation explains the usual initialization without requiring a full model-training run.

import numpy as np
A = np.array([[0.1, 0.2]])
B = np.zeros((2, 1))
G = np.ones((2, 2))
grad_B = G @ A.T
grad_A = B.T @ G
assert np.any(grad_B != 0) and np.all(grad_A == 0)
B = B - 0.01 * grad_B
assert np.any(B.T @ G != 0)
print('Initial grad_B:', grad_B, 'Initial grad_A:', grad_A)


# ### Independent experiment
# Compare zero B/random A with both factors zero by deriving the first gradients. Explain how QLoRA changes frozen-base storage and why it does not imply all activations or optimizer states are four-bit.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Dense parameters 12 adapter parameters 14 delta rank 2
# 2048x4096 rank 1 adapter count 6144
# 2048x4096 rank 2 adapter count 12288
# 2048x4096 rank 4 adapter count 24576
# 2048x4096 rank 16 adapter count 98304
# Initial grad_B: [[0.3]
#  [0.3]] Initial grad_A: [[0. 0.]]
# ```
