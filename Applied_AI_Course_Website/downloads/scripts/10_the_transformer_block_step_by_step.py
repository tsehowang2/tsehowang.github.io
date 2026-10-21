# # Tutorial 10: The transformer block step by step
# 
# **Prerequisite:** lecture 10. **Compute:** CPU; standard library and NumPy.
# 
# Predict the result before each code cell. Run cells in order from a fresh kernel. Tutorials 01–16 run without coursekit. Tutorials 17–32 use the extracted full course pack linked on the tutorial page. Reference outputs are labeled below; keep your own actual results.

import random
from pathlib import Path
import sys, json, math, time, random
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
COURSE = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RUNS = COURSE / 'runs'
RUNS.mkdir(exist_ok=True)
random.seed(7)
print('Outputs directory:', RUNS)


# ### Pause before the next step
# Draw h=x+Attention(LN(x)), followed by y=h+MLP(LN(h)). State the shape at each addition.

# The block alternates communication across tokens and feature processing at each token. Normalization, residual addition and the MLP can be inspected without training. This cell isolates these paths; week 12 combines them with causal attention.

import numpy as np
rng = np.random.default_rng(7)
x = rng.normal(size=(6, 4))

def ln(a):
    return (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-05)
h = ln(x)
print('Row means', h.mean(-1), 'row variances', h.var(-1))
W1 = rng.normal(size=(4, 8))
W2 = rng.normal(size=(8, 4))
mlp = np.maximum(h @ W1, 0) @ W2
y = x + mlp
assert y.shape == x.shape
changed = h.copy()
changed[2] += 1
delta = np.maximum(changed @ W1, 0) @ W2 - mlp
assert np.allclose(np.delete(delta, 2, axis=0), 0)
print('MLP is positionwise; a changed row did not change the other rows.')


# ### Count the specified architecture
# Ignore biases and normalization parameters. Keep d_ff=4*d_model fixed to expose the extra gated-MLP matrix. An implementation may choose a different d_ff to match its parameter budget.

d_model = 64
H = 8
H_kv = 2
d_head = d_model // H
d_ff = 4 * d_model
mha = 4 * d_model * d_model
gqa = 2 * d_model * d_model + 2 * d_model * H_kv * d_head
plain_mlp = 2 * d_model * d_ff
gated_mlp = 3 * d_model * d_ff
assert mha + plain_mlp == 49152 and gqa < mha and (gated_mlp > plain_mlp)
print('MHA, GQA:', mha, gqa, 'Plain, gated MLP:', plain_mlp, gated_mlp)
print('MHA plus plain MLP:', mha + plain_mlp, 'GQA plus gated MLP:', gqa + gated_mlp)


# ### Independent experiment
# Set the MLP output matrix to zero. Verify that this residual sublayer becomes the identity. Replace the toy ReLU with a GELU implementation and explain which property tests should remain unchanged.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Row means [-5.55111512e-17 -5.55111512e-17  8.32667268e-17  0.00000000e+00
#   3.05311332e-16  0.00000000e+00] row variances [0.9999481  0.99998662 0.99995906 0.99997048 0.99996235 0.99998554]
# MLP is positionwise; a changed row did not change the other rows.
# MHA, GQA: 16384 10240 Plain, gated MLP: 32768 49152
# MHA plus plain MLP: 49152 GQA plus gated MLP: 59392
# ```
