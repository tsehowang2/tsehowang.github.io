# # Tutorial 08: Scaling and masking attention correctly
# 
# **Prerequisite:** lecture 08. **Compute:** CPU; NumPy.
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


# Padding is distinct from causal visibility.
import numpy as np
q = np.array([1.0, 0.0])

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
keys = np.array([[1.0, 0.0], [0.0, 1.0]])
values = np.array([[2.0, 0.0], [0.0, 4.0]])
weights = softmax((q @ keys.T / np.sqrt(2))[None, :])
out = weights @ values
print('weights', weights, 'output', out)
assert np.allclose(out, [[1.3395231, 1.3209538]], atol=1e-05)
padded_keys = np.vstack([keys, [10.0, 10.0]])
valid = np.array([True, True, False])
scores = q @ padded_keys.T / np.sqrt(q.size)
padded_weights = softmax(np.where(valid, scores, -np.inf)[None, :])
assert padded_weights[0, 2] == 0
assert np.allclose(padded_weights[0, :2], weights[0])
print('Padding weights:', padded_weights.round(4))


# Softmax is over keys. The causal mask includes the current position and forbids future positions. Perturbing a future input tests the behavior independently of mask construction.

def attention(x, wq, wk, wv, causal=True):
    Q, K, V = (x @ wq, x @ wk, x @ wv)
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if causal:
        scores = np.where(np.triu(np.ones(scores.shape, dtype=bool), k=1), -np.inf, scores)
    weights = softmax(scores)
    return (weights @ V, weights)
rng = np.random.default_rng(7)
x = rng.normal(size=(6, 4))
mats = [rng.normal(size=(4, 3)) for _ in range(3)]
changed = x.copy()
changed[-1] += 10
original, _ = attention(x, *mats)
modified, w = attention(changed, *mats)
assert np.allclose(original[:-1], modified[:-1])
unmasked, _ = attention(x, *mats, causal=False)
changed_unmasked, _ = attention(changed, *mats, causal=False)
assert not np.allclose(unmasked[:-1], changed_unmasked[:-1])
assert np.allclose(w.sum(axis=-1), 1)
print('Causal invariance passes; unmasked operation fails the same property as expected.')
for T in [128, 512, 2048]:
    print('T', T, 'float32 score MiB per head', T * T * 4 / 2 ** 20)


# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# weights [[0.66976155 0.33023845]] output [[1.3395231 1.3209538]]
# Padding weights: [[0.6698 0.3302 0.    ]]
# Causal invariance passes; unmasked operation fails the same property as expected.
# T 128 float32 score MiB per head 0.0625
# T 512 float32 score MiB per head 1.0
# T 2048 float32 score MiB per head 16.0
# ```
