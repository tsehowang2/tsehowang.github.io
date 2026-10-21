# # Tutorial 09: Multiple attention heads and position
# 
# **Prerequisite:** lecture 09. **Compute:** CPU; standard library and NumPy.
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
# Track the location of feature six at position two through the split and merge.

# A multi-head layer divides projected features into parallel heads. Reshaping does not learn anything. The projection matrices do. Here we inspect the shape transformation, then show how positional vectors distinguish otherwise identical tokens.

import numpy as np
B, T, d_model, H = (1, 6, 8, 2)
d_head = d_model // H
x = np.arange(B * T * d_model).reshape(B, T, d_model)
heads = x.reshape(B, T, H, d_head).transpose(0, 2, 1, 3)
merged = heads.transpose(0, 2, 1, 3).reshape(B, T, d_model)
print('Original', x.shape, 'split', heads.shape, 'merged', merged.shape)
assert np.array_equal(x, merged)
pos = np.arange(T)[:, None]
dims = np.arange(0, d_model, 2)
angles = pos / 10000 ** (dims / d_model)
pe = np.zeros((T, d_model))
pe[:, 0::2] = np.sin(angles)
pe[:, 1::2] = np.cos(angles)
print('Position vectors:', pe[:3])
assert not np.allclose(pe[0], pe[1])


# ### RoPE with different position angles
# Compare angle pairs (30°,90°) and (60°,120°). Both have the same relative offset. This two-coordinate calculation illustrates relative-position dependence; it does not by itself establish long-context reliability.

import math

def rotate(v, theta):
    c, s = (math.cos(theta), math.sin(theta))
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]

def dp(a, b):
    return sum((x * y for x, y in zip(a, b)))
q = k = [1.0, 0.0]
for q_deg, k_deg in [(0, 0), (30, 90), (60, 120)]:
    value = dp(rotate(q, math.radians(q_deg)), rotate(k, math.radians(k_deg)))
    expected = math.cos(math.radians(k_deg - q_deg))
    assert math.isclose(value, expected, abs_tol=1e-12)
    print(q_deg, k_deg, 'relative offset', k_deg - q_deg, 'dot', value)


# ### Independent experiment
# Implement one attention calculation per head, then concatenate. Compare with a single head of dimension eight, using distinct but recorded matrices. Discuss what is an architectural difference and what is merely a layout transformation.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Original (1, 6, 8) split (1, 2, 6, 4) merged (1, 6, 8)
# Position vectors: [[ 0.00000000e+00  1.00000000e+00  0.00000000e+00  1.00000000e+00
#    0.00000000e+00  1.00000000e+00  0.00000000e+00  1.00000000e+00]
#  [ 8.41470985e-01  5.40302306e-01  9.98334166e-02  9.95004165e-01
#    9.99983333e-03  9.99950000e-01  9.99999833e-04  9.99999500e-01]
#  [ 9.09297427e-01 -4.16146837e-01  1.98669331e-01  9.80066578e-01
#    1.99986667e-02  9.99800007e-01  1.99999867e-03  9.99998000e-01]]
# 0 0 relative offset 0 dot 1.0
# 30 90 relative offset 60 dot 0.5
# 60 120 relative offset 60 dot 0.5
# ```
