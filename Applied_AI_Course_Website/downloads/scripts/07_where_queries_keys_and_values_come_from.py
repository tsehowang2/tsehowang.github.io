# # Tutorial 07: Where queries keys and values come from
# 
# **Prerequisite:** lecture 07. **Compute:** CPU; standard library and NumPy.
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
# Calculate one projection and one dot product by hand before checking the arrays.

# We now generate weights from a query and keys. Use one vector for what is sought, one key per candidate, and one value per candidate. The numerical projections below are invented and deliberately tiny.

import numpy as np
x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Wq = np.array([[1.0, 0.0], [0.0, 1.0]])
Wk = np.array([[1.0, 1.0], [0.0, 1.0]])
Wv = np.array([[2.0, 0.0], [0.0, 4.0]])
Q, K, V = (x @ Wq, x @ Wk, x @ Wv)
q = Q[2]
scores = q @ K.T
e = np.exp(scores - scores.max())
weights = e / e.sum()
print('Q', Q, 'K', K, 'V', V, 'scores', scores, 'weights', weights, 'output', weights @ V, sep='\n')
assert scores.shape == (3,)
assert (weights @ V).shape == (2,)


# ### Independent experiment
# Change Wv while leaving Wq and Wk fixed. Show that the weights stay fixed while the output changes. Then change Wk. Explain why the same operation cannot describe both changes.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Q
# [[1. 0.]
#  [0. 1.]
#  [1. 1.]]
# K
# [[1. 1.]
#  [0. 1.]
#  [1. 2.]]
# V
# [[2. 0.]
#  [0. 4.]
#  [2. 4.]]
# scores
# [2. 1. 3.]
# weights
# [0.24472847 0.09003057 0.66524096]
# output
# [1.81993885 3.02108612]
# ```
