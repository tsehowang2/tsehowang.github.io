# # Tutorial 05: Why language needs context and order
# 
# **Prerequisite:** lecture 05. **Compute:** CPU; standard library and NumPy.
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
# Trace the first and second state updates by hand. Which part carries earlier information?

# A mean of token vectors loses order. A tiny recurrence carries state through the sequence, so changing order changes its result. This is a numerical illustration of context, not a trained recurrent model.

import numpy as np
emb = {'I': np.array([1.0, 0.0]), 'like': np.array([0.0, 1.0]), 'school': np.array([1.0, 1.0])}
a = ['I', 'like', 'school']
b = list(reversed(a))

def mean(words):
    return np.mean([emb[w] for w in words], axis=0)

def recur(words):
    h = np.zeros(2)
    for w in words:
        h = np.tanh(0.5 * h + emb[w])
    return h
assert np.allclose(mean(a), mean(b))
print('Same mean:', mean(a), mean(b), 'Different recurrence:', recur(a), recur(b))
assert not np.allclose(recur(a), recur(b))


# ### Independent experiment
# Move one informative token farther from the final position and record its effect on the final state. Compare recurrence coefficients .1 and .9. Explain why this untrained example cannot establish a general performance claim.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# Same mean: [0.66666667 0.66666667] [0.66666667 0.66666667] Different recurrence: [0.82798683 0.88112963] [0.82798683 0.41411251]
# ```
