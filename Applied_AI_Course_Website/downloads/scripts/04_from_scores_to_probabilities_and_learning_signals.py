# # Tutorial 04: From scores to probabilities and learning signals
# 
# **Prerequisite:** lecture 04. **Compute:** CPU; standard library and NumPy.
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
# Before running, decide whether the train score should increase or decrease.

# Scores can be any real numbers. Softmax produces positive probabilities that sum to one. Cross entropy asks how much probability the model gave the observed target. A numerical derivative will check the analytic gradient.

import numpy as np

def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()
z = np.array([2.0, 1.0, 0.0])
target = 1
p = softmax(z)
loss = -np.log(p[target])
gradient = p.copy()
gradient[target] -= 1
print('bus/train/bicycle probabilities', p, 'train loss', loss, 'gradient', gradient)
eps = 1e-05
numerical = []
for i in range(3):
    d = np.zeros(3)
    d[i] = eps
    numerical.append((-np.log(softmax(z + d)[target]) + np.log(softmax(z - d)[target])) / (2 * eps))
assert np.allclose(gradient, numerical, atol=1e-07)
assert np.allclose(softmax(z + 10000), p)
print('After a small gradient step:', softmax(z - 0.1 * gradient))


# ### Independent experiment
# Compare temperature .5, 1 and 2 on fixed logits. Report entropy and target loss separately. Explain why a sharper distribution can increase the loss for a non-maximal target.
# 
# Submit a short report containing your prediction, the actual result, a comparison and a limitation. Record any environment or data changes. Do not substitute an illustrative number for a measured result.

# ### Reference output
# Reference output from the CPU teaching run (paths and timings vary). This is a comparison aid, not your experiment result.
# 
# ```text
# bus/train/bicycle probabilities [0.66524096 0.24472847 0.09003057] train loss 1.4076059644443804 gradient [ 0.66524096 -0.75527153  0.09003057]
# After a small gradient step: [0.63800773 0.27053504 0.09145723]
# ```
